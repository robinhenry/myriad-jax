import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from myriad.configs.default import Config
from myriad.core.replay_buffer import ReplayBuffer
from myriad.utils import to_array

from .episode_manager import save_episodes_to_disk
from .initialization import initialize_environment_and_agent
from .logging_utils import maybe_init_wandb, wandb
from .metrics_logger import MetricsLogger
from .scan_utils import (
    make_chunk_runner,
    make_chunked_collector,
    make_on_policy_chunk_runner,
)
from .shared import TrainingEnvState
from .step_functions import (
    make_collection_step_fn,
    make_eval_rollout_fn,
    make_sample_transition,
    make_train_step_fn,
)
from .types import TrainingResults

logger = logging.getLogger(__name__)


def _run_training_loop(config: Config, wandb_run: Any) -> TrainingResults:
    """Executes the training loop and returns metrics + trained agent.

    Returns:
        TrainingResults containing trained agent state, training metrics history,
        evaluation metrics history, and configuration.
    """

    # Initialize everything
    key = jax.random.PRNGKey(config.run.seed)
    key, env_key, agent_key, buffer_key = jax.random.split(key, 4)

    # Create environment and agent using shared initialization
    env, agent, action_space = initialize_environment_and_agent(config)

    # Initialize parallel environments
    env_keys = jax.random.split(env_key, config.run.num_envs)
    obs, env_states = jax.vmap(env.reset, in_axes=(0, None, None))(env_keys, env.params, env.config)
    # Convert observations to arrays for platform (lean approach: single conversion point)
    obs_array = jax.vmap(to_array)(obs)
    training_env_states = TrainingEnvState(env_state=env_states, obs=obs_array)

    # Initialize agent using the initial observation from one environment
    # Use original NamedTuple observation (not converted array) to allow field introspection
    # Extract first element from batched NamedTuple using tree.map
    sample_obs = jax.tree.map(lambda x: x[0], obs)
    agent_state = agent.init(agent_key, sample_obs, agent.params)

    # Determine training mode and initialize accordingly
    use_rollout_training = config.run.rollout_steps is not None

    if use_rollout_training:
        # On-policy training (e.g., PPO, A2C, PQN): no replay buffer needed
        replay_buffer = None
        buffer_state = None
        assert config.run.rollout_steps is not None  # should always be true if use_rollout_training is true

        # Create chunked collector for efficient rollout collection
        collection_step_fn = make_collection_step_fn(agent, env, config.run.num_envs)
        rollout_fn = make_chunked_collector(
            collection_step_fn=collection_step_fn,
            num_envs=config.run.num_envs,
            chunk_size=config.run.scan_chunk_size,
            total_steps=config.run.rollout_steps,
        )
    else:
        # Off-policy training (e.g., DQN): use replay buffer
        if config.run.buffer_size is None:
            raise ValueError("buffer_size must be set in config for off-policy training (when rollout_steps is None)")
        replay_buffer = ReplayBuffer(buffer_size=config.run.buffer_size)
        # Convert sample_obs to array for buffer initialization (matches training transition structure)
        sample_obs_array = to_array(sample_obs)
        sample_transition = make_sample_transition(buffer_key, sample_obs_array, action_space)
        buffer_state = replay_buffer.init(sample_transition)
        rollout_fn = None

    # Build jitted execution primitives
    eval_rollout_fn = make_eval_rollout_fn(agent, env, config.run.eval_rollouts, config.run.eval_max_steps)

    # Chunking configuration:
    # - scan_chunk_size controls how many rollout-update cycles (on-policy) or training steps (off-policy)
    #   are batched into a single jax.lax.scan
    # - Larger chunks reduce Python overhead but increase XLA compile time
    # - chunk_size is ensured to be at least 1 to prevent errors
    chunk_size = max(1, config.run.scan_chunk_size)

    if use_rollout_training:
        # On-policy: create chunk runner that batches multiple rollout-update cycles
        # This avoids returning to Python after each rollout, addressing the regression
        # from the previous platform where everything was in a single jitted scan
        run_chunk_fn = make_on_policy_chunk_runner(
            rollout_fn=rollout_fn,
            agent=agent,
            chunk_size=chunk_size,
            rollout_steps=config.run.rollout_steps,
        )
    else:
        # Off-policy: create chunk runner that batches multiple step-update cycles
        train_step_fn = make_train_step_fn(agent, env, replay_buffer, config.run.num_envs)
        batch_size = config.agent.batch_size if config.agent.batch_size is not None else 1
        run_chunk_fn = make_chunk_runner(train_step_fn, batch_size)

    # Training runs for steps_per_env steps in each environment
    steps_per_env = config.run.steps_per_env
    log_frequency = config.run.log_frequency
    eval_frequency = config.run.eval_frequency

    # Initialize unified metrics logger
    metrics_logger = MetricsLogger(wandb_run=wandb_run)

    # Initialize progress bar for training loop
    steps_completed = 0
    pbar = tqdm(
        total=steps_per_env,
        desc="Training",
        unit="steps",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    # Track metrics for progress bar display
    pbar_metrics = {}

    while steps_completed < steps_per_env:
        remaining_steps = steps_per_env - steps_completed

        # Helper function for boundary alignment
        def _steps_until_boundary(current_step: int, frequency: int) -> int:
            """Calculate steps until next logging/eval boundary.

            This helper ensures chunks align with logging and evaluation frequencies,
            preventing partial metrics from being logged.

            Args:
                current_step: The current training step counter
                frequency: The logging or eval frequency (0 or negative means disabled)

            Returns:
                Number of steps until the next boundary
            """
            if frequency <= 0:
                return chunk_size
            remainder = current_step % frequency
            result = frequency if remainder == 0 else frequency - remainder
            return result

        # Unified chunked training for both on-policy and off-policy
        # Boundary alignment:
        # Determine how many steps/cycles to run before the next logging or eval boundary.
        # For on-policy: steps = rollout-update cycles
        # For off-policy: steps = individual training steps
        steps_to_log = _steps_until_boundary(steps_completed, log_frequency)
        steps_to_eval = _steps_until_boundary(steps_completed, eval_frequency)

        if use_rollout_training:
            # On-policy: Calculate number of rollout-update cycles to run
            # Each cycle advances by rollout_steps, so we need to scale boundaries
            assert config.run.rollout_steps is not None
            cycles_to_log = steps_to_log // config.run.rollout_steps if steps_to_log > 0 else chunk_size
            cycles_to_eval = steps_to_eval // config.run.rollout_steps if steps_to_eval > 0 else chunk_size
            cycles_remaining = (remaining_steps + config.run.rollout_steps - 1) // config.run.rollout_steps
            num_cycles = min(chunk_size, cycles_remaining, cycles_to_log, cycles_to_eval)

            # Ensure we run at least one cycle if there are remaining steps
            num_cycles = max(1, num_cycles) if remaining_steps > 0 else 0

            # Create active mask for cycles
            active_mask = (jnp.arange(chunk_size) < num_cycles).astype(jnp.bool_)

            # Run chunked on-policy training
            (key, agent_state, training_env_states), metrics_history = run_chunk_fn(
                (key, agent_state, training_env_states),
                active_mask,
            )

            # Calculate actual steps completed (num_cycles * rollout_steps, capped at remaining)
            steps_this_chunk = min(num_cycles * config.run.rollout_steps, remaining_steps)
        else:
            # Off-policy: Calculate number of individual training steps to run
            steps_this_chunk = min(chunk_size, remaining_steps, steps_to_log, steps_to_eval)

            # Create a boolean mask for the scan:
            # - active_mask always has length chunk_size (for consistent JIT compilation)
            # - Only the first steps_this_chunk elements are True
            # - Inactive iterations (False elements) execute but don't update state
            active_mask = (jnp.arange(chunk_size) < steps_this_chunk).astype(jnp.bool_)

            # Run chunked off-policy training
            (key, agent_state, training_env_states, buffer_state), metrics_history = run_chunk_fn(
                (key, agent_state, training_env_states, buffer_state),
                active_mask,
            )

        steps_completed += steps_this_chunk
        global_step = steps_completed * config.run.num_envs

        # Update progress bar
        pbar.update(steps_this_chunk)

        # Extract latest metrics for progress bar display
        try:
            # Get the most recent metrics from the history (last value in each array)
            if "loss" in metrics_history:
                latest_loss = float(jax.device_get(metrics_history["loss"][-1]))
                pbar_metrics["loss"] = f"{latest_loss:.3f}"
            if "reward" in metrics_history:
                latest_reward = float(jax.device_get(metrics_history["reward"][-1]))
                pbar_metrics["reward"] = f"{latest_reward:.2f}"
            if pbar_metrics:
                pbar.set_postfix(pbar_metrics, refresh=False)
        except (KeyError, IndexError, TypeError):
            # Metrics might not be available in first iteration or for some agents
            pass

        # Log training metrics (handles both local capture and W&B)
        should_log = steps_completed % log_frequency == 0
        if should_log:
            metrics_logger.log_training_step(
                global_step=global_step,
                steps_per_env=steps_completed,
                metrics_history=metrics_history,
                steps_this_chunk=steps_this_chunk,
            )

        # Periodically run evaluation rollouts without touching the training buffer
        should_eval = eval_frequency > 0 and steps_completed > 0 and steps_completed % eval_frequency == 0
        if should_eval:
            # Determine if we should save episodes this cycle
            should_save_episodes = (
                config.run.eval_episode_save_frequency > 0
                and steps_completed % config.run.eval_episode_save_frequency == 0
            )

            # Run eval once with appropriate flag (efficient: no double evaluation)
            key, eval_key = jax.random.split(key)
            eval_key, eval_results_jax = eval_rollout_fn(eval_key, agent_state, return_episodes=should_save_episodes)
            key = eval_key

            # Convert to host (handle nested episodes dict if present)
            eval_results_host = {}
            for name, value in eval_results_jax.items():
                if name == "episodes":
                    eval_results_host[name] = {k: jax.device_get(v) for k, v in value.items()}
                else:
                    eval_results_host[name] = jax.device_get(value)

            # Log metrics (always)
            metrics_logger.log_evaluation(
                global_step=global_step, steps_per_env=steps_completed, eval_results=eval_results_host
            )

            # Update progress bar with evaluation results
            if "episode_return" in eval_results_host:
                mean_return = float(np.mean(eval_results_host["episode_return"]))
                pbar_metrics["eval_return"] = f"{mean_return:.2f}"
                pbar.set_postfix(pbar_metrics, refresh=False)

            # Save episodes to disk and log to W&B if collected
            if should_save_episodes and "episodes" in eval_results_host:
                save_count = config.run.eval_episode_save_count or config.run.eval_rollouts
                episode_dir = save_episodes_to_disk(eval_results_host, global_step, save_count, config)
                # Log to W&B if successful
                if episode_dir is not None:
                    metrics_logger.log_episodes(episode_dir, global_step)

    # Always log the final step if it wasn't just logged
    # This ensures training_metrics.global_steps[-1] reflects actual completion
    total_env_steps = steps_completed * config.run.num_envs
    if steps_completed % log_frequency != 0:
        metrics_logger.log_training_step(
            global_step=total_env_steps,
            steps_per_env=steps_completed,
            metrics_history=metrics_history,
            steps_this_chunk=steps_this_chunk,
        )

    metrics_logger.log_final(total_env_steps)

    # Close progress bar
    pbar.close()

    # Get captured metrics and return complete results
    training_metrics, eval_metrics = metrics_logger.get_results()

    return TrainingResults(
        agent_state=agent_state,
        training_metrics=training_metrics,
        eval_metrics=eval_metrics,
        config=config,
        final_env_state=training_env_states,
    )


def train_and_evaluate(config: Config) -> TrainingResults:
    """
    Main entry point for a training run.
    Initializes everything and runs the outer training loop.

    Args:
        config: Training configuration specifying environment, agent, and run parameters.

    Returns:
        TrainingResults containing:
        - agent_state: Trained agent (ready for inference)
        - training_metrics: Training history (loss, reward, etc.)
        - eval_metrics: Evaluation history (episode returns, lengths)
        - config: Configuration used (for reproducibility)
        - final_env_state: Final environment states (can be used to resume training)
    """
    wandb_run = maybe_init_wandb(config)

    try:
        results = _run_training_loop(config, wandb_run)
        return results
    finally:
        if wandb_run is not None and wandb is not None:
            wandb.finish()
