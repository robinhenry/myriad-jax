from functools import partial
from pathlib import Path
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from myriad.agents import make_agent
from myriad.agents.agent import Agent, AgentState
from myriad.configs.default import Config, EvalConfig
from myriad.core.replay_buffer import ReplayBuffer, ReplayBufferState
from myriad.core.spaces import Space
from myriad.core.types import BaseModel, Transition
from myriad.envs import make_env
from myriad.envs.environment import Environment
from myriad.utils import to_array

from .logging_utils import maybe_init_wandb, wandb
from .metrics_logger import MetricsLogger
from .scan_utils import (
    make_chunk_runner,
    make_chunked_collector,
    make_on_policy_chunk_runner,
    mask_tree,
    where_mask,
)
from .shared import TrainingEnvState
from .types import EvaluationResults, TrainingResults


def _make_train_step_fn(
    agent: Agent,
    env: Environment,
    replay_buffer: ReplayBuffer | None,
    num_envs: int,
) -> Callable:
    """Factory to create a jitted, vmapped training step function.

    Observation Handling Design (Lean Approach)
    -------------------------------------------
    The platform converts observations to arrays immediately after env.step() and env.reset() calls.
    This ensures all platform utilities (where_mask, mask_tree) operate on pure arrays, maximizing
    throughput with zero overhead in the hot path.

    Why convert to arrays:
    - Environments may return structured observations (e.g., NamedTuples like PhysicsState)
    - Platform utilities require homogeneous arrays for efficient JAX operations
    - Converting once per step (vectorized) is faster than checking types in every utility

    Trade-offs:
    - Memory: Negligible (e.g., 320 KB for 10k envs with 4D obs)
    - Performance: Single vectorized conversion vs repeated type checks
    - Agents: Can still use to_array() utility to handle either format in their own code
    """

    # Vmap the environment step and reset functions so we drive all envs in lockstep
    vmapped_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    vmapped_env_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    # Vectorized observation conversion ensures platform always operates on arrays
    to_array_batch = jax.vmap(to_array)

    @partial(jax.jit, static_argnames=["batch_size"])
    def train_step(
        key: chex.PRNGKey,
        agent_state: AgentState,
        training_env_states: TrainingEnvState,
        buffer_state: ReplayBufferState | None,
        batch_size: int,
    ) -> tuple[chex.PRNGKey, AgentState, TrainingEnvState, ReplayBufferState | None, dict]:
        """Executes one step of training across all parallel environments. This function is pure and jitted."""

        last_obs = training_env_states.obs

        # Select actions using per-env keys while sharing agent state and params
        key, action_key = jax.random.split(key)
        action_keys = jax.random.split(action_key, num_envs)
        # Keep a shared agent_state while still batching per-env actions by using `out_axes=(0, None)`
        actions, agent_state = jax.vmap(
            agent.select_action,
            in_axes=(0, 0, None, None, None),
            out_axes=(0, None),
        )(action_keys, last_obs, agent_state, agent.params, False)

        # Step environments in parallel and capture the resulting transitions
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, next_env_states, rewards, dones, _ = vmapped_env_step(
            step_keys, training_env_states.env_state, actions, env.params, env.config
        )
        # Convert observations to arrays for platform utilities (zero overhead in hot path)
        next_obs_array = to_array_batch(next_obs)
        dones_bool = dones.astype(jnp.bool_)

        # Create transition for this step
        transitions = Transition(last_obs, actions, rewards, next_obs_array, dones_bool)

        # Handle replay buffer if present (off-policy algorithms like DQN)
        if replay_buffer is not None and buffer_state is not None:
            key, buffer_key = jax.random.split(key)
            buffer_state, batch = replay_buffer.add_and_sample(buffer_state, transitions, batch_size, buffer_key)
        else:
            # For on-policy algorithms, use the current transition directly
            batch = transitions

        # Update the agent with the sampled batch and report metrics
        key, update_key = jax.random.split(key)
        agent_state, metrics = agent.update(update_key, agent_state, batch, agent.params)

        # Handle auto-resetting environments that are done by splitting keys once more
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_envs)

        # Reset only the environments that are done
        new_obs, new_env_states = vmapped_env_reset(reset_keys, env.params, env.config)
        # Convert reset observations to arrays
        new_obs_array = to_array_batch(new_obs)

        # If done, use the new state, otherwise keep the existing one (pure array operations)
        final_obs = where_mask(dones_bool, new_obs_array, next_obs_array)
        final_env_states = mask_tree(dones_bool, new_env_states, next_env_states)

        # Store the final states and observations as the new training environment state
        new_training_env_state = TrainingEnvState(env_state=final_env_states, obs=final_obs)

        return (
            key,
            agent_state,
            new_training_env_state,
            buffer_state,
            metrics,
        )

    return train_step


def _make_collection_step_fn(
    agent: Agent,
    env: Environment,
    num_envs: int,
) -> Callable:
    """Factory to create a single-step collection function for on-policy algorithms.

    This creates a step function that collects one transition without performing agent updates.
    It's designed to be used with make_chunked_collector for efficient rollout collection.
    """
    # Vmap the environment step and reset functions
    vmapped_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    vmapped_env_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    # Vectorized observation conversion ensures platform always operates on arrays
    to_array_batch = jax.vmap(to_array)

    def collection_step(
        key: chex.PRNGKey,
        agent_state: AgentState,
        training_env_states: TrainingEnvState,
    ) -> tuple[tuple[chex.PRNGKey, AgentState, TrainingEnvState], Transition]:
        """Execute one step of rollout collection: select action, step env, collect transition."""
        last_obs = training_env_states.obs

        # Select actions using per-env keys
        key, action_key = jax.random.split(key)
        action_keys = jax.random.split(action_key, num_envs)
        actions, agent_state = jax.vmap(
            agent.select_action,
            in_axes=(0, 0, None, None, None),
            out_axes=(0, None),
        )(action_keys, last_obs, agent_state, agent.params, False)

        # Step environments in parallel
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, next_env_states, rewards, dones, _ = vmapped_env_step(
            step_keys, training_env_states.env_state, actions, env.params, env.config
        )
        # Convert observations to arrays for platform utilities
        next_obs_array = to_array_batch(next_obs)
        dones_bool = dones.astype(jnp.bool_)

        # Handle auto-reset for completed episodes
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_envs)
        new_obs, new_env_states = vmapped_env_reset(reset_keys, env.params, env.config)
        # Convert reset observations to arrays
        new_obs_array = to_array_batch(new_obs)

        final_obs = where_mask(dones_bool, new_obs_array, next_obs_array)
        final_env_states = mask_tree(dones_bool, new_env_states, next_env_states)
        new_training_env_state = TrainingEnvState(env_state=final_env_states, obs=final_obs)

        # Create transition for this step
        transition = Transition(last_obs, actions, rewards, next_obs_array, dones_bool)

        return (key, agent_state, new_training_env_state), transition

    return collection_step


def _get_eval_settings(config: Config | EvalConfig) -> tuple[int, int, int]:
    """Extract evaluation settings from either Config or EvalConfig.

    Args:
        config: Either a training Config or an EvalConfig

    Returns:
        Tuple of (seed, eval_rollouts, eval_max_steps)
    """
    if isinstance(config, EvalConfig):
        return config.seed, config.eval_rollouts, config.eval_max_steps
    else:
        return config.run.seed, config.run.eval_rollouts, config.run.eval_max_steps


def _make_eval_rollout_fn(agent: Agent, env: Environment, eval_rollouts: int, eval_max_steps: int) -> Callable:
    """Factory to create a jitted evaluation rollout aligned with the training loop style.

    Design Note: Dynamic vs Static Control Flow
    --------------------------------------------
    This evaluation function uses jax.lax.while_loop (dynamic control flow) rather than the
    fixed-size masked scans used in training. This is an intentional design choice:

    Why while_loop for evaluation:
    1. **Early termination benefit**: Episodes can finish at different times. Using while_loop
       allows us to stop as soon as all episodes complete, avoiding wasted computation.
    2. **Infrequent execution**: Evaluation happens much less frequently than training steps
       (e.g., every 10k-100k steps), so the compilation overhead is negligible.
    3. **Variable episode lengths**: Some environments have highly variable episode durations.
       Early exit can save significant computation when episodes finish quickly.
    4. **Accurate metrics**: We need to track exact episode returns and lengths without
       padding artifacts that would occur with masked iterations.

    Why fixed-size scans for training:
    1. **Frequent execution**: Training steps run continuously, so avoiding recompilation is critical.
    2. **Predictable boundaries**: Logging and eval frequencies create natural boundaries.
    3. **Batch processing**: Training benefits from fixed batch sizes for stability.

    The compilation cost of while_loop is amortized over many training steps between evaluations,
    and the performance benefit from early termination outweighs this cost.

    Args:
        agent: The agent to evaluate
        env: The environment to evaluate in
        eval_rollouts: Number of evaluation episodes to run
        eval_max_steps: Maximum steps per episode

    Returns:
        Callable that performs evaluation rollouts. The returned function accepts:
        - key: PRNG key
        - agent_state: Agent state
        - return_episodes: If True, include full trajectories in results (default: False)
    """
    vmapped_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    vmapped_env_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    # Vectorized observation conversion ensures platform always operates on arrays
    to_array_batch = jax.vmap(to_array)
    num_eval_envs = eval_rollouts
    max_eval_steps = eval_max_steps

    @partial(jax.jit, static_argnames=["return_episodes"])
    def eval_rollout(
        key: chex.PRNGKey, agent_state: AgentState, return_episodes: bool = False
    ) -> tuple[chex.PRNGKey, dict[str, chex.Array]]:
        # Reset evaluation environments
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_eval_envs)
        obs, env_states = vmapped_env_reset(reset_keys, env.params, env.config)
        # Convert observations to arrays
        obs_array = to_array_batch(obs)
        eval_env_state = TrainingEnvState(env_state=env_states, obs=obs_array)

        # Initialize metric accumulators
        episode_returns = jnp.zeros((num_eval_envs,), dtype=jnp.float32)
        episode_lengths = jnp.zeros((num_eval_envs,), dtype=jnp.int32)
        dones = jnp.zeros((num_eval_envs,), dtype=bool)
        max_steps = jnp.asarray(max_eval_steps, dtype=jnp.int32)

        # Initialize episode data collectors if requested
        if return_episodes:
            # Get a sample action to determine shape
            sample_action = agent.select_action(jax.random.PRNGKey(0), obs_array[0], agent_state, agent.params, False)[
                0
            ]

            # Pre-allocate arrays for collecting full trajectories
            # Shape: (num_eval_envs, max_eval_steps, ...)
            episode_obs = jnp.zeros((num_eval_envs, max_eval_steps, *obs_array.shape[1:]), dtype=obs_array.dtype)
            episode_actions = jnp.zeros(
                (num_eval_envs, max_eval_steps, *sample_action.shape), dtype=sample_action.dtype
            )
            episode_rewards = jnp.zeros((num_eval_envs, max_eval_steps), dtype=jnp.float32)
            episode_dones = jnp.zeros((num_eval_envs, max_eval_steps), dtype=bool)
        else:
            # Use None as placeholders when not collecting episodes
            episode_obs = episode_actions = episode_rewards = episode_dones = None

        def cond_fun(carry: tuple) -> chex.Array:
            if return_episodes:
                _, _, _, _, dones, step, _, _, _, _ = carry
            else:
                _, _, _, _, dones, step = carry
            continue_steps = step < max_steps
            incomplete = jnp.logical_not(jnp.all(dones))
            # Exit early once every evaluation episode terminates
            return jnp.logical_and(continue_steps, incomplete)

        def body_fun(carry: tuple) -> tuple:
            if return_episodes:
                key, env_state, returns, lengths, dones, step, ep_obs, ep_actions, ep_rewards, ep_dones = carry
            else:
                key, env_state, returns, lengths, dones, step = carry

            # Drive each evaluation environment independently but under one loop
            key, action_key, step_key = jax.random.split(key, 3)
            action_keys = jax.random.split(action_key, num_eval_envs)
            actions, _ = jax.vmap(agent.select_action, in_axes=(0, 0, None, None, None))(
                action_keys, env_state.obs, agent_state, agent.params, True
            )

            step_keys = jax.random.split(step_key, num_eval_envs)
            next_obs, next_env_states, rewards, step_dones, _ = vmapped_env_step(
                step_keys, env_state.env_state, actions, env.params, env.config
            )
            # Convert observations to arrays
            next_obs_array = to_array_batch(next_obs)

            step_dones = step_dones.astype(jnp.bool_)
            active = jnp.logical_not(dones)
            active_f32 = active.astype(rewards.dtype)

            returns = returns + rewards * active_f32
            lengths = lengths + active.astype(lengths.dtype)

            # Store episode data if collecting trajectories
            if return_episodes:
                # Store data at current step index for each environment
                ep_obs = ep_obs.at[:, step].set(env_state.obs)
                ep_actions = ep_actions.at[:, step].set(actions)
                ep_rewards = ep_rewards.at[:, step].set(rewards)
                ep_dones = ep_dones.at[:, step].set(step_dones)

            env_state = TrainingEnvState(
                env_state=mask_tree(active, next_env_states, env_state.env_state),
                obs=where_mask(active, next_obs_array, env_state.obs),
            )

            dones = jnp.logical_or(dones, step_dones)
            step = step + jnp.array(1, dtype=step.dtype)

            if return_episodes:
                return key, env_state, returns, lengths, dones, step, ep_obs, ep_actions, ep_rewards, ep_dones
            else:
                return key, env_state, returns, lengths, dones, step

        # Run loop with early termination
        initial_step = jnp.array(0, dtype=jnp.int32)
        if return_episodes:
            initial_carry = (
                key,
                eval_env_state,
                episode_returns,
                episode_lengths,
                dones,
                initial_step,
                episode_obs,
                episode_actions,
                episode_rewards,
                episode_dones,
            )
        else:
            initial_carry = (key, eval_env_state, episode_returns, episode_lengths, dones, initial_step)

        final_carry = jax.lax.while_loop(cond_fun, body_fun, initial_carry)

        if return_episodes:
            (
                key,
                _,
                final_returns,
                final_lengths,
                final_dones,
                _,
                final_obs,
                final_actions,
                final_rewards,
                final_dones_ep,
            ) = final_carry
        else:
            key, _, final_returns, final_lengths, final_dones, _ = final_carry

        # Package metrics for the caller
        metrics = {
            "episode_return": final_returns,
            "episode_length": final_lengths,
            "dones": final_dones,
        }

        # Add episode data if requested
        if return_episodes:
            metrics["episodes"] = {
                "observations": final_obs,
                "actions": final_actions,
                "rewards": final_rewards,
                "dones": final_dones_ep,
            }

        return key, metrics

    return eval_rollout


def _save_episodes_to_disk(
    episode_data: dict[str, Any],
    global_step: int,
    save_count: int,
    config: Config,
) -> str | None:
    """Save episode trajectories to disk for later analysis.

    Episodes are saved as compressed numpy archives (.npz) with the following structure:
    - {eval_episode_save_dir}/step_{global_step}/episode_{i}.npz for each episode

    Args:
        episode_data: Dictionary containing episode data from evaluation
        global_step: Current training step (for naming/organization)
        save_count: Number of episodes to save (saves first N from eval_rollouts)
        config: Training configuration (for metadata and save directory)

    Returns:
        Path to the episode directory if successful, None otherwise
    """
    # Extract episode trajectories
    if "episodes" not in episode_data:
        return None  # No episode data to save

    episodes = episode_data["episodes"]
    episode_lengths = episode_data["episode_length"]
    episode_returns = episode_data["episode_return"]

    # Create output directory using configured path
    base_dir = Path(config.run.eval_episode_save_dir)
    episodes_dir = base_dir / f"step_{global_step:08d}"

    try:
        episodes_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Warning: Failed to create episode directory {episodes_dir}: {e}")
        return None

    # Save only the first save_count episodes
    num_to_save = min(save_count, len(episode_lengths))
    saved_count = 0

    for i in range(num_to_save):
        ep_len = int(episode_lengths[i])
        ep_file = episodes_dir / f"episode_{i}.npz"

        try:
            # Extract valid portion of each episode (no padding)
            np.savez_compressed(
                ep_file,
                observations=episodes["observations"][i, :ep_len],
                actions=episodes["actions"][i, :ep_len],
                rewards=episodes["rewards"][i, :ep_len],
                dones=episodes["dones"][i, :ep_len],
                # Metadata
                episode_length=ep_len,
                episode_return=float(episode_returns[i]),
                global_step=global_step,
                seed=config.run.seed,
            )
            saved_count += 1
        except (OSError, IOError) as e:
            print(f"Warning: Failed to save episode {i} to {ep_file}: {e}")
            continue

    if saved_count > 0:
        print(f"Saved {saved_count}/{num_to_save} episodes to {episodes_dir}")
        return str(episodes_dir)
    else:
        return None


def _initialize_environment_and_agent(
    config: Config | EvalConfig,
) -> tuple[Environment, Agent, Space]:
    """Initialize environment and agent (shared by train and evaluate functions).

    Args:
        config: configuration specifying environment and agent parameters.

    Returns:
        Tuple of (environment, agent, action_space)
    """
    # Create the environment
    env_kwargs = _get_factory_kwargs(config.env)
    env = make_env(config.env.name, **env_kwargs)

    # Create the agent
    agent_kwargs = _get_factory_kwargs(config.agent)
    action_space = env.get_action_space(env.config)
    agent = make_agent(config.agent.name, action_space=action_space, **agent_kwargs)

    return env, agent, action_space


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
    env, agent, action_space = _initialize_environment_and_agent(config)

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
        collection_step_fn = _make_collection_step_fn(agent, env, config.run.num_envs)
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
        sample_transition = _make_sample_transition(buffer_key, sample_obs_array, action_space)
        buffer_state = replay_buffer.init(sample_transition)
        rollout_fn = None

    # Build jitted execution primitives
    eval_rollout_fn = _make_eval_rollout_fn(agent, env, config.run.eval_rollouts, config.run.eval_max_steps)

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
        train_step_fn = _make_train_step_fn(agent, env, replay_buffer, config.run.num_envs)
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
                episode_dir = _save_episodes_to_disk(eval_results_host, global_step, save_count, config)
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


def evaluate(
    config: Config | EvalConfig,
    agent_state: AgentState | None = None,
    return_episodes: bool = False,
) -> EvaluationResults:
    """
    Evaluation-only entry point (no training).

    Useful for:
    - Non-learning controllers (random, bang-bang, PID)
    - Pre-trained models
    - Baseline comparisons
    - Benchmarking and validation

    Args:
        config: Configuration specifying environment, agent, and evaluation parameters.
            Can be either a full Config (for training) or EvalConfig (evaluation-only).
        agent_state: Optional pre-initialized agent state. If None, agent will be initialized
            with random weights using config.seed (or config.run.seed for Config).
        return_episodes: If True, return full episode trajectories in addition to metrics.
            This includes observations, actions, rewards, and dones for each step.

    Returns:
        EvaluationResults containing:
        - Summary statistics (mean_return, std_return, min, max)
        - Raw episode data (episode_returns, episode_lengths)
        - Optional trajectory data (if return_episodes=True)
        - Metadata (num_episodes, seed)
    """
    # Handle both EvalConfig and Config for wandb
    wandb_config = config.wandb if isinstance(config, EvalConfig) else config.wandb
    wandb_run = (
        maybe_init_wandb(config)
        if isinstance(config, Config)
        else (maybe_init_wandb(config) if wandb_config.enabled else None)
    )

    try:
        # Extract evaluation settings (works for both Config and EvalConfig)
        seed, eval_rollouts, eval_max_steps = _get_eval_settings(config)

        # Initialize RNG
        key = jax.random.PRNGKey(seed)
        key, env_key, agent_key = jax.random.split(key, 3)

        # Create environment and agent using shared initialization
        env, agent, _ = _initialize_environment_and_agent(config)

        # Initialize agent state if not provided
        if agent_state is None:
            # Get a sample observation to initialize the agent
            # Use original NamedTuple observation (not converted array) to allow field introspection
            obs, _ = env.reset(env_key, env.params, env.config)
            agent_state = agent.init(agent_key, obs, agent.params)

        # Create and run evaluation rollout
        eval_rollout_fn = _make_eval_rollout_fn(agent, env, eval_rollouts, eval_max_steps)
        key, eval_key = jax.random.split(key)
        eval_key, eval_results_jax = eval_rollout_fn(eval_key, agent_state, return_episodes=return_episodes)

        # Convert results from device to host
        episode_returns = jax.device_get(eval_results_jax["episode_return"])
        episode_lengths = jax.device_get(eval_results_jax["episode_length"])

        # Convert episodes if present
        episodes_data = None
        if return_episodes and "episodes" in eval_results_jax:
            episodes_data = {k: jax.device_get(v) for k, v in eval_results_jax["episodes"].items()}

        # Compute summary statistics
        mean_return = float(np.mean(episode_returns))
        std_return = float(np.std(episode_returns))
        min_return = float(np.min(episode_returns))
        max_return = float(np.max(episode_returns))

        mean_length = float(np.mean(episode_lengths))
        std_length = float(np.std(episode_lengths))
        min_length = int(np.min(episode_lengths))
        max_length = int(np.max(episode_lengths))

        # Create results object
        results = EvaluationResults(
            mean_return=mean_return,
            std_return=std_return,
            min_return=min_return,
            max_return=max_return,
            mean_length=mean_length,
            std_length=std_length,
            min_length=min_length,
            max_length=max_length,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            num_episodes=eval_rollouts,
            seed=seed,
            episodes=episodes_data,
        )

        # Log to wandb if enabled
        if wandb_run is not None:
            metrics_logger = MetricsLogger(wandb_run=wandb_run)
            # Convert to dict format expected by logger
            eval_results_dict = {
                "episode_return": episode_returns,
                "episode_length": episode_lengths,
                "dones": jax.device_get(eval_results_jax.get("dones", np.ones(eval_rollouts, dtype=bool))),
            }
            metrics_logger.log_evaluation(global_step=0, steps_per_env=0, eval_results=eval_results_dict)
            metrics_logger.log_final(0)

        return results

    finally:
        if wandb_run is not None and wandb is not None:
            wandb.finish()


def _get_factory_kwargs(config: BaseModel) -> dict:
    """Converts a dataclass config object to a dict for factory functions."""
    kwargs = config.model_dump()
    assert isinstance(kwargs, dict)
    kwargs.pop("name")  # The name is used for lookup, not as a parameter
    kwargs.pop("batch_size", None)  # batch_size is a platform parameter, not passed to factories
    return kwargs


def _make_sample_transition(key: chex.PRNGKey, sample_obs: chex.Array, action_space: Space) -> Transition:
    """Creates a sample transition PyTree for replay buffer initialization."""
    sample_action = action_space.sample(key)
    sample_reward = jnp.array(0.0, dtype=jnp.float32)
    sample_done = jnp.array(False, dtype=jnp.bool_)

    return Transition(
        sample_obs,
        sample_action,
        sample_reward,
        sample_obs,  # next_obs has the same shape as obs
        sample_done,
    )
