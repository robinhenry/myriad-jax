from functools import partial
from typing import Any, Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np

from aion.agents import make_agent
from aion.agents.agent import Agent, AgentState
from aion.configs.default import Config
from aion.core.replay_buffer import ReplayBuffer, ReplayBufferState
from aion.core.spaces import Space
from aion.core.types import BaseModel, Transition
from aion.envs import make_env
from aion.envs.environment import Environment

from .logging_utils import (
    build_train_payload,
    maybe_init_wandb,
    prepare_metrics_host,
    summarize_metric,
    wandb,
)
from .scan_utils import make_chunk_runner, mask_tree, where_mask
from .shared import TrainingEnvState


def _make_train_step_fn(
    agent: Agent,
    env: Environment,
    replay_buffer: ReplayBuffer | None,
    num_envs: int,
) -> Callable:
    """Factory to create a jitted, vmapped training step function"""

    # Vmap the environment step and reset functions so we drive all envs in lockstep
    vmapped_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    vmapped_env_reset = jax.vmap(env.reset, in_axes=(0, None, None))

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
            in_axes=(0, 0, None, None),
            out_axes=(0, None),
        )(action_keys, last_obs, agent_state, agent.params)

        # Step environments in parallel and capture the resulting transitions
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, next_env_states, rewards, dones, _ = vmapped_env_step(
            step_keys, training_env_states.env_state, actions, env.params, env.config
        )
        dones_bool = dones.astype(jnp.bool_)

        # Create transition for this step
        transitions = Transition(last_obs, actions, rewards, next_obs, dones_bool)

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

        # If done, use the new state, otherwise keep the existing one
        final_obs = where_mask(dones_bool, new_obs, next_obs)
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


def _make_rollout_collection_fn(
    agent: Agent,
    env: Environment,
    num_envs: int,
    rollout_steps: int,
) -> Callable:
    """Factory to create a jitted rollout collection function for on-policy algorithms."""

    # Vmap the environment step and reset functions
    vmapped_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    vmapped_env_reset = jax.vmap(env.reset, in_axes=(0, None, None))

    @jax.jit
    def collect_rollout(
        key: chex.PRNGKey,
        agent_state: AgentState,
        training_env_states: TrainingEnvState,
    ) -> tuple[chex.PRNGKey, AgentState, TrainingEnvState, Transition]:
        """Collect a rollout of transitions for on-policy training."""

        def step_fn(carry, _):
            """Single step of rollout collection."""
            key, agent_state, env_states = carry
            last_obs = env_states.obs

            # Select actions
            key, action_key = jax.random.split(key)
            action_keys = jax.random.split(action_key, num_envs)
            actions, agent_state = jax.vmap(
                agent.select_action,
                in_axes=(0, 0, None, None),
                out_axes=(0, None),
            )(action_keys, last_obs, agent_state, agent.params)

            # Step environments
            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, num_envs)
            next_obs, next_env_states, rewards, dones, _ = vmapped_env_step(
                step_keys, env_states.env_state, actions, env.params, env.config
            )
            dones_bool = dones.astype(jnp.bool_)

            # Handle auto-reset
            key, reset_key = jax.random.split(key)
            reset_keys = jax.random.split(reset_key, num_envs)
            new_obs, new_env_states = vmapped_env_reset(reset_keys, env.params, env.config)

            final_obs = where_mask(dones_bool, new_obs, next_obs)
            final_env_states = mask_tree(dones_bool, new_env_states, next_env_states)
            new_training_env_state = TrainingEnvState(env_state=final_env_states, obs=final_obs)

            # Store transition
            transition = Transition(last_obs, actions, rewards, next_obs, dones_bool)

            return (key, agent_state, new_training_env_state), transition

        # Collect rollout_steps transitions
        init_carry = (key, agent_state, training_env_states)
        (key, agent_state, new_env_states), transitions = jax.lax.scan(step_fn, init_carry, None, length=rollout_steps)

        # Reshape transitions: (rollout_steps, num_envs, ...) -> (rollout_steps * num_envs, ...)
        batch_transitions = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1), transitions
        )

        return key, agent_state, new_env_states, batch_transitions

    return collect_rollout


def _make_eval_rollout_fn(agent: Agent, env: Environment, config: Config) -> Callable:
    """Factory to create a jitted evaluation rollout aligned with the training loop style."""

    vmapped_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    vmapped_env_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    num_eval_envs = config.run.eval_rollouts
    max_eval_steps = config.run.eval_max_steps

    @jax.jit
    def eval_rollout(key: chex.PRNGKey, agent_state: AgentState) -> tuple[chex.PRNGKey, dict[str, chex.Array]]:
        # Reset evaluation environments
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_eval_envs)
        obs, env_states = vmapped_env_reset(reset_keys, env.params, env.config)
        eval_env_state = TrainingEnvState(env_state=env_states, obs=obs)

        # Initialize metric accumulators
        episode_returns = jnp.zeros((num_eval_envs,), dtype=jnp.float32)
        episode_lengths = jnp.zeros((num_eval_envs,), dtype=jnp.int32)
        dones = jnp.zeros((num_eval_envs,), dtype=bool)
        max_steps = jnp.asarray(max_eval_steps, dtype=jnp.int32)

        def cond_fun(
            carry: tuple[chex.PRNGKey, TrainingEnvState, chex.Array, chex.Array, chex.Array, chex.Array],
        ) -> chex.Array:
            _, _, _, _, dones, step = carry
            continue_steps = step < max_steps
            incomplete = jnp.logical_not(jnp.all(dones))
            # Exit early once every evaluation episode terminates
            return jnp.logical_and(continue_steps, incomplete)

        def body_fun(
            carry: tuple[chex.PRNGKey, TrainingEnvState, chex.Array, chex.Array, chex.Array, chex.Array],
        ) -> tuple[chex.PRNGKey, TrainingEnvState, chex.Array, chex.Array, chex.Array, chex.Array]:
            key, env_state, returns, lengths, dones, step = carry

            # Drive each evaluation environment independently but under one loop
            key, action_key, step_key = jax.random.split(key, 3)
            action_keys = jax.random.split(action_key, num_eval_envs)
            actions, _ = jax.vmap(agent.select_action, in_axes=(0, 0, None, None))(
                action_keys, env_state.obs, agent_state, agent.params
            )

            step_keys = jax.random.split(step_key, num_eval_envs)
            next_obs, next_env_states, rewards, step_dones, _ = vmapped_env_step(
                step_keys, env_state.env_state, actions, env.params, env.config
            )

            step_dones = step_dones.astype(jnp.bool_)
            active = jnp.logical_not(dones)
            active_f32 = active.astype(rewards.dtype)

            returns = returns + rewards * active_f32
            lengths = lengths + active.astype(lengths.dtype)

            env_state = TrainingEnvState(
                env_state=mask_tree(active, next_env_states, env_state.env_state),
                obs=where_mask(active, next_obs, env_state.obs),
            )

            dones = jnp.logical_or(dones, step_dones)
            step = step + jnp.array(1, dtype=step.dtype)

            return key, env_state, returns, lengths, dones, step

        # Run loop with early termination
        initial_step = jnp.array(0, dtype=jnp.int32)
        initial_carry = (key, eval_env_state, episode_returns, episode_lengths, dones, initial_step)
        final_carry = jax.lax.while_loop(cond_fun, body_fun, initial_carry)
        key, _, final_returns, final_lengths, final_dones, _ = final_carry

        # Package metrics for the caller
        metrics = {
            "episode_return": final_returns,
            "episode_length": final_lengths,
            "dones": final_dones,
        }

        return key, metrics

    return eval_rollout


def _run_training_loop(config: Config, wandb_run: Any) -> None:
    """Executes the training loop and emits host-side and W&B logs."""

    # Initialize everything
    key = jax.random.PRNGKey(config.run.seed)
    key, env_key, agent_key, buffer_key = jax.random.split(key, 4)

    # Create the environment
    env_kwargs = _get_factory_kwargs(config.env)
    env = make_env(config.env.name, **env_kwargs)

    # Create the agent
    agent_kwargs = _get_factory_kwargs(config.agent)
    action_space = env.get_action_space(env.config)
    agent = make_agent(config.agent.name, action_space=action_space, **agent_kwargs)

    # Initialize parallel environments
    env_keys = jax.random.split(env_key, config.run.num_envs)
    obs, env_states = jax.vmap(env.reset, in_axes=(0, None, None))(env_keys, env.params, env.config)
    training_env_states = TrainingEnvState(env_state=env_states, obs=obs)

    # Initialize agent using the initial observation from one environment
    sample_obs = obs[0]  # type: ignore
    agent_state = agent.init(agent_key, sample_obs, agent.params)

    # Determine training mode and initialize accordingly
    use_rollout_training = config.run.rollout_steps is not None

    if use_rollout_training:
        # On-policy training (e.g., PQN): no replay buffer needed
        replay_buffer = None
        buffer_state = None
        assert config.run.rollout_steps is not None  # should always be true if use_rollout_training is true
        rollout_fn = _make_rollout_collection_fn(agent, env, config.run.num_envs, config.run.rollout_steps)
    else:
        # Off-policy training (e.g., DQN): use replay buffer
        if config.run.buffer_size is None:
            raise ValueError("buffer_size must be set in config for off-policy training (when rollout_steps is None)")
        replay_buffer = ReplayBuffer(buffer_size=config.run.buffer_size)
        sample_transition = _make_sample_transition(buffer_key, sample_obs, action_space)
        buffer_state = replay_buffer.init(sample_transition)
        rollout_fn = None

    # Build jitted execution primitives
    train_step_fn = _make_train_step_fn(agent, env, replay_buffer, config.run.num_envs)
    eval_rollout_fn = _make_eval_rollout_fn(agent, env, config)

    # Chunking configuration:
    # - scan_chunk_size controls how many training steps are batched into a single jax.lax.scan
    # - Larger chunks reduce Python overhead but increase XLA compile time
    # - chunk_size is ensured to be at least 1 to prevent errors
    chunk_size = max(1, config.run.scan_chunk_size)
    # For off-policy agents, batch_size is required; for on-policy, it's ignored
    batch_size = config.agent.batch_size if config.agent.batch_size is not None else 1
    run_chunk_fn = make_chunk_runner(train_step_fn, batch_size)

    total_steps = config.run.total_timesteps // config.run.num_envs
    log_frequency = config.run.log_frequency
    eval_frequency = config.run.eval_frequency

    steps_completed = 0
    while steps_completed < total_steps:
        remaining_steps = total_steps - steps_completed

        if use_rollout_training:
            # Should always be true if use_rollout_training is true
            assert config.run.rollout_steps is not None
            assert rollout_fn is not None

            # On-policy training: collect rollout, then update agent
            key, agent_state, training_env_states, rollout_batch = rollout_fn(key, agent_state, training_env_states)

            # Update agent with collected rollout
            key, update_key = jax.random.split(key)
            agent_state, metrics = agent.update(update_key, agent_state, rollout_batch, agent.params)

            # Convert single metrics dict to history format for logging
            metrics_history = jax.tree_util.tree_map(lambda x: jnp.array([x]), metrics)
            steps_this_chunk = config.run.rollout_steps * config.run.num_envs
        else:
            # Off-policy training: use chunked step-by-step updates

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

            # Boundary alignment:
            # Determine how many steps to run before the next logging or eval boundary.
            # This ensures we can log/eval at exact frequencies without partial metrics.
            # steps_this_chunk is limited by:
            # 1. chunk_size: The configured maximum chunk size
            # 2. remaining_steps: Don't overshoot total_timesteps
            # 3. steps_to_log: Align with logging frequency
            # 4. steps_to_eval: Align with evaluation frequency
            steps_to_log = _steps_until_boundary(steps_completed, log_frequency)
            steps_to_eval = _steps_until_boundary(steps_completed, eval_frequency)
            steps_this_chunk = min(chunk_size, remaining_steps, steps_to_log, steps_to_eval)

            # Create a boolean mask for the scan:
            # - active_mask always has length chunk_size (for consistent JIT compilation)
            # - Only the first steps_this_chunk elements are True
            # - Inactive iterations (False elements) execute but don't update state
            active_mask = (jnp.arange(chunk_size) < steps_this_chunk).astype(jnp.bool_)
            (key, agent_state, training_env_states, buffer_state), metrics_history = run_chunk_fn(
                (key, agent_state, training_env_states, buffer_state),
                active_mask,
            )

        steps_completed += steps_this_chunk
        global_step = steps_completed * config.run.num_envs

        # Host-side logging pulls the latest metrics emitted during the training chunk
        metrics_host = prepare_metrics_host(metrics_history, steps_this_chunk)

        should_log = steps_completed % log_frequency == 0 and metrics_host
        if should_log and wandb_run is not None and wandb is not None:
            train_payload = build_train_payload(metrics_host)
            if train_payload:
                train_payload["train/global_env_steps"] = float(global_step)
                train_payload["train/steps_per_env"] = float(steps_completed)
                wandb.log(train_payload, step=global_step)

        # Periodically run evaluation rollouts without touching the training buffer
        should_eval = eval_frequency > 0 and steps_completed > 0 and steps_completed % eval_frequency == 0
        if should_eval:
            key, eval_key = jax.random.split(key)
            eval_key, eval_metrics = eval_rollout_fn(eval_key, agent_state)
            key = eval_key
            eval_metrics_host = {name: jax.device_get(value) for name, value in eval_metrics.items()}
            eval_returns = eval_metrics_host.get("episode_return")
            eval_lengths = eval_metrics_host.get("episode_length")
            eval_dones = eval_metrics_host.get("dones")

            if wandb_run is not None and wandb is not None:
                eval_payload: dict[str, float] = {}
                if eval_returns is not None:
                    eval_payload.update(summarize_metric("eval/", "episode_return", eval_returns))
                if eval_lengths is not None:
                    eval_payload.update(summarize_metric("eval/", "episode_length", eval_lengths))
                if eval_dones is not None:
                    termination_rate = float(np.asarray(eval_dones, dtype=np.float32).mean())
                    eval_payload["eval/termination_rate"] = termination_rate
                    eval_payload["eval/non_termination_rate"] = float(1.0 - termination_rate)

                if eval_payload:
                    eval_payload["eval/global_env_steps"] = float(global_step)
                    wandb.log(eval_payload, step=global_step)

    total_env_steps = steps_completed * config.run.num_envs
    if wandb_run is not None and wandb is not None:
        wandb.log({"train/final_env_steps": float(total_env_steps)}, step=total_env_steps)


def train_and_evaluate(config: Config):
    """
    Main entry point for a training run.
    Initializes everything and runs the outer training loop.
    """
    wandb_run = maybe_init_wandb(config)

    try:
        _run_training_loop(config, wandb_run)
    finally:
        if wandb_run is not None and wandb is not None:
            wandb.finish()


def _get_factory_kwargs(config: BaseModel) -> dict:
    """Converts a dataclass config object to a dict for factory functions."""
    kwargs = config.model_dump()
    assert isinstance(kwargs, dict)
    kwargs.pop("name")  # The name is used for lookup, not as a parameter
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
