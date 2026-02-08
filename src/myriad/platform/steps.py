"""JAX step functions and primitives for training, collection, and evaluation.

This module contains:
1. Masking primitives (tree_select, mask_tree, where_mask) for conditional state updates
2. Step function factories for training, collection, and evaluation loops

All functions here are pure and designed to be jitted for maximum performance.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from myriad.agents.agent import Agent, AgentState
from myriad.core.replay_buffer import ReplayBuffer, ReplayBufferState
from myriad.core.spaces import Space
from myriad.core.types import PRNGKey, Transition
from myriad.envs.environment import Environment
from myriad.utils import to_array

from .types import TrainingEnvState

# =============================================================================
# Masking Primitives
# =============================================================================


def tree_select(mask: Array, new_tree: Any, old_tree: Any) -> Any:
    """Select between pytrees using a scalar boolean. For batched selection, use mask_tree."""
    return jax.tree_util.tree_map(lambda new, old: jax.lax.select(mask, new, old), new_tree, old_tree)


def _expand_mask(mask: Array, target_ndim: int) -> Array:
    """Reshapes a mask so it can broadcast to a target rank."""
    expand_dims = target_ndim - mask.ndim
    if expand_dims <= 0:
        return mask
    return mask.reshape(mask.shape + (1,) * expand_dims)


def where_mask(mask: Array, new_value: Array, old_value: Array) -> Array:
    """Selects array values using a boolean mask, supporting broadcasting."""
    mask_bool = mask.astype(jnp.bool_)
    return jnp.where(_expand_mask(mask_bool, new_value.ndim), new_value, old_value)


def mask_tree(mask: Array, new_tree: Any, old_tree: Any) -> Any:
    """Select between pytrees using a vector boolean mask (per-element batched selection)."""
    return jax.tree_util.tree_map(lambda new, old: where_mask(mask, new, old), new_tree, old_tree)


# =============================================================================
# Step Function Factories
# =============================================================================


class _EnvStepResult(NamedTuple):
    """Result of a single environment step with auto-reset handling."""

    key: PRNGKey
    agent_state: AgentState
    new_training_env_state: TrainingEnvState
    transition: Transition


def _make_env_stepper(
    agent: Agent,
    env: Environment,
    num_envs: int,
) -> Callable[[PRNGKey, AgentState, TrainingEnvState], _EnvStepResult]:
    """Creates reusable vmapped env stepping logic with auto-reset.

    Shared by make_train_step_fn and make_collection_step_fn. Handles:
    - Vmapped env.step/reset
    - Action selection
    - Observation → array conversion (for platform utilities)
    - Auto-reset via masking
    """
    vmapped_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    vmapped_env_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    to_array_batch = jax.vmap(to_array)

    def env_step(
        key: PRNGKey,
        agent_state: AgentState,
        training_env_states: TrainingEnvState,
    ) -> _EnvStepResult:
        """Perform one environment step with auto-reset handling."""
        last_obs = training_env_states.obs

        # Split keys for action selection, env step, and reset in one operation
        key, action_key, step_key, reset_key = jax.random.split(key, 4)
        action_keys = jax.random.split(action_key, num_envs)
        step_keys = jax.random.split(step_key, num_envs)
        reset_keys = jax.random.split(reset_key, num_envs)

        # Select actions using per-env keys while sharing agent state and params
        actions, agent_state = jax.vmap(
            agent.select_action,
            in_axes=(0, 0, None, None, None),
            out_axes=(0, None),
        )(action_keys, last_obs, agent_state, agent.params, False)

        # Step environments in parallel
        next_obs, next_env_states, rewards, dones, _ = vmapped_env_step(
            step_keys, training_env_states.env_state, actions, env.params, env.config
        )
        next_obs_array = to_array_batch(next_obs)
        dones_bool = dones.astype(jnp.bool_)

        # Create transition (uses pre-reset next_obs for correct TD targets)
        transition = Transition(last_obs, actions, rewards, next_obs_array, dones_bool)

        # Handle auto-reset for completed episodes
        new_obs, new_env_states = vmapped_env_reset(reset_keys, env.params, env.config)
        new_obs_array = to_array_batch(new_obs)

        # If done, use the reset state, otherwise keep the stepped state
        final_obs = where_mask(dones_bool, new_obs_array, next_obs_array)
        final_env_states = mask_tree(dones_bool, new_env_states, next_env_states)
        new_training_env_state = TrainingEnvState(env_state=final_env_states, obs=final_obs)

        return _EnvStepResult(key, agent_state, new_training_env_state, transition)

    return env_step


def make_train_step_fn(
    agent: Agent,
    env: Environment,
    replay_buffer: ReplayBuffer | None,
    num_envs: int,
) -> Callable:
    """Factory to create a jitted, vmapped training step function.

    This function wraps the common env stepping logic with replay buffer handling
    and agent updates for off-policy training.
    """
    env_step = _make_env_stepper(agent, env, num_envs)

    @partial(jax.jit, static_argnames=["batch_size"])
    def train_step(
        key: PRNGKey,
        agent_state: AgentState,
        training_env_states: TrainingEnvState,
        buffer_state: ReplayBufferState | None,
        batch_size: int,
    ) -> tuple[PRNGKey, AgentState, TrainingEnvState, ReplayBufferState | None, dict]:
        """Executes one step of training across all parallel environments."""
        # Step environments and collect transition
        result = env_step(key, agent_state, training_env_states)

        # Handle replay buffer if present (off-policy algorithms like DQN)
        if replay_buffer is not None and buffer_state is not None:
            key, buffer_key = jax.random.split(result.key)
            buffer_state, batch = replay_buffer.add_and_sample(buffer_state, result.transition, batch_size, buffer_key)
        else:
            key = result.key
            batch = result.transition

        # Update the agent with the sampled batch
        key, update_key = jax.random.split(key)
        agent_state, metrics = agent.update(update_key, result.agent_state, batch, agent.params)

        return (key, agent_state, result.new_training_env_state, buffer_state, metrics)

    return train_step


def make_collection_step_fn(
    agent: Agent,
    env: Environment,
    num_envs: int,
) -> Callable:
    """Factory to create a single-step collection function for on-policy algorithms.

    This creates a step function that collects one transition without performing agent updates.
    It's designed to be used with make_chunked_collector for efficient rollout collection.
    """
    env_step = _make_env_stepper(agent, env, num_envs)

    def collection_step(
        key: PRNGKey,
        agent_state: AgentState,
        training_env_states: TrainingEnvState,
    ) -> tuple[tuple[PRNGKey, AgentState, TrainingEnvState], Transition]:
        """Execute one step of rollout collection: select action, step env, collect transition."""
        result = env_step(key, agent_state, training_env_states)
        return (result.key, result.agent_state, result.new_training_env_state), result.transition

    return collection_step


def make_eval_rollout_fn(agent: Agent, env: Environment, eval_rollouts: int, eval_max_steps: int) -> Callable:
    """Factory to create a jitted evaluation rollout function.

    Uses while_loop for early termination when all episodes complete. This differs from
    training's fixed-size scans because eval runs infrequently and benefits from early exit.

    Args:
        agent: The agent to evaluate
        env: The environment to evaluate in
        eval_rollouts: Number of parallel evaluation episodes
        eval_max_steps: Maximum steps per episode

    Returns:
        Function (key, agent_state, return_episodes=False) -> (key, metrics_dict)
    """
    vmapped_env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None, None))
    vmapped_env_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    # Vectorized observation conversion ensures platform always operates on arrays
    to_array_batch = jax.vmap(to_array)
    num_eval_envs = eval_rollouts
    max_eval_steps = eval_max_steps

    @partial(jax.jit, static_argnames=["return_episodes"])
    def eval_rollout(
        key: PRNGKey, agent_state: AgentState, return_episodes: bool = False
    ) -> tuple[PRNGKey, dict[str, Array]]:
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
            episode_obs = episode_actions = episode_rewards = episode_dones = None  # type: ignore[assignment]

        def cond_fun(carry: tuple) -> Array:
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
            initial_carry = (key, eval_env_state, episode_returns, episode_lengths, dones, initial_step)  # type: ignore[assignment]

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
            ) = final_carry  # type: ignore[misc]
        else:
            key, _, final_returns, final_lengths, final_dones, _ = final_carry  # type: ignore[misc]

        # Package metrics for the caller
        metrics = {
            "episode_return": final_returns,
            "episode_length": final_lengths,
            "dones": final_dones,
        }

        # Add episode data if requested
        if return_episodes:
            metrics["episodes"] = {  # type: ignore[assignment]
                "observations": final_obs,
                "actions": final_actions,
                "rewards": final_rewards,
                "dones": final_dones_ep,
            }

        return key, metrics

    return eval_rollout


def make_sample_transition(key: PRNGKey, sample_obs: Array, action_space: Space) -> Transition:
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
