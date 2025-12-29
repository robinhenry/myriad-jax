"""JAX step function factories for training, collection, and evaluation.

This module contains pure JAX transformation utilities that define how individual
steps are executed in training and evaluation loops. These are low-level primitives
consumed by higher-level orchestration code.

All functions here are pure and designed to be jitted for maximum performance.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import chex
import jax
import jax.numpy as jnp

from myriad.agents.agent import Agent, AgentState
from myriad.core.replay_buffer import ReplayBuffer, ReplayBufferState
from myriad.core.spaces import Space
from myriad.core.types import Transition
from myriad.envs.environment import Environment
from myriad.utils import to_array

from .scan_utils import mask_tree, where_mask
from .shared import TrainingEnvState


def make_train_step_fn(
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


def make_collection_step_fn(
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


def make_eval_rollout_fn(agent: Agent, env: Environment, eval_rollouts: int, eval_max_steps: int) -> Callable:
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
            episode_obs = episode_actions = episode_rewards = episode_dones = None  # type: ignore[assignment]

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
            ) = final_carry
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


def make_sample_transition(key: chex.PRNGKey, sample_obs: chex.Array, action_space: Space) -> Transition:
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
