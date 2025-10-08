"""Functions for interacting with the environment - optimized for GPU scaling."""

from functools import partial
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from aion.envs.toy_env_v1 import EnvParams, reset as env_reset, step as env_step


def run_episode(
    env_reset_fn: Callable,
    env_step_fn: Callable,
    agent_select_action: Callable,
    key: chex.PRNGKey,
    env_params: EnvParams,
    train_state: TrainState,
    max_steps: int,
) -> Tuple[chex.Array, chex.Array]:
    """
    Runs a single episode of the environment with the given agent.
    Optimized for JIT compilation.

    Args:
        env_reset_fn: Environment reset function
        env_step_fn: Environment step function
        agent_select_action: Agent action selection function
        key: JAX random key
        env_params: Environment parameters
        train_state: Agent training state
        max_steps: Maximum number of steps

    Returns:
        A tuple of (observations, rewards)
    """
    key, reset_key = jax.random.split(key)
    obs, state = env_reset_fn(reset_key, env_params)

    def scan_step(carry, _):
        key, obs, state, step = carry
        key, action_key, step_key = jax.random.split(key, 3)

        action, _ = agent_select_action(action_key, obs, train_state, step)
        next_obs, next_state, reward, done, _ = env_step_fn(step_key, state, action, env_params)

        # Continue with next observation or stop if done
        obs = jnp.where(done, obs, next_obs)
        state = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), state, next_state)

        return (key, obs, state, step + 1), (obs, reward)

    initial_carry = (key, obs, state, 0)
    _, (observations, rewards) = jax.lax.scan(scan_step, initial_carry, None, length=max_steps)

    return observations, rewards


@partial(jax.jit, static_argnums=(1, 4, 5))
def run_episodes_parallel(
    key: chex.PRNGKey,
    agent_select_action: Callable,
    train_state: TrainState,
    env_params: EnvParams,
    num_envs: int,
    max_steps_in_episode: int,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Runs multiple episodes in parallel using jax.vmap for optimal GPU utilization.

    This function is JIT-compiled and optimized for large-scale parallel execution
    on GPUs with thousands of environments.

    Args:
        key: JAX random key
        agent_select_action: The agent's action selection method
        train_state: The training state of the agent
        env_params: The environment parameters
        num_envs: The number of parallel environments
        max_steps_in_episode: The maximum number of steps per episode

    Returns:
        A tuple of (obs, actions, rewards, next_obs, dones) trajectories
        Each with shape (max_steps_in_episode, num_envs, ...)
    """

    def scan_step(carry, _):
        """Single step for all environments in parallel."""
        key, obs, env_state, step = carry
        key, action_key, step_key = jax.random.split(key, 3)

        # Parallel action selection across environments
        actions, _ = jax.vmap(agent_select_action, in_axes=(0, 0, None, None))(
            jax.random.split(action_key, num_envs), obs, train_state, step
        )

        # Parallel environment stepping
        next_obs, next_env_state, rewards, dones, _ = jax.vmap(env_step, in_axes=(0, 0, 0, None))(
            jax.random.split(step_key, num_envs), env_state, actions, env_params
        )

        transition = (obs, actions, rewards, next_obs, dones)
        return (key, next_obs, next_env_state, step + 1), transition

    # Initialize all environments in parallel
    key, reset_key = jax.random.split(key)
    initial_obs, initial_states = jax.vmap(env_reset, in_axes=(0, None))(
        jax.random.split(reset_key, num_envs), env_params
    )

    # Run the parallel rollout
    initial_carry = (key, initial_obs, initial_states, 0)
    _, trajectories = jax.lax.scan(scan_step, initial_carry, None, length=max_steps_in_episode)

    return trajectories
