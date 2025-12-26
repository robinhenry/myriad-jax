"""Environment wrappers for compatibility with different frameworks.

This module provides wrappers to adapt Myriad environments to different interfaces,
particularly for compatibility with standard RL frameworks that expect array observations.
"""

from typing import Callable, TypeVar

import chex
import jax.tree_util as jtu

from myriad.envs.environment import Environment, EnvironmentConfig, EnvironmentParams, EnvironmentState

S = TypeVar("S", bound=EnvironmentState)
P = TypeVar("P", bound=EnvironmentParams)
C = TypeVar("C", bound=EnvironmentConfig)


def make_array_obs_env(
    env: Environment[S, P, C],
    obs_to_array: Callable | None = None,
) -> Environment[S, P, C]:
    """Wrap an environment to convert NamedTuple observations to arrays.

    This wrapper is useful for compatibility with standard RL frameworks (Gym, Gymnasium)
    and neural network agents that expect flat array observations.

    Args:
        env: Environment with potentially structured observations (NamedTuple, etc.)
        obs_to_array: Optional conversion function. If None, assumes observations have
            a `.to_array()` method (the standard pattern for Myriad observations).

    Returns:
        Environment with array observations

    Example:
        >>> from myriad.envs.cartpole.tasks.control import make_env
        >>> env = make_env()  # Returns CartPoleObs observations
        >>> array_env = make_array_obs_env(env)  # Returns array observations
        >>>
        >>> key = jax.random.PRNGKey(0)
        >>> obs, state = array_env.reset(key, array_env.params, array_env.config)
        >>> print(obs.shape)  # (4,)
        >>>
        >>> # If you need a custom conversion:
        >>> def custom_converter(obs):
        ...     return jnp.array([obs.x, obs.theta])  # Only position and angle
        >>> partial_env = make_array_obs_env(env, obs_to_array=custom_converter)
    """

    # Default: use the `.to_array()` method
    if obs_to_array is None:

        def obs_to_array(obs):
            return obs.to_array()

    def wrapped_step(
        key: chex.PRNGKey,
        state: S,
        action: chex.Array,
        params: P,
        config: C,
    ):
        obs, next_state, reward, done, info = env.step(key, state, action, params, config)
        return obs_to_array(obs), next_state, reward, done, info

    def wrapped_reset(
        key: chex.PRNGKey,
        params: P,
        config: C,
    ):
        obs, state = env.reset(key, params, config)
        return obs_to_array(obs), state

    # Return a new Environment with wrapped functions
    return env._replace(
        step=wrapped_step,
        reset=wrapped_reset,
    )


def vectorize_obs_conversion(
    obs_to_array: Callable,
) -> Callable:
    """Vectorize an observation conversion function for batched environments.

    This is useful when you have a batch of NamedTuple observations and want to
    convert them all to arrays efficiently using jax.tree_map.

    Args:
        obs_to_array: Function that converts a single observation to an array

    Returns:
        Vectorized conversion function

    Example:
        >>> # For a batch of CartPoleObs
        >>> obs_batch = jax.vmap(env.reset)(keys, params_batch, config_batch)[0]
        >>> vectorized_converter = vectorize_obs_conversion(lambda obs: obs.to_array())
        >>> array_batch = jtu.tree_map(vectorized_converter, obs_batch)
    """

    def vectorized_converter(obs_batch):
        """Convert a batch of observations to arrays."""
        return jtu.tree_map(obs_to_array, obs_batch)

    return vectorized_converter
