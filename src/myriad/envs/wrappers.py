"""Environment wrappers for compatibility with different frameworks.

This module provides wrappers to adapt Myriad environments to different interfaces.
"""

from typing import Any, Callable, TypeVar

from jax import Array

from myriad.envs.environment import Environment, EnvironmentConfig, EnvironmentParams, EnvironmentState

S = TypeVar("S", bound=EnvironmentState)
C = TypeVar("C", bound=EnvironmentConfig)
P = TypeVar("P", bound=EnvironmentParams)


def make_array_obs_env(
    env: Environment[S, C, P, Any],
    obs_to_array: Callable[[Any], Array] | None = None,
) -> Environment[S, C, P, Array]:  # type: ignore[type-var]
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
    converter = obs_to_array if obs_to_array is not None else lambda obs: obs.to_array()

    def wrapped_step(
        key: Array,
        state: S,
        action: Array,
        params: P,
        config: C,
    ):
        obs, next_state, reward, done, info = env.step(key, state, action, params, config)
        return converter(obs), next_state, reward, done, info

    def wrapped_reset(
        key: Array,
        params: P,
        config: C,
    ):
        obs, state = env.reset(key, params, config)
        return converter(obs), state

    # Return a new Environment with wrapped functions
    return env._replace(
        step=wrapped_step,
        reset=wrapped_reset,
    )
