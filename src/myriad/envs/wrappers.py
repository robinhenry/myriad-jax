"""Environment wrappers for compatibility with different frameworks.

This module provides wrappers to adapt Myriad environments to different interfaces.
"""

import math
import warnings
from typing import Any, Callable, NamedTuple, TypeVar

import jax.numpy as jnp
from jax import Array

from myriad.envs.environment import Environment, EnvironmentConfig, EnvironmentParams, EnvironmentState
from myriad.utils.observations import to_array

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


# ---------------------------------------------------------------------------
# Frame stacking
# ---------------------------------------------------------------------------


class FrameStackState(NamedTuple):
    """State for a frame-stacking wrapper.

    Attributes:
        env_state: The wrapped environment's state (any pytree-compatible type).
        obs_buffer: Ring buffer of the last ``n_frames`` observations.
            Shape: ``(n_frames, obs_dim)``, newest frame in the **last** slot.
    """

    env_state: Any
    obs_buffer: Array


def make_frame_stack_env(env: Environment, n_frames: int) -> Environment:
    """Wrap an environment to stack the last ``n_frames`` observations.

    Returns flat ``Array`` observations of shape ``(n_frames * obs_dim,)``.
    The wrapped env's state becomes :class:`FrameStackState`, which bundles the
    inner env state with a ring buffer of recent observations.

    On :func:`reset`, the buffer is zero-filled with the initial observation
    placed in the last slot (newest position). On :func:`step`, the buffer is
    rolled by one and the new observation is inserted at the end.

    Both functions are pure JAX — compatible with :func:`jax.jit`,
    :func:`jax.vmap`, and :func:`jax.lax.scan`.

    Args:
        env: Environment to wrap. Observations must be flat :class:`~jax.Array`
            or have a ``.to_array()`` method (the standard Myriad pattern).
        n_frames: Number of consecutive observations to stack.

    Returns:
        A new :class:`~myriad.envs.environment.Environment` whose ``reset``
        and ``step`` return stacked observations and :class:`FrameStackState`.

    Example:
        >>> import jax
        >>> from myriad.envs import make_env
        >>> from myriad.envs.wrappers import make_frame_stack_env
        >>> inner = make_env("cartpole-control")
        >>> env = make_frame_stack_env(inner, n_frames=4)
        >>> obs, state = env.reset(jax.random.PRNGKey(0), env.params, env.config)
        >>> obs.shape  # (4 * 4,)
        (16,)
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1 (got {n_frames}).")
    if n_frames == 1:
        warnings.warn(
            "make_frame_stack_env called with n_frames=1. This adds wrapper overhead with no benefit "
            "— the stacked obs is identical to the raw obs. Use the env directly instead.",
            UserWarning,
            stacklevel=2,
        )

    inner_obs_shape = env.get_obs_shape(env.config)
    obs_dim = math.prod(inner_obs_shape)

    def _get_obs_shape(config) -> tuple[int, ...]:
        return (n_frames * math.prod(env.get_obs_shape(config)),)

    def _reset(key, params, config):
        obs, inner_state = env.reset(key, params, config)
        obs_flat = to_array(obs).reshape(obs_dim)
        buffer = jnp.concatenate([jnp.zeros((n_frames - 1, obs_dim), dtype=obs_flat.dtype), obs_flat[None]], axis=0)
        return buffer.reshape(-1), FrameStackState(env_state=inner_state, obs_buffer=buffer)

    def _step(key, state, action, params, config):
        obs, inner_state, reward, done, info = env.step(key, state.env_state, action, params, config)
        obs_flat = to_array(obs).reshape(obs_dim)
        new_buffer = jnp.concatenate([state.obs_buffer[1:], obs_flat[None]], axis=0)
        return new_buffer.reshape(-1), FrameStackState(env_state=inner_state, obs_buffer=new_buffer), reward, done, info

    return env._replace(
        reset=_reset,
        step=_step,
        get_obs_shape=_get_obs_shape,
    )
