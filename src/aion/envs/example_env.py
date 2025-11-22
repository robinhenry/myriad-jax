"""A toy control environment implemented as pure JAX functions."""

import logging
from typing import Any, Dict, NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from aion.core.spaces import Box

from .environment import Environment

logger = logging.getLogger(__name__)


@struct.dataclass
class EnvConfig:
    """Static configuration for the environment."""

    min_x: float = 0.0
    max_x: float = 10.0
    max_u: float = 3.0
    target_tol: float = 0.1
    max_steps: int = 100


@struct.dataclass
class EnvParams:
    """Dynamic parameters for the environment."""

    a: float
    x_target: float


class EnvState(NamedTuple):
    """Environment state containing position and time step."""

    x: chex.Array  # JAX array for position
    t: chex.Array  # JAX array for time step


def create_env_params(a: float = 1.0, x_target: float = 5.0) -> EnvParams:
    """
    Factory function to create and validate EnvParams and EnvConfig.
    """
    return EnvParams(a=a, x_target=x_target)


def _step(
    key: chex.PRNGKey, state: EnvState, action: chex.Array, params: EnvParams, config: EnvConfig
) -> tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
    """Step the environment forward one time step."""
    x, t = state.x, state.t

    # Clip action to valid range
    u = jnp.clip(action, -config.max_u, config.max_u)

    # Update state
    x_next = x + params.a * u
    x_next = jnp.clip(x_next, config.min_x, config.max_x)

    # Calculate reward as the negative absolute difference from the target
    reward = -jnp.abs(x_next - params.x_target)

    # Update time step
    t_next = t + 1

    # Check if the episode is done
    max_steps_reached = t_next >= config.max_steps
    target_reached = jnp.abs(x_next - params.x_target) < config.target_tol
    done = jnp.where(jnp.logical_or(max_steps_reached, target_reached), 1.0, 0.0)

    # Create next state and observation
    next_state = EnvState(x_next, t_next)
    obs_next = get_obs(next_state, params, config)

    return obs_next, next_state, reward, done, {}


def _reset(key: chex.PRNGKey, params: EnvParams, config: EnvConfig) -> tuple[chex.Array, EnvState]:
    """Reset the environment to initial state."""

    # Reset state to a random value
    x = jax.random.uniform(key, shape=(), minval=config.min_x, maxval=config.max_x)
    t = jnp.array(0)
    state = EnvState(x, t)
    obs = get_obs(state, params, config)
    return obs, state


step = jax.jit(_step, static_argnames=["config"])
reset = jax.jit(_reset, static_argnames=["config"])


def get_obs(state: EnvState, params: EnvParams, config: EnvConfig) -> chex.Array:
    return jnp.array([state.x, params.x_target])


def get_obs_shape(_config: EnvConfig) -> tuple:
    """Get the size of the observation space."""
    return (2,)


def get_action_space(config: EnvConfig) -> Box:
    """Get the continuous action space for the environment."""
    return Box(low=-config.max_u, high=config.max_u, shape=())


def make_env(
    config: EnvConfig | None = None,
    params: EnvParams | None = None,
    **kwargs,
) -> Environment[EnvState, EnvParams, EnvConfig]:
    """
    Creates an instance of the the environment with optional custom settings.

    This function allows for flexible creation:
    1. Pass a complete `params` object directly.
    2. Pass keyword arguments (e.g., `a=0.9`, `x_target=...`) which will be
       used to create a new `params` object.

    Args:
        config: A custom EnvConfig object. If None, a default one is created.
        params: A pre-constructed EnvParams object. If provided, kwargs are ignored.
        **kwargs: Keyword arguments (a, b, x_target) for creating EnvParams if
                  `params` is not provided.

    Returns:
        An Environment object.
    """
    # If no config is provided, create a default one
    if config is None:
        config = EnvConfig()

    # If no params object is provided, create one from kwargs
    if params is None:
        params = create_env_params(**kwargs)
    elif kwargs:
        logger.warning("`params` object was provided, so keyword arguments %s will be ignored.", list(kwargs.keys()))

    return Environment(
        step=step,
        reset=reset,
        get_action_space=get_action_space,
        get_obs_shape=get_obs_shape,
        params=params,
        config=config,
    )
