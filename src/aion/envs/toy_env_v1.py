"""A toy control environment implemented as pure JAX functions."""

from functools import partial
from typing import Any, Dict, NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from .base import Environment


@dataclass
class EnvConfig:
    """Static configuration for the environment."""

    min_x: float = 0.0
    max_x: float = 20.0
    max_steps: int = 100
    prediction_horizon: int = 10


@dataclass
class EnvParams:
    """Dynamic parameters for the environment."""

    a: float
    b: float
    x_target: chex.Array


def create_env_params(
    config: EnvConfig,
    a: float = 1.0,
    b: float = 1.0,
    x_target: chex.Array | float = 5.0,
) -> EnvParams:
    """
    Factory function to create and validate EnvParams and EnvConfig.
    """
    processed_x_target = jnp.asarray(x_target)
    if processed_x_target.ndim == 0:
        processed_x_target = jnp.full((config.max_steps,), processed_x_target)
    elif len(processed_x_target) != config.max_steps:
        raise ValueError(f"Length of x_target ({len(processed_x_target)}) must match max_steps ({config.max_steps}).")

    params = EnvParams(a=a, b=b, x_target=processed_x_target)
    return params


class EnvState(NamedTuple):
    """Environment state containing position and time step."""

    x: chex.Array  # JAX array for position
    t: chex.Array  # JAX array for time step


def create_constant_target(value: float, length: int) -> chex.Array:
    """Create a constant target trajectory."""
    return jnp.full((length,), value)


def create_sine_target(amplitude: float, frequency: float, offset: float, length: int) -> chex.Array:
    """Create a sinusoidal target trajectory."""
    t = jnp.linspace(0, 2 * jnp.pi * frequency, length)
    return amplitude * jnp.sin(t) + offset


def create_linear_target(start: float, end: float, length: int) -> chex.Array:
    """Create a linear target trajectory."""
    return jnp.linspace(start, end, length)


def create_step_target(values: list, steps_per_value: int) -> chex.Array:
    """Create a step-wise target trajectory."""
    return jnp.repeat(jnp.array(values), steps_per_value)


def step(
    key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams, config: EnvConfig
) -> tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
    """Step the environment forward one time step."""
    x, t = state.x, state.t

    # Map action from {0, 1} to {-1, 1}
    u = action * 2 - 1

    # Update state
    x_next = params.a * u + params.b * x
    x_next = jnp.clip(x_next, config.min_x, config.max_x)

    # Get current target value based on time step
    max_idx = jnp.size(params.x_target) - 1
    time_idx = jnp.clip(state.t, 0, max_idx)
    current_target = params.x_target[time_idx]  # type: ignore

    # Calculate reward as the negative absolute difference from the target
    reward = -jnp.abs(x_next - current_target)

    # Update time step
    t_next = t + 1

    # Check if the episode is done
    done = jnp.where(t_next >= config.max_steps, 1.0, 0.0)

    # Create next state and observation
    next_state = EnvState(x_next, t_next)
    obs_next = get_obs(next_state, params, config)

    return obs_next, next_state, reward, done, {}


def reset(key: chex.PRNGKey, params: EnvParams, config: EnvConfig) -> tuple[chex.Array, EnvState]:
    """Reset the environment to initial state."""
    # Reset state to a random value
    x = jax.random.uniform(key, shape=(), minval=config.min_x, maxval=config.max_x)
    t = jnp.array(0)
    state = EnvState(x, t)
    obs = get_obs(state, params, config)
    return obs, state


@partial(jax.jit, static_argnames=["config"])
def jit_step(
    key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams, config: EnvConfig
) -> tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
    return step(key, state, action, params, config)


@partial(jax.jit, static_argnames=["config"])
def jit_reset(key: chex.PRNGKey, params: EnvParams, config: EnvConfig) -> tuple[chex.Array, EnvState]:
    return reset(key, params, config)


def get_obs(state: EnvState, params: EnvParams, config: EnvConfig) -> chex.Array:
    """Get observation from state."""
    # Get current target from trajectory
    max_idx = jnp.size(params.x_target) - 1

    # Create indices for the lookahead window
    indices = state.t + jnp.arange(config.prediction_horizon)

    # Clip indices to stay within bounds of the target array
    clipped_indices = jnp.clip(indices, 0, max_idx)

    future_targets = params.x_target[clipped_indices]  # type: ignore

    # Concatenate state and future targets
    return jnp.concatenate([jnp.array([state.x]), future_targets])


def get_action_space_size() -> int:
    """Get the size of the action space."""
    return 2  # Two discrete actions: 0 (decrease) and 1 (increase)


def get_obs_space_size(config: EnvConfig) -> int:
    """Get the size of the observation space."""
    return 1 + config.prediction_horizon  # [x, x_target_t, ..., x_target_t+prediction_horizon-1]


def make_env(
    config: EnvConfig | None = None,
    params: EnvParams | None = None,
    **kwargs,
) -> Environment:
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
        params = create_env_params(config=config, **kwargs)
    elif kwargs:
        print(
            "Warning: `params` object was provided, so keyword arguments " f"({list(kwargs.keys())}) will be ignored."
        )

    return Environment(
        step=jit_step,
        reset=jit_reset,
        get_action_space_size=get_action_space_size,
        default_params=params,
        config=config,
    )
