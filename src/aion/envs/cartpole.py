"""CartPole environment implemented as pure JAX functions.

Classic cart-pole balancing problem where an agent must balance a pole on a cart
by applying left or right forces to the cart.
"""

from functools import partial
from typing import Any, Dict, NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from aion.core.spaces import Discrete

from .environment import Environment


@struct.dataclass
class EnvConfig:
    """Static configuration for the CartPole environment."""

    # Physics parameters
    gravity: float = 9.8
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_length: float = 0.5  # Half-length of the pole
    force_magnitude: float = 10.0
    dt: float = 0.02  # Time step for physics integration

    # Termination thresholds
    theta_threshold: float = 12.0 * 2 * jnp.pi / 360  # ~12 degrees in radians
    x_threshold: float = 2.4

    # Episode settings
    max_steps: int = 500


@struct.dataclass
class EnvParams:
    """Dynamic parameters for the CartPole environment.

    Currently empty as CartPole typically doesn't need randomized parameters,
    but kept for consistency with the protocol.
    """

    pass


class EnvState(NamedTuple):
    """Environment state for CartPole.

    Attributes:
        x: Cart position
        x_dot: Cart velocity
        theta: Pole angle (0 = upright)
        theta_dot: Pole angular velocity
        t: Current timestep
    """

    x: chex.Array
    x_dot: chex.Array
    theta: chex.Array
    theta_dot: chex.Array
    t: chex.Array


def create_env_params() -> EnvParams:
    """Factory function to create EnvParams."""
    return EnvParams()


def _step(
    key: chex.PRNGKey, state: EnvState, action: chex.Array, params: EnvParams, config: EnvConfig
) -> tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
    """Step the environment forward one time step using Euler integration."""
    x, x_dot, theta, theta_dot, t = state

    # Convert action from {0, 1} to force direction {-1, 1}
    force = jnp.where(action == 0, -config.force_magnitude, config.force_magnitude)

    # Calculate physics (equations from OpenAI Gym CartPole-v1)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    total_mass = config.cart_mass + config.pole_mass
    pole_mass_length = config.pole_mass * config.pole_length

    # Temporary variable for angular acceleration calculation
    temp = (force + pole_mass_length * theta_dot**2 * sin_theta) / total_mass

    # Angular acceleration
    theta_acc = (config.gravity * sin_theta - cos_theta * temp) / (
        config.pole_length * (4.0 / 3.0 - config.pole_mass * cos_theta**2 / total_mass)
    )

    # Linear acceleration
    x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass

    # Euler integration
    x_next = x + config.dt * x_dot
    x_dot_next = x_dot + config.dt * x_acc
    theta_next = theta + config.dt * theta_dot
    theta_dot_next = theta_dot + config.dt * theta_acc

    # Update time step
    t_next = t + 1

    # Check termination conditions
    theta_out_of_bounds = jnp.abs(theta_next) > config.theta_threshold
    x_out_of_bounds = jnp.abs(x_next) > config.x_threshold
    max_steps_reached = t_next >= config.max_steps

    done = jnp.where(theta_out_of_bounds | x_out_of_bounds | max_steps_reached, 1.0, 0.0)

    # Reward: +1 for each timestep the pole is balanced
    reward = jnp.array(1.0)

    # Create next state and observation
    next_state = EnvState(x_next, x_dot_next, theta_next, theta_dot_next, t_next)
    obs_next = get_obs(next_state, params, config)

    return obs_next, next_state, reward, done, {}


def _reset(key: chex.PRNGKey, params: EnvParams, config: EnvConfig) -> tuple[chex.Array, EnvState]:
    """Reset the environment to initial state with small random perturbations."""
    # Initialize state with small random values around zero
    init_state = jax.random.uniform(key, shape=(4,), minval=-0.05, maxval=0.05)

    state = EnvState(
        x=init_state[0],
        x_dot=init_state[1],
        theta=init_state[2],
        theta_dot=init_state[3],
        t=jnp.array(0),
    )

    obs = get_obs(state, params, config)
    return obs, state


@partial(jax.jit, static_argnames=["config"])
def step(
    key: chex.PRNGKey, state: EnvState, action: chex.Array, params: EnvParams, config: EnvConfig
) -> tuple[chex.Array, EnvState, chex.Array, chex.Array, Dict[str, Any]]:
    return _step(key, state, action, params, config)


@partial(jax.jit, static_argnames=["config"])
def reset(key: chex.PRNGKey, params: EnvParams, config: EnvConfig) -> tuple[chex.Array, EnvState]:
    return _reset(key, params, config)


def get_obs(state: EnvState, params: EnvParams, config: EnvConfig) -> chex.Array:
    """Get observation from state.

    Returns a 4D vector: [x, x_dot, theta, theta_dot]
    """
    return jnp.array([state.x, state.x_dot, state.theta, state.theta_dot])


def get_obs_shape(_config: EnvConfig) -> tuple:
    """Get the shape of the observation space."""
    return (4,)


def get_action_space(_config: EnvConfig) -> Discrete:
    """Get the discrete action space for the environment.

    Returns:
        Discrete space with 2 actions: 0 (push left) and 1 (push right)
    """
    return Discrete(n=2)


def make_env(
    config: EnvConfig | None = None,
    params: EnvParams | None = None,
    **kwargs,
) -> Environment[EnvState, EnvParams, EnvConfig]:
    """Create a CartPole environment instance.

    Args:
        config: A custom EnvConfig object. If None, a default one is created.
        params: A pre-constructed EnvParams object. If None, a default one is created.
        **kwargs: Currently unused, but kept for consistency with other environments.

    Returns:
        An Environment object containing the CartPole environment.
    """
    if config is None:
        config = EnvConfig()

    if params is None:
        params = create_env_params()
    elif kwargs:
        print(f"Warning: `params` object was provided, so keyword arguments ({list(kwargs.keys())}) will be ignored.")

    return Environment(
        params=params,
        config=config,
        get_action_space=get_action_space,
        get_obs_shape=get_obs_shape,
        reset=reset,
        step=step,
    )
