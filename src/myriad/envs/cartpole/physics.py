"""Pure stateless physics for the CartPole system.

This module contains the ground truth dynamics for the cart-pole system,
completely decoupled from any task-specific logic (rewards, terminations, observations).

The physics can be reused by different tasks (control, SysID, etc.) and can be
directly accessed by model-based methods like MPC planners or Neural ODEs.
"""

from typing import NamedTuple

import chex
import jax.numpy as jnp
from flax import struct


class PhysicsState(NamedTuple):
    """Pure physical state of the cart-pole system.

    For CartPole, this is a fully observable system, so PhysicsState
    serves as both the internal state and the observation. This eliminates
    duplication and makes observability explicit.

    Attributes:
        x: Cart position (m)
        x_dot: Cart velocity (m/s)
        theta: Pole angle from vertical (rad, 0 = upright)
        theta_dot: Pole angular velocity (rad/s)
    """

    x: chex.Array
    x_dot: chex.Array
    theta: chex.Array
    theta_dot: chex.Array

    def to_array(self) -> chex.Array:
        """Convert to flat array for NN-based agents.

        Returns:
            Array of shape (4,) with [x, x_dot, theta, theta_dot]
        """
        return jnp.stack([self.x, self.x_dot, self.theta, self.theta_dot])

    @classmethod
    def from_array(cls, arr: chex.Array) -> "PhysicsState":
        """Create state from flat array.

        Args:
            arr: Array of shape (4,) with [x, x_dot, theta, theta_dot]

        Returns:
            PhysicsState instance
        """
        chex.assert_shape(arr, (4,))
        return cls(
            x=arr[0],  # type: ignore
            x_dot=arr[1],  # type: ignore
            theta=arr[2],  # type: ignore
            theta_dot=arr[3],  # type: ignore
        )


@struct.dataclass
class PhysicsConfig:
    """Static physics constants for the cart-pole system.

    These are compile-time constants passed as static_argnames to jit.
    Changing these values requires recompilation but enables better optimization.
    """

    gravity: float = 9.8  # m/s^2
    cart_mass: float = 1.0  # kg
    pole_mass: float = 0.1  # kg
    pole_length: float = 0.5  # m (half-length of the pole)
    force_magnitude: float = 10.0  # N
    dt: float = 0.02  # s (integration timestep)


@struct.dataclass
class PhysicsParams:
    """Dynamic physics parameters for domain randomization.

    Currently empty for CartPole, but maintained for protocol consistency
    and future domain randomization support (e.g., varying masses/lengths per episode).
    """

    ...


def step_physics(
    state: PhysicsState,
    action: chex.Array,
    params: PhysicsParams,
    config: PhysicsConfig,
) -> PhysicsState:
    """Pure physics step using Euler integration.

    This function computes the next physical state given the current state and action.
    It contains NO task logic: no rewards, terminations, or observations.

    The cart-pole dynamics are based on the equations from Barto, Sutton, Anderson (1983).

    Args:
        state: Current physical state (x, x_dot, theta, theta_dot)
        action: Discrete action {0, 1} representing force direction
        params: Dynamic parameters (unused, reserved for future randomization)
        config: Static physics constants

    Returns:
        Next physical state after one dt timestep
    """
    x, x_dot, theta, theta_dot = state

    # Convert discrete action {0, 1} to continuous force {-1, +1} * force_magnitude
    force = (2 * action - 1) * config.force_magnitude

    # Cart-pole dynamics
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    total_mass = config.cart_mass + config.pole_mass
    pole_mass_length = config.pole_mass * config.pole_length

    # Intermediate calculation for angular acceleration
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

    return PhysicsState(x_next, x_dot_next, theta_next, theta_dot_next)


def create_physics_params(**kwargs) -> PhysicsParams:
    """Factory function to create PhysicsParams.

    Args:
        **kwargs: Reserved for future domain randomization parameters

    Returns:
        PhysicsParams instance
    """
    return PhysicsParams()
