"""Pure stateless physics for the Pendulum system.

This module contains the ground truth dynamics for the simple pendulum,
completely decoupled from any task-specific logic (rewards, terminations, observations).

The physics can be reused by different tasks (control, system ID, etc.) and can be
directly accessed by model-based methods like MPC planners or Neural ODEs.
"""

from typing import NamedTuple

import chex
import jax.numpy as jnp
from flax import struct
from jax import Array


class PhysicsState(NamedTuple):
    """Pure physical state of the pendulum system.

    Attributes:
        theta: Angle from vertical down (rad, 0 = hanging down, pi = upright)
        theta_dot: Angular velocity (rad/s)
    """

    theta: Array
    theta_dot: Array

    def to_array(self) -> Array:
        """Convert to flat array for NN-based agents.

        Returns:
            Array of shape (2,) with [theta, theta_dot]
        """
        return jnp.stack([self.theta, self.theta_dot])

    @classmethod
    def from_array(cls, arr: Array) -> "PhysicsState":
        """Create state from flat array.

        Args:
            arr: Array of shape (2,) with [theta, theta_dot]

        Returns:
            PhysicsState instance
        """
        chex.assert_shape(arr, (2,))
        return cls(
            theta=arr[0],  # type: ignore
            theta_dot=arr[1],  # type: ignore
        )


@struct.dataclass
class PhysicsConfig:
    """Static physics constants for the pendulum system.

    These are compile-time constants passed as static_argnames to jit.
    Changing these values requires recompilation but enables better optimization.
    """

    gravity: float = 9.8  # m/s^2
    mass: float = 1.0  # kg
    length: float = 1.0  # m
    dt: float = 0.05  # s (integration timestep)
    max_torque: float = 2.0  # N*m
    max_speed: float = 8.0  # rad/s


@struct.dataclass
class PhysicsParams:
    """Dynamic physics parameters for domain randomization.

    Currently empty for Pendulum, but maintained for protocol consistency
    and future domain randomization support (e.g., varying masses/lengths per episode).
    """

    ...


def step_physics(
    state: PhysicsState,
    action: Array,
    params: PhysicsParams,
    config: PhysicsConfig,
) -> PhysicsState:
    """Pure physics step using Euler integration.

    The pendulum dynamics follow the standard equation:
    theta_ddot = (3g/2l)*sin(theta) + (3/ml^2)*tau

    where theta=0 is hanging down and tau is the applied torque.

    Args:
        state: Current physical state (theta, theta_dot)
        action: Continuous torque in [-max_torque, max_torque]
        params: Dynamic parameters (unused, reserved for future randomization)
        config: Static physics constants

    Returns:
        Next physical state after one dt timestep
    """
    theta, theta_dot = state

    # Clip action to valid torque range
    torque = jnp.clip(action, -config.max_torque, config.max_torque)

    # Pendulum dynamics: theta_ddot = (3g/2l)*sin(theta) + (3/ml^2)*tau
    # Note: sin(theta) term is positive because theta=0 is hanging down
    theta_ddot = (3.0 * config.gravity / (2.0 * config.length)) * jnp.sin(theta) + (
        3.0 / (config.mass * config.length**2)
    ) * torque

    # Euler integration
    theta_dot_next = theta_dot + config.dt * theta_ddot
    theta_next = theta + config.dt * theta_dot_next  # Semi-implicit Euler

    # Clip velocity to max_speed
    theta_dot_next = jnp.clip(theta_dot_next, -config.max_speed, config.max_speed)

    return PhysicsState(theta_next, theta_dot_next)


def create_physics_params(**kwargs) -> PhysicsParams:
    """Factory function to create PhysicsParams.

    Args:
        **kwargs: Reserved for future domain randomization parameters

    Returns:
        PhysicsParams instance
    """
    return PhysicsParams()
