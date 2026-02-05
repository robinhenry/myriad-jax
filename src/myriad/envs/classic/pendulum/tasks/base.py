"""Shared utilities for Pendulum task wrappers."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from myriad.core.spaces import Box
from myriad.core.types import PRNGKey

from ..physics import PhysicsConfig, PhysicsState


@struct.dataclass
class TaskConfig:
    """Base configuration shared by all Pendulum tasks.

    These define the task-specific episode limits.
    """

    max_steps: int = 200


class PendulumObservation(NamedTuple):
    """Observation for the pendulum system.

    Uses cos/sin representation to avoid angle discontinuity at +/-pi.
    This provides bounded values suitable for neural network inputs.

    Attributes:
        cos_theta: Cosine of angle from vertical down
        sin_theta: Sine of angle from vertical down
        theta_dot: Angular velocity (rad/s)
    """

    cos_theta: Array
    sin_theta: Array
    theta_dot: Array

    def to_array(self) -> Array:
        """Convert to flat array for NN-based agents.

        Returns:
            Array of shape (3,) with [cos_theta, sin_theta, theta_dot]
        """
        return jnp.stack([self.cos_theta, self.sin_theta, self.theta_dot])


def get_pendulum_obs(physics_state: PhysicsState) -> PendulumObservation:
    """Extract standard Pendulum observation from physics state.

    Converts angle to cos/sin representation to avoid discontinuity.

    Args:
        physics_state: PhysicsState with theta, theta_dot fields

    Returns:
        PendulumObservation with cos_theta, sin_theta, theta_dot
    """
    return PendulumObservation(
        cos_theta=jnp.cos(physics_state.theta),
        sin_theta=jnp.sin(physics_state.theta),
        theta_dot=physics_state.theta_dot,
    )


def get_pendulum_obs_shape() -> tuple[int, ...]:
    """Get the shape of the Pendulum observation space.

    Returns:
        Observation shape tuple (3,) for [cos_theta, sin_theta, theta_dot]
    """
    return (3,)


def get_pendulum_action_space(config: PhysicsConfig) -> Box:
    """Get the continuous action space for Pendulum.

    Args:
        config: Physics configuration with max_torque

    Returns:
        Box space for torque in [-max_torque, max_torque]
    """
    return Box(low=-config.max_torque, high=config.max_torque, shape=(1,))


def sample_initial_physics(key: PRNGKey) -> PhysicsState:
    """Sample initial physics state with random angle and velocity.

    Initializes the pendulum with random angle in [-pi, pi] and
    small random velocity.

    Args:
        key: RNG key for random initialization

    Returns:
        PhysicsState with random initial conditions
    """
    key1, key2 = jax.random.split(key)

    # Random angle in [-pi, pi]
    theta = jax.random.uniform(key1, shape=(), minval=-jnp.pi, maxval=jnp.pi)

    # Small random velocity in [-1, 1]
    theta_dot = jax.random.uniform(key2, shape=(), minval=-1.0, maxval=1.0)

    return PhysicsState(theta=theta, theta_dot=theta_dot)
