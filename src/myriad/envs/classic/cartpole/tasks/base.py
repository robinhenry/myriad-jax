"""Shared utilities for CartPole task wrappers."""

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from myriad.core.spaces import Discrete
from myriad.core.types import PRNGKey

from ..physics import PhysicsState


@struct.dataclass
class TaskConfig:
    """Base configuration shared by all CartPole tasks.

    These define the task-specific termination conditions and episode limits.
    """

    max_steps: int = 500
    theta_threshold: float = 0.2094395102393195  # 12 degrees in radians
    x_threshold: float = 2.4  # meters


def check_termination(physics_state: PhysicsState, t: Array, task_config: TaskConfig) -> Array:
    """Common termination check for CartPole tasks.

    The episode terminates if:
    - Pole angle exceeds threshold (falls over)
    - Cart position exceeds threshold (goes off track)
    - Maximum timestep reached

    Args:
        physics_state: PhysicsState with x and theta fields
        t: Current timestep counter
        task_config: TaskConfig with thresholds and max_steps

    Returns:
        done: 1.0 if terminated, 0.0 otherwise (as float for JAX compatibility)
    """
    theta_out_of_bounds = jnp.abs(physics_state.theta) > task_config.theta_threshold
    x_out_of_bounds = jnp.abs(physics_state.x) > task_config.x_threshold
    max_steps_reached = t >= task_config.max_steps

    done = (theta_out_of_bounds | x_out_of_bounds | max_steps_reached).astype(jnp.float32)
    return done


def get_cartpole_obs(physics_state: PhysicsState) -> PhysicsState:
    """Extract standard CartPole observation from physics state.

    For CartPole, the system is fully observable, so observation = state.
    This eliminates duplication and makes observability explicit.

    Args:
        physics_state: PhysicsState with x, x_dot, theta, theta_dot fields

    Returns:
        PhysicsState (observation = state for fully observable system)
    """
    return physics_state


def get_cartpole_obs_shape() -> tuple[int, ...]:
    """Get the shape of the CartPole observation space.

    Returns:
        Observation shape tuple (4,) for [x, x_dot, theta, theta_dot]
    """
    return (4,)


def get_cartpole_action_space() -> Discrete:
    """Get the discrete action space for CartPole.

    Returns:
        Discrete space with 2 actions: 0 (push left) and 1 (push right)
    """
    return Discrete(n=2)


def sample_initial_physics(key: PRNGKey):
    """Sample initial physics state with small random perturbations.

    Initializes the pole with small random perturbations around the upright equilibrium position.

    Args:
        key: RNG key for random initialization

    Returns:
        PhysicsState with small perturbations in range [-0.05, 0.05] for all state variables
    """

    init_values = jax.random.uniform(key, shape=(4,), minval=-0.05, maxval=0.05)

    return PhysicsState(
        x=init_values[0],
        x_dot=init_values[1],
        theta=init_values[2],
        theta_dot=init_values[3],
    )
