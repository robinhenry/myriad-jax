"""Control task wrapper for Pendulum.

Standard swing-up task: Swing the pendulum to the upright position and balance.
Reward: -(theta_from_up^2 + 0.1*theta_dot^2 + 0.001*torque^2)
"""

from typing import Any, Dict, NamedTuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from myriad.core.spaces import Box
from myriad.core.types import PRNGKey
from myriad.envs.environment import Environment

from ..physics import (
    PhysicsConfig,
    PhysicsParams,
    create_physics_params,
    step_physics,
)
from .base import (
    PendulumObservation,
    TaskConfig,
    get_pendulum_action_space,
    get_pendulum_obs,
    get_pendulum_obs_shape,
    sample_initial_physics,
)


class ControlTaskState(NamedTuple):
    """State for the control task.

    Attributes:
        physics: The underlying physics state (theta, theta_dot)
        t: Current timestep counter
    """

    physics: "PhysicsState"  # noqa: F821 - forward reference
    t: Array


# Import PhysicsState after ControlTaskState definition to avoid circular import
from ..physics import PhysicsState  # noqa: E402


@struct.dataclass
class ControlTaskConfig:
    """Configuration for the Pendulum control task.

    Composed of physics config and task config for clean separation.
    """

    physics: PhysicsConfig = struct.field(default_factory=PhysicsConfig)
    task: TaskConfig = struct.field(default_factory=TaskConfig)

    @property
    def max_steps(self) -> int:
        """Required by EnvironmentConfig protocol."""
        return self.task.max_steps


@struct.dataclass
class ControlTaskParams:
    """Parameters for the control task."""

    physics: PhysicsParams = struct.field(default_factory=PhysicsParams)


def _angle_normalize(x: Array) -> Array:
    """Normalize angle to [-pi, pi].

    Args:
        x: Angle in radians

    Returns:
        Angle normalized to [-pi, pi]
    """
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def _step(
    key: PRNGKey,
    state: ControlTaskState,
    action: Array,
    params: ControlTaskParams,
    config: ControlTaskConfig,
) -> tuple[PendulumObservation, ControlTaskState, Array, Array, Dict[str, Any]]:
    """Step the control task forward one timestep.

    Args:
        key: RNG key (unused for deterministic control task, but part of protocol)
        state: Current task state
        action: Continuous torque in [-max_torque, max_torque]
        params: Task parameters
        config: Task configuration (static)

    Returns:
        obs_next: Next observation (PendulumObservation)
        next_state: Next environment state
        reward: Reward (negative cost)
        done: Termination flag (1.0 if done, 0.0 otherwise)
        info: Empty dict (no auxiliary information)
    """
    # Extract scalar torque from action array
    torque = jnp.squeeze(action)

    # Step the pure physics
    next_physics = step_physics(state.physics, torque, params.physics, config.physics)

    # Increment timestep
    t_next = state.t + 1

    # Check termination (no early termination, only max steps)
    done = (t_next >= config.task.max_steps).astype(jnp.float32)

    # Compute reward: -(theta_from_up^2 + 0.1*theta_dot^2 + 0.001*torque^2)
    # theta_from_up is angle from upright (pi from hanging)
    theta_from_up = _angle_normalize(next_physics.theta - jnp.pi)
    costs = theta_from_up**2 + 0.1 * next_physics.theta_dot**2 + 0.001 * torque**2
    reward = -costs

    # Create next state
    next_state = ControlTaskState(physics=next_physics, t=t_next)

    # Extract observation
    obs_next = get_obs(next_state, params, config)

    return obs_next, next_state, reward, done, {}


def _reset(
    key: PRNGKey,
    params: ControlTaskParams,
    config: ControlTaskConfig,
) -> tuple[PendulumObservation, ControlTaskState]:
    """Reset the control task to initial state.

    Initializes the pendulum with random angle and small velocity.

    Args:
        key: RNG key for random initialization
        params: Task parameters
        config: Task configuration (static)

    Returns:
        obs: Initial observation (PendulumObservation)
        state: Initial task state
    """
    # Sample initial physics state with random perturbations
    physics = sample_initial_physics(key)

    state = ControlTaskState(physics=physics, t=jnp.array(0))
    obs = get_obs(state, params, config)

    return obs, state


# JIT the step and reset functions with config as static argument
step = jax.jit(_step, static_argnames=["config"])
reset = jax.jit(_reset, static_argnames=["config"])


def get_obs(
    state: ControlTaskState,
    params: ControlTaskParams,
    config: ControlTaskConfig,
) -> PendulumObservation:
    """Extract observation from state.

    For control task, observation is cos/sin/theta_dot representation.
    Neural network agents can call `.to_array()` for flat array representation.

    Args:
        state: Current task state
        params: Task parameters (unused)
        config: Task configuration (unused)

    Returns:
        PendulumObservation with cos_theta, sin_theta, theta_dot
    """
    return get_pendulum_obs(state.physics)


def get_obs_shape(config: ControlTaskConfig) -> tuple[int, ...]:
    """Get the shape of the observation space.

    Args:
        config: Task configuration (unused)

    Returns:
        Observation shape tuple
    """
    return get_pendulum_obs_shape()


def get_action_space(config: ControlTaskConfig) -> Box:
    """Get the continuous action space for the environment.

    Args:
        config: Task configuration with physics config

    Returns:
        Box space for torque in [-max_torque, max_torque]
    """
    return get_pendulum_action_space(config.physics)


def make_env(
    config: ControlTaskConfig | None = None,
    params: ControlTaskParams | None = None,
    **kwargs,
) -> Environment[ControlTaskState, ControlTaskConfig, ControlTaskParams, PendulumObservation]:
    """Create a Pendulum control task environment.

    Args:
        config: Custom ControlTaskConfig. If None, uses defaults.
        params: Custom ControlTaskParams. If None, creates from kwargs.
        **kwargs: Keyword arguments for creating config/params if not provided.

    Returns:
        Environment instance for the control task
    """
    if config is None:
        # Parse kwargs into nested config structure
        physics_fields = {"gravity", "mass", "length", "dt", "max_torque", "max_speed"}
        task_fields = {"max_steps"}

        physics_kwargs = {k: v for k, v in kwargs.items() if k in physics_fields}
        task_kwargs = {k: v for k, v in kwargs.items() if k in task_fields}

        config = ControlTaskConfig(
            physics=PhysicsConfig(**physics_kwargs) if physics_kwargs else PhysicsConfig(),
            task=TaskConfig(**task_kwargs) if task_kwargs else TaskConfig(),
        )

    if params is None:
        params = ControlTaskParams(physics=create_physics_params())

    return Environment(
        step=step,
        reset=reset,
        get_action_space=get_action_space,
        get_obs_shape=get_obs_shape,
        params=params,
        config=config,
    )
