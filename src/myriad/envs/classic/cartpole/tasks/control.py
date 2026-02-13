"""Control task wrapper for CartPole.

Standard balancing task: Keep the pole upright for as long as possible.
Reward: +1 per timestep the pole remains balanced.
"""

from typing import Any, Dict, NamedTuple

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from myriad.core.spaces import Discrete
from myriad.core.types import PRNGKey
from myriad.envs.environment import Environment

from ..physics import PhysicsConfig, PhysicsParams, PhysicsState, create_physics_params, step_physics
from .base import (
    TaskConfig,
    check_termination,
    get_cartpole_action_space,
    get_cartpole_obs,
    get_cartpole_obs_shape,
    sample_initial_physics,
)


class ControlTaskState(NamedTuple):
    """State for the control task.

    Attributes:
        physics: The underlying physics state (x, x_dot, theta, theta_dot)
        t: Current timestep counter
    """

    physics: PhysicsState
    t: Array


@struct.dataclass
class ControlTaskConfig:
    """Configuration for the CartPole control task.

    Composed of physics config and task config for clean separation.
    """

    physics: PhysicsConfig = struct.field(default_factory=PhysicsConfig)
    task: TaskConfig = struct.field(default_factory=TaskConfig)

    @property
    def dt(self) -> float:
        """Timestep duration in seconds."""
        return self.physics.dt

    @property
    def max_steps(self) -> int:
        """Required by EnvironmentConfig protocol."""
        return self.task.max_steps


@struct.dataclass
class ControlTaskParams:
    """Parameters for the control task."""

    physics: PhysicsParams = struct.field(default_factory=PhysicsParams)


def _step(
    key: PRNGKey,
    state: ControlTaskState,
    action: Array,
    params: ControlTaskParams,
    config: ControlTaskConfig,
) -> tuple[PhysicsState, ControlTaskState, Array, Array, Dict[str, Any]]:
    """Step the control task forward one timestep.

    Args:
        key: RNG key (unused for deterministic control task, but part of protocol)
        state: Current task state
        action: Discrete action {0, 1}
        params: Task parameters
        config: Task configuration (static)

    Returns:
        obs_next: Next observation (PhysicsState = fully observable)
        next_state: Next environment state
        reward: Reward (+1.0 per step)
        done: Termination flag (1.0 if done, 0.0 otherwise)
        info: Empty dict (no auxiliary information)
    """
    # Step the pure physics
    next_physics = step_physics(state.physics, action, params.physics, config.physics)

    # Increment timestep
    t_next = state.t + 1

    # Check termination
    done = check_termination(next_physics, t_next, config.task)

    # Compute reward (standard CartPole: +1 per step)
    reward = jnp.float32(1.0)

    # Create next state
    next_state = ControlTaskState(physics=next_physics, t=t_next)

    # Extract observation
    obs_next = get_obs(next_state, params, config)

    return obs_next, next_state, reward, done, {}


def _reset(
    key: PRNGKey,
    params: ControlTaskParams,
    config: ControlTaskConfig,
) -> tuple[PhysicsState, ControlTaskState]:
    """Reset the control task to initial state.

    Initializes the pole with small random perturbations around the upright position.

    Args:
        key: RNG key for random initialization
        params: Task parameters
        config: Task configuration (static)

    Returns:
        obs: Initial observation (PhysicsState with named fields)
        state: Initial task state
    """
    # Sample initial physics state with small random perturbations
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
) -> PhysicsState:
    """Extract observation from state.

    For control task, observation is the physical state as a NamedTuple with named fields.
    Neural network agents can call `.to_array()` for flat array representation.

    Args:
        state: Current task state
        params: Task parameters (unused)
        config: Task configuration (unused)

    Returns:
        PhysicsState with named fields (x, x_dot, theta, theta_dot)
    """
    return get_cartpole_obs(state.physics)


def get_obs_shape(config: ControlTaskConfig) -> tuple[int, ...]:
    """Get the shape of the observation space.

    Args:
        config: Task configuration (unused)

    Returns:
        Observation shape tuple
    """
    return get_cartpole_obs_shape()


def get_action_space(config: ControlTaskConfig) -> Discrete:
    """Get the discrete action space for the environment.

    Args:
        config: Task configuration (unused)

    Returns:
        Discrete space with 2 actions: 0 (push left) and 1 (push right)
    """
    return get_cartpole_action_space()


def make_env(
    config: ControlTaskConfig | None = None,
    params: ControlTaskParams | None = None,
    **kwargs,
) -> Environment[ControlTaskState, ControlTaskConfig, ControlTaskParams, PhysicsState]:
    """Create a CartPole control task environment.

    Args:
        config: Custom ControlTaskConfig. If None, uses defaults.
        params: Custom ControlTaskParams. If None, creates from kwargs.
        **kwargs: Keyword arguments for creating config/params if not provided.

    Returns:
        Environment instance for the control task
    """
    if config is None:
        # Parse kwargs into nested config structure
        physics_fields = {"gravity", "cart_mass", "pole_mass", "pole_length", "force_magnitude", "dt"}
        task_fields = {"max_steps", "theta_threshold", "x_threshold"}

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
