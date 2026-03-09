"""Shared utilities for CcaS-CcaR + GFP gene circuit task wrappers."""

from typing import NamedTuple

import chex
import jax.numpy as jnp
from flax import struct
from jax import Array

from myriad.core.spaces import Discrete
from myriad.core.types import PRNGKey

from ..physics import PhysicsConfig, PhysicsParams, PhysicsState, step_physics


class CcasrGfpControlObs(NamedTuple):
    """CcaS-CcaR control task observation with named fields.

    Note: This is a partially observable system. The agent does not directly
    observe the light input (U) or the CcaSR concentration (H).

    Attributes:
        F_normalized: GFP fluorescence normalized by F_obs_normalizer
        F_target: Target trajectory [current, t+1, ..., t+n_horizon]
    """

    F_normalized: Array
    F_target: Array

    def to_array(self) -> Array:
        """Convert to flat array for NN-based agents.

        Returns:
            Array of shape (1 + n_horizon + 1,) with [F, F_target...]
        """
        return jnp.concatenate([jnp.array([self.F_normalized]), self.F_target])

    @classmethod
    def from_array(cls, arr: Array) -> "CcasrGfpControlObs":
        """Create observation from flat array.

        Args:
            arr: Array of shape (1 + n_horizon + 1,) with [F, F_target...]

        Returns:
            CcasCcarControlObs instance
        """
        chex.assert_rank(arr, 1)
        return cls(
            F_normalized=arr[0],  # type: ignore
            F_target=arr[1:],  # type: ignore
        )


@struct.dataclass
class TaskConfig:
    """Base configuration shared by all CcaS-CcaR tasks.

    These define the task-specific episode limits and observation normalization.
    """

    max_steps: int = 288  # 288 steps * 5 min/step = 24 hours
    F_obs_normalizer: float = 80.0  # Normalization constant for F observations


@struct.dataclass
class BaseCcasrGfpTaskConfig:
    """Shared config fields for all CcaS-CcaR task wrappers.

    Provides physics config, task config, and the max_steps protocol property
    in one place so individual task configs don't repeat them.
    """

    physics: PhysicsConfig = struct.field(default_factory=PhysicsConfig)
    task: TaskConfig = struct.field(default_factory=TaskConfig)

    @property
    def max_steps(self) -> int:
        """Required by EnvironmentConfig protocol."""
        return self.task.max_steps


def step_physics_interval(
    key: PRNGKey,
    physics: PhysicsState,
    t: Array,
    prev_action: Array,
    action: Array,
    params: PhysicsParams,
    config: PhysicsConfig,
) -> tuple[PhysicsState, Array]:
    """Step physics forward one interval and return (next_physics, t + 1).

    Centralises the interval_start calculation and step_physics call that
    every task _step function would otherwise duplicate.
    """
    interval_start = t * config.timestep_minutes
    next_physics = step_physics(
        key,
        physics,
        action,
        params,
        config,
        previous_action=prev_action,
        interval_start=interval_start,
    )
    return next_physics, t + 1


def check_termination(t: Array, task_config: TaskConfig) -> Array:
    """Common termination check for CcaS-CcaR tasks.

    The episode terminates when maximum timesteps is reached.

    Args:
        t: Current timestep counter
        task_config: TaskConfig with max_steps

    Returns:
        done: 1.0 if terminated, 0.0 otherwise (as float for JAX compatibility)
    """
    return (t >= task_config.max_steps).astype(jnp.float32)


def get_action_space() -> Discrete:
    """Get the discrete action space for CcaS-CcaR.

    Returns:
        Discrete space with 2 actions: 0 (red light) and 1 (green light)
    """
    return Discrete(n=2)


def sample_initial_physics(key: PRNGKey) -> PhysicsState:
    """Sample initial physics state.

    We start from zero proteins at time 0. This represents the initial state before
    any light input.

    Args:
        key: RNG key for random initialization (unused)

    Returns:
        PhysicsState initialized to zero concentrations
    """
    return PhysicsState.create(
        time=jnp.array(0.0),
        H=jnp.array(0.0),
        F=jnp.array(0.0),
    )


def generate_constant_target(
    n_horizon: int,
    F_target_value: float,
) -> Array:
    """Generate a constant target trajectory.

    Args:
        n_horizon: Number of future timesteps to predict
        F_target_value: Constant target value for F

    Returns:
        Array of shape (n_horizon + 1,) with constant target values
    """
    return jnp.full(n_horizon + 1, F_target_value, dtype=jnp.float32)


def generate_sinewave_target(
    key: PRNGKey,
    t: Array,
    n_horizon: int,
    timestep_minutes: float,
    period_minutes: float = 600.0,
    amplitude: float = 20.0,
    vshift: float = 30.0,
) -> Array:
    """Generate a sinusoidal target trajectory.

    Creates a time-varying target that follows a sine wave pattern.
    Used for testing tracking performance on dynamic targets.

    Args:
        key: RNG key for random phase initialization
        t: Current timestep counter
        n_horizon: Number of future timesteps to predict
        timestep_minutes: Duration of each RL timestep in minutes
        period_minutes: Period of the sine wave in minutes (default: 600 = 10 hours)
        amplitude: Amplitude of the sine wave (default: 20)
        vshift: Vertical shift / DC offset (default: 30)

    Returns:
        Array of shape (n_horizon + 1,) with sinusoidal target values
    """
    # Convert timestep to actual time in minutes
    current_time_minutes = t * timestep_minutes

    # Generate future time points
    future_steps = jnp.arange(n_horizon + 1)
    future_times = current_time_minutes + future_steps * timestep_minutes

    # Compute sine wave: vshift + amplitude * sin(2π * time / period)
    omega = 2.0 * jnp.pi / period_minutes
    targets = vshift + amplitude * jnp.sin(omega * future_times)

    return targets
