"""Shared utilities for CcaS-CcaR gene circuit task wrappers."""


import chex
import jax.numpy as jnp
from flax import struct

from myriad.core.spaces import Discrete

from ..physics import PhysicsState


@struct.dataclass
class TaskConfig:
    """Base configuration shared by all CcaS-CcaR tasks.

    These define the task-specific episode limits and observation normalization.
    """

    max_steps: int = 288  # 288 steps * 5 min/step = 24 hours
    F_obs_normalizer: float = 80.0  # Normalization constant for F observations


def check_termination(t: chex.Array, task_config: TaskConfig) -> chex.Array:
    """Common termination check for CcaS-CcaR tasks.

    The episode terminates when maximum timesteps is reached.
    Unlike CartPole, there are no early termination conditions based on state.

    Args:
        t: Current timestep counter
        task_config: TaskConfig with max_steps

    Returns:
        done: 1.0 if terminated, 0.0 otherwise (as float for JAX compatibility)
    """
    done = (t >= task_config.max_steps).astype(jnp.float32)
    return done


def get_ccas_ccar_action_space() -> Discrete:
    """Get the discrete action space for CcaS-CcaR.

    Returns:
        Discrete space with 2 actions: 0 (light off) and 1 (light on)
    """
    return Discrete(n=2)


def sample_initial_physics(key: chex.PRNGKey) -> PhysicsState:
    """Sample initial physics state.

    For the gene circuit, we start from zero proteins at time 0.
    This represents the initial state before any light input.

    Args:
        key: RNG key for random initialization (unused currently)

    Returns:
        PhysicsState initialized to zero concentrations
    """
    return PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(0.0),
        F=jnp.array(0.0),
    )


def generate_constant_target(
    n_horizon: int,
    F_target_value: float,
) -> chex.Array:
    """Generate a constant target trajectory.

    Args:
        n_horizon: Number of future timesteps to predict
        F_target_value: Constant target value for F

    Returns:
        Array of shape (n_horizon + 1,) with constant target values
    """
    return jnp.full(n_horizon + 1, F_target_value, dtype=jnp.float32)


def generate_sinewave_target(
    key: chex.PRNGKey,
    t: chex.Array,
    n_horizon: int,
    timestep_minutes: float,
    period_minutes: float = 600.0,
    amplitude: float = 20.0,
    vshift: float = 30.0,
) -> chex.Array:
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
