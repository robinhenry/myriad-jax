"""A classical PID controller agent.

A deterministic, stateful control policy that uses Proportional-Integral-Derivative
(PID) control to minimize the error between an observed variable and a setpoint.

Control Law:
    u(t) = Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt
    where e(t) = setpoint - obs[obs_field]

Action Space Behavior:
    - Box: Continuous control clipped to action_space bounds
    - Discrete(n): Continuous control discretized into n bins

Note: This is a non-learning agent (update() does nothing).
"""

from typing import Any, Tuple

import jax.numpy as jnp
from flax import struct
from jax import Array

from myriad.core.spaces import Box, Discrete, Space
from myriad.core.types import Observation, PRNGKey
from myriad.utils.observations import get_field_index, to_array

from ..agent import Agent


@struct.dataclass
class AgentParams:
    """Static parameters for the PID controller agent.

    Attributes:
        action_space: Action space (Box or Discrete)
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        setpoint: Desired value for the controlled variable
        obs_field: Field name from the observation NamedTuple to control (ie, to compare to the setpoint)
        dt: Time step for integral/derivative computation (seconds)
        anti_windup: Maximum absolute value for integral term (prevents windup)
        control_low: Lower bound for continuous control signal (for discretization)
        control_high: Upper bound for continuous control signal (for discretization)
        bin_edges: Bin edges for discretizing continuous control to discrete actions (Discrete only)
    """

    action_space: Space = struct.field(pytree_node=False)
    kp: float
    ki: float
    kd: float
    setpoint: float
    obs_field: str = struct.field(pytree_node=False)
    dt: float
    anti_windup: float
    control_low: float
    control_high: float
    bin_edges: Array | None = None


@struct.dataclass
class AgentState:
    """PID controller state.

    Attributes:
        obs_index: Array index corresponding to obs_field (computed at init time)
        integral_error: Accumulated integral of error over time
        previous_error: Error from previous timestep (for derivative term)
    """

    obs_index: int
    integral_error: Array
    previous_error: Array


def _init(key: PRNGKey, sample_obs: Observation, params: AgentParams) -> AgentState:
    """Initialize the PID controller and compute observation index."""
    obs_index = get_field_index(sample_obs, params.obs_field)
    return AgentState(
        obs_index=obs_index,
        integral_error=jnp.array(0.0),
        previous_error=jnp.array(0.0),
    )


def _select_action(
    key: PRNGKey,
    obs: Observation,
    state: AgentState,
    params: AgentParams,
    deterministic: bool,
) -> Tuple[Array, AgentState]:
    """Select PID control action based on error from setpoint.

    Computes: u(t) = Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt
    where e(t) = setpoint - obs[obs_field]

    For Box action spaces, clips the continuous control to bounds.
    For Discrete action spaces, discretizes the continuous control into bins.

    Args:
        key: Random key (unused, policy is deterministic)
        obs: Current observation (NamedTuple-like)
        state: Current agent state (contains integral, previous error)
        params: Agent hyperparameters (PID gains, setpoint, etc.)
        deterministic: Ignored (PID is always deterministic)

    Returns:
        Tuple of (action, updated agent_state)
    """
    # Convert observation to array (zero overhead if already array)
    obs_array = to_array(obs)

    # Extract the observation value at the specified field index
    obs_value = obs_array[state.obs_index]

    # Compute error: e(t) = setpoint - measurement
    error = params.setpoint - obs_value

    # Proportional term
    p_term = params.kp * error

    # Integral term with anti-windup
    integral = state.integral_error + error * params.dt
    integral = jnp.clip(integral, -params.anti_windup, params.anti_windup)
    i_term = params.ki * integral

    # Derivative term
    derivative = (error - state.previous_error) / params.dt
    d_term = params.kd * derivative

    # Compute continuous control output
    control = p_term + i_term + d_term

    # Convert continuous control to action based on action space type
    if isinstance(params.action_space, Box):
        # Continuous action: clip to bounds
        action = jnp.clip(control, params.action_space.low, params.action_space.high)
        action = jnp.broadcast_to(action, params.action_space.shape)
    elif isinstance(params.action_space, Discrete):
        # Discrete action: discretize continuous control into bins
        # Clip control to valid range first
        control_clipped = jnp.clip(control, params.control_low, params.control_high)
        # Use searchsorted to find the bin (digitize behavior)
        # bin_edges has n+1 edges for n actions, searchsorted returns values in [0, n]
        # We need to handle edge cases: values at edges map to valid actions
        assert params.bin_edges is not None, "bin_edges must be set for Discrete action spaces"
        action_idx = jnp.searchsorted(params.bin_edges, control_clipped, side="right") - 1
        # Clip to valid action range [0, n-1]
        action_idx = jnp.clip(action_idx, 0, params.action_space.n - 1)
        action = jnp.asarray(action_idx, dtype=params.action_space.dtype)
    else:
        raise ValueError(f"Unsupported action space type: {type(params.action_space)}")

    # Update state
    new_state = AgentState(
        obs_index=state.obs_index,
        integral_error=integral,
        previous_error=error,
    )

    return action, new_state


def _update(key: PRNGKey, state: AgentState, batch: Any, params: AgentParams) -> Tuple[AgentState, dict]:
    """Update the PID controller (no learning, returns empty metrics)."""
    return state, {}


def make_agent(
    action_space: Space,
    kp: float = 1.0,
    ki: float = 0.0,
    kd: float = 0.0,
    setpoint: float = 0.0,
    obs_field: str = "theta",
    dt: float = 1.0,
    anti_windup: float = 10.0,
    control_low: float | None = None,
    control_high: float | None = None,
) -> Agent[AgentState, AgentParams, Observation]:
    """Factory function to create a PID controller agent.

    Args:
        action_space: Action space (Box or Discrete)
        kp: Proportional gain. Higher values increase responsiveness but may cause
            oscillation. Default 1.0.
        ki: Integral gain. Eliminates steady-state error but may cause overshoot.
            Default 0.0 (no integral action).
        kd: Derivative gain. Dampens oscillations and improves stability.
            Default 0.0 (no derivative action).
        setpoint: Desired value for obs[obs_field]. Default 0.0.
        obs_field: Field name from observation NamedTuple to control.
            Default "theta" (pole angle for CartPole).
        dt: Time step in seconds for integral/derivative computation.
            Should match environment step rate. Default 1s (1Hz).
        anti_windup: Maximum absolute value for integral term to prevent windup.
            Default 10.0.
        control_low: Lower bound for continuous control signal. For Box action spaces,
            defaults to action_space.low. For Discrete, must be specified. Default None.
        control_high: Upper bound for continuous control signal. For Box action spaces,
            defaults to action_space.high. For Discrete, must be specified. Default None.

    Returns:
        Agent instance with PID control policy

    Raises:
        ValueError: If action_space is not Box/Discrete, obs_field is invalid,
            or control bounds are not specified for Discrete action space
    """

    if not obs_field or not isinstance(obs_field, str):
        raise ValueError(f"obs_field must be a non-empty string, got {obs_field!r}")

    # Determine control bounds and compute bin edges for discrete actions
    bin_edges = None
    if isinstance(action_space, Box):
        # For Box, use action space bounds as control bounds
        _control_low = control_low if control_low is not None else float(action_space.low)
        _control_high = control_high if control_high is not None else float(action_space.high)
    elif isinstance(action_space, Discrete):
        # For Discrete, control bounds must be explicitly specified
        if control_low is None or control_high is None:
            raise ValueError(
                "For Discrete action spaces, control_low and control_high must be specified. "
                "These define the continuous control range to discretize into bins."
            )
        _control_low = control_low
        _control_high = control_high
        # Pre-compute bin edges for discretization (n+1 edges for n actions)
        bin_edges = jnp.linspace(_control_low, _control_high, action_space.n + 1)
    else:
        raise ValueError(f"PID control only supports Box and Discrete action spaces, got {type(action_space)}")

    # Create parameters
    params = AgentParams(
        action_space=action_space,
        kp=kp,
        ki=ki,
        kd=kd,
        setpoint=setpoint,
        obs_field=obs_field,
        dt=dt,
        anti_windup=anti_windup,
        control_low=_control_low,
        control_high=_control_high,
        bin_edges=bin_edges,
    )

    return Agent(
        params=params,
        init=_init,
        select_action=_select_action,
        update=_update,
    )
