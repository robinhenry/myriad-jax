"""PID controller agent for JAX-based RL environments.

A deterministic, stateful control policy that uses Proportional-Integral-Derivative
control to minimize error between observation and setpoint.

Control Law:
    u(t) = Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt
    where e(t) = setpoint - obs[obs_field]

This is a classical control strategy useful for:
    - Baseline comparisons with learned policies
    - Stabilization and tracking tasks
    - System identification experiments
    - Debugging environment dynamics

Note: This is a non-learning agent (update() does nothing).
"""

from typing import Any, Tuple

import chex
import jax.numpy as jnp
from flax import struct

from myriad.core.spaces import Box, Space
from myriad.utils.observations import get_field_index, to_array

from ..agent import Agent


@struct.dataclass
class AgentParams:
    """Static parameters for the PID controller agent.

    Attributes:
        action_space: Action space (must be Box for continuous control)
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        setpoint: Desired value for the controlled variable
        obs_field: Field name from observation NamedTuple to control
        dt: Time step for integral/derivative computation (seconds)
        anti_windup: Maximum absolute value for integral term (prevents windup)
    """

    action_space: Space = struct.field(pytree_node=False)
    kp: float
    ki: float
    kd: float
    setpoint: float
    obs_field: str = struct.field(pytree_node=False)
    dt: float = 0.02  # Default 50Hz
    anti_windup: float = 10.0


@struct.dataclass
class AgentState:
    """PID controller state.

    Attributes:
        obs_index: Array index corresponding to obs_field (computed at init time)
        integral_error: Accumulated integral of error over time
        previous_error: Error from previous timestep (for derivative term)
    """

    obs_index: int
    integral_error: chex.Array
    previous_error: chex.Array


def _init(_key: chex.PRNGKey, sample_obs: chex.Array, params: AgentParams) -> AgentState:
    """Initialize the PID controller and compute observation index."""
    obs_index = get_field_index(sample_obs, params.obs_field)
    return AgentState(
        obs_index=obs_index,
        integral_error=jnp.array(0.0),
        previous_error=jnp.array(0.0),
    )


def _select_action(
    _key: chex.PRNGKey,
    obs: chex.Array,
    agent_state: AgentState,
    params: AgentParams,
    deterministic: bool = False,  # noqa: ARG001
) -> Tuple[chex.Array, AgentState]:
    """Select PID control action based on error from setpoint.

    Computes: u(t) = Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt
    where e(t) = setpoint - obs[obs_field]

    Args:
        _key: Random key (unused, policy is deterministic)
        obs: Current observation (NamedTuple or array)
        agent_state: Current agent state (contains integral, previous error)
        params: Agent hyperparameters (PID gains, setpoint, etc.)
        deterministic: Ignored (PID is always deterministic)

    Returns:
        Tuple of (action, updated agent_state)
    """
    # Convert observation to array (zero overhead if already array)
    obs_array = to_array(obs)

    # Extract the observation value at the specified field index
    obs_value = obs_array[agent_state.obs_index]

    # Compute error: e(t) = setpoint - measurement
    error = params.setpoint - obs_value

    # Proportional term
    p_term = params.kp * error

    # Integral term with anti-windup
    integral = agent_state.integral_error + error * params.dt
    integral = jnp.clip(integral, -params.anti_windup, params.anti_windup)
    i_term = params.ki * integral

    # Derivative term
    derivative = (error - agent_state.previous_error) / params.dt
    d_term = params.kd * derivative

    # Compute control output
    control = p_term + i_term + d_term

    # Clip to action space bounds (cast to Box for type safety)
    assert isinstance(params.action_space, Box)
    action = jnp.clip(control, params.action_space.low, params.action_space.high)

    # Broadcast to action shape if needed
    action = jnp.broadcast_to(action, params.action_space.shape)

    # Update state
    new_state = AgentState(
        obs_index=agent_state.obs_index,
        integral_error=integral,
        previous_error=error,
    )

    return action, new_state


def _update(
    _key: chex.PRNGKey, agent_state: AgentState, _transition: Any, _params: AgentParams
) -> Tuple[AgentState, dict]:
    """Update the PID controller (no learning, returns empty metrics)."""
    return agent_state, {}


def make_agent(
    action_space: Space,
    kp: float = 1.0,
    ki: float = 0.0,
    kd: float = 0.0,
    setpoint: float = 0.0,
    obs_field: str = "theta",
    dt: float = 0.02,
    anti_windup: float = 10.0,
) -> Agent:
    """Factory function to create a PID controller agent.

    The agent will automatically detect the observation field index when initialized
    by introspecting the sample observation's NamedTuple structure.

    Args:
        action_space: Action space (must be Box for continuous control)
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
            Should match environment step rate. Default 0.02 (50Hz).
        anti_windup: Maximum absolute value for integral term to prevent windup.
                    Default 10.0.

    Returns:
        Agent instance with PID control policy

    Raises:
        ValueError: If action_space is not Box or obs_field is invalid
    """

    if not obs_field or not isinstance(obs_field, str):
        raise ValueError(f"obs_field must be a non-empty string, got {obs_field!r}")

    if not isinstance(action_space, Box):
        raise ValueError(f"PID control only supports Box action spaces, got {type(action_space)}")

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
    )

    return Agent(
        params=params,
        init=_init,
        select_action=_select_action,
        update=_update,
    )
