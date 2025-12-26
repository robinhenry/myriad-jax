"""Bang-bang controller agent for JAX-based RL environments.

A deterministic, stateless control policy that switches between two action values
based on a threshold comparison of a selected observation field.

Control Logic:
    - If obs[obs_field] <= threshold: Select "low" action
    - If obs[obs_field] > threshold: Select "high" action

Action Space Behavior:
    - Discrete(n): Low=0, High=n-1 (requires n >= 2)
    - Box: Low=action_space.low, High=action_space.high

This is a classical control strategy useful for:
    - Baseline comparisons with learned policies
    - Simple stabilization tasks
    - System identification experiments
    - Debugging environment dynamics

Note: This is a non-learning agent (update() does nothing).
"""

from typing import Any, Tuple

import chex
import jax.numpy as jnp
from flax import struct

from myriad.core.spaces import Box, Discrete, Space
from myriad.utils.observations import get_field_index, to_array

from .agent import Agent


@struct.dataclass
class AgentParams:
    """Static parameters for the bang-bang controller agent.

    Attributes:
        action_space: Action space (Box or Discrete)
        threshold: Switching threshold for bang-bang control
        obs_field: Field name from observation NamedTuple to use for threshold comparison
        low_action: Pre-computed low action value (for JIT efficiency)
        high_action: Pre-computed high action value (for JIT efficiency)
    """

    action_space: Space = struct.field(pytree_node=False)
    threshold: float
    obs_field: str = struct.field(pytree_node=False)
    low_action: chex.Array
    high_action: chex.Array


@struct.dataclass
class AgentState:
    """Bang-bang controller state.

    Attributes:
        obs_index: Array index corresponding to obs_field (computed at init time)
    """

    obs_index: int


def _init(_key: chex.PRNGKey, sample_obs: chex.Array, params: AgentParams) -> AgentState:
    """Initialize the bang-bang controller and compute observation index."""
    obs_index = get_field_index(sample_obs, params.obs_field)
    return AgentState(obs_index=obs_index)


def _select_action(
    _key: chex.PRNGKey,
    obs: chex.Array,
    agent_state: AgentState,
    params: AgentParams,
    deterministic: bool = False,
) -> Tuple[chex.Array, AgentState]:
    """Select bang-bang action based on observation threshold.

    For Box action spaces: Returns low bound if obs[field] <= threshold, else high bound
    For Discrete action spaces: Returns 0 if obs[field] <= threshold, else n-1

    Args:
        _key: Random key (unused, policy is deterministic)
        obs: Current observation (NamedTuple or array)
        agent_state: Current agent state (contains obs_index)
        params: Agent hyperparameters (contains threshold, pre-computed actions)
        deterministic: Ignored (bang-bang is always deterministic)

    Returns:
        Tuple of (action, unchanged agent_state)
    """
    # Convert observation to array (zero overhead if already array)
    obs_array = to_array(obs)

    # Extract the observation value at the specified field index
    obs_value = obs_array[agent_state.obs_index]

    # Use pre-computed action values for JIT efficiency (no Python control flow)
    action = jnp.where(obs_value > params.threshold, params.high_action, params.low_action)

    return action, agent_state


def _update(
    _key: chex.PRNGKey, agent_state: AgentState, _transition: Any, _params: AgentParams
) -> Tuple[AgentState, dict]:
    """Update the bang-bang controller (no learning, returns empty metrics)."""
    return agent_state, {}


def make_agent(
    action_space: Space,
    threshold: float = 0.0,
    obs_field: str = "theta",
) -> Agent:
    """Factory function to create a bang-bang controller agent.

    The agent will automatically detect the observation field index when initialized
    by introspecting the sample observation's NamedTuple structure.

    Args:
        action_space: Action space (Box or Discrete)
        threshold: Switching threshold. If obs[obs_field] > threshold, use "high" action.
                  Default 0.0.
        obs_field: Field name from observation NamedTuple to use for threshold comparison.
                  Default "theta" (pole angle for CartPole).

    Returns:
        Agent instance with bang-bang control policy
    """

    if not obs_field or not isinstance(obs_field, str):
        raise ValueError(f"obs_field must be a non-empty string, got {obs_field!r}")

    # Pre-compute action values once at initialization for JIT efficiency
    if isinstance(action_space, Discrete):
        low_action = jnp.array(0, dtype=action_space.dtype)
        high_action = jnp.array(action_space.n - 1, dtype=action_space.dtype)
    elif isinstance(action_space, Box):
        low_action = jnp.broadcast_to(action_space.low, action_space.shape)
        high_action = jnp.broadcast_to(action_space.high, action_space.shape)
    else:
        raise ValueError(f"Bang-bang control only supports Box and Discrete action space, got {type(action_space)}")

    # Create parameters
    params = AgentParams(
        action_space=action_space,
        threshold=threshold,
        obs_field=obs_field,
        low_action=low_action,
        high_action=high_action,
    )

    return Agent(
        params=params,
        init=_init,
        select_action=_select_action,
        update=_update,
    )
