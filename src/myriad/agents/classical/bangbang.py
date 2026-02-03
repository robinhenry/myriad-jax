"""A classical bang-bang controller agent.

A deterministic, stateless control policy that switches between two action values
based on a threshold comparison of a selected observation field.

Control Logic:
    - If obs[obs_field] <= threshold: Select "low" action
    - If obs[obs_field] > threshold: Select "high" action

Action Space Behavior:
    - Discrete(n): Low=0, High=n-1 (requires n >= 2)
    - Box: Low=action_space.low, High=action_space.high

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
    """Static parameters for the bang-bang controller agent.

    Attributes:
        action_space: Action space (Box or Discrete)
        threshold: Switching threshold for bang-bang control
        obs_field: Field name from observation NamedTuple to use for threshold comparison
        low_action: Pre-computed low action value (for JIT efficiency)
        high_action: Pre-computed high action value (for JIT efficiency)
        invert: If True, swap action selection (high when below threshold, low when above)
    """

    action_space: Space = struct.field(pytree_node=False)
    threshold: float
    obs_field: str = struct.field(pytree_node=False)
    low_action: Array
    high_action: Array
    invert: bool = False


@struct.dataclass
class AgentState:
    """Bang-bang controller state.

    Attributes:
        obs_index: Array index corresponding to obs_field (computed at init time)
    """

    obs_index: int


def _init(key: PRNGKey, sample_obs: Observation, params: AgentParams) -> AgentState:
    """Initialize the bang-bang controller and compute observation index."""
    obs_index = get_field_index(sample_obs, params.obs_field)
    return AgentState(obs_index=obs_index)


def _select_action(
    key: PRNGKey,
    obs: Observation,
    state: AgentState,
    params: AgentParams,
    deterministic: bool,
) -> Tuple[Array, AgentState]:
    """Select bang-bang action based on observation threshold.

    Normal mode (invert=False):
        - obs[field] <= threshold: low action
        - obs[field] > threshold: high action

    Inverted mode (invert=True):
        - obs[field] <= threshold: high action
        - obs[field] > threshold: low action

    Args:
        key: Random key (unused, policy is deterministic)
        obs: Current observation (NamedTuple-like)
        state: Current agent state (contains obs_index)
        params: Agent hyperparameters (contains threshold, pre-computed actions, invert flag)
        deterministic: Ignored (bang-bang is always deterministic)

    Returns:
        Tuple of (action, unchanged state)
    """
    # Convert observation to array (zero overhead if already array)
    obs_array = to_array(obs)

    # Extract the observation value at the specified field index
    obs_value = obs_array[state.obs_index]

    # Select action based on threshold comparison
    # Compute both normal and inverted actions, then select based on invert flag
    # This avoids Python control flow for JIT compatibility
    normal_action = jnp.where(obs_value > params.threshold, params.high_action, params.low_action)
    inverted_action = jnp.where(obs_value > params.threshold, params.low_action, params.high_action)

    # Select between normal and inverted based on params.invert (JAX-compatible)
    action = jnp.where(params.invert, inverted_action, normal_action)

    return action, state


def _update(key: PRNGKey, state: AgentState, batch: Any, params: AgentParams) -> Tuple[AgentState, dict]:
    """Update the bang-bang controller (no learning, returns empty metrics)."""
    return state, {}


def make_agent(
    action_space: Space,
    threshold: float = 0.0,
    obs_field: str = "theta",
    invert: bool = False,
) -> Agent[AgentState, AgentParams, Observation]:
    """Factory function to create a bang-bang controller agent.

    Args:
        action_space: Action space (Box or Discrete)
        threshold: Bang-bang witching threshold. Default 0.0.
        obs_field: Field name from observation NamedTuple to use for threshold comparison.
            Default "theta" (pole angle for CartPole).
        invert: If False (default): high action when obs > threshold. If True: low action when
            obs > threshold (swapped polarity)

    Returns:
        Agent instance with bang-bang control policy

    Example:
        >>> # Example: bang-bang controller for CartPole
        >>> # Normal: push right when pole tilts right
        >>> agent = make_agent(action_space, threshold=0.0, obs_field="theta")
        >>>
        >>> # Inverted: push left when pole tilts right (opposite polarity)
        >>> agent = make_agent(action_space, threshold=0.0, obs_field="theta", invert=True)
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
        invert=invert,
    )

    return Agent(
        params=params,
        init=_init,
        select_action=_select_action,
        update=_update,
    )
