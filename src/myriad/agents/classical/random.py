from typing import Any, Tuple

from flax import struct
from jax import Array

from myriad.core.spaces import Space
from myriad.core.types import Observation, PRNGKey

from ..agent import Agent


@struct.dataclass
class AgentParams:
    """Static parameters for the random agent."""

    action_space: Space


@struct.dataclass
class AgentState:
    """A random agent has no state"""

    ...


def _init(key: PRNGKey, sample_obs: Observation, params: AgentParams) -> AgentState:
    """Initialize a random agent"""
    return AgentState()


def _select_action(
    key: PRNGKey,
    obs: Observation,
    state: AgentState,
    params: AgentParams,
    deterministic: bool,
) -> Tuple[Array, AgentState]:
    """Select a random action (deterministic flag ignored for random agent)."""
    return params.action_space.sample(key), state


def _update(key: PRNGKey, state: AgentState, batch: Any, params: AgentParams) -> Tuple[AgentState, dict]:
    return state, {}


def make_agent(action_space: Space) -> Agent[AgentState, AgentParams, Observation]:
    """Factory function to create an instance of the RandomAgent."""
    params = AgentParams(action_space=action_space)

    return Agent(
        params=params,
        init=_init,
        select_action=_select_action,
        update=_update,
    )
