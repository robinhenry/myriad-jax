from typing import Any, Tuple

import chex
from flax import struct

from aion.core.spaces import Space

from .agent import Agent


@struct.dataclass
class AgentParams:
    """Static parameters for the random agent."""

    action_space: Space


@struct.dataclass
class AgentState:
    """A random agent has no state"""

    ...


def _init(_key: chex.PRNGKey, _sample_obs: chex.Array, _params: AgentParams) -> AgentState:
    return AgentState()


def _select_action(
    key: chex.PRNGKey,
    _obs: chex.Array,
    agent_state: AgentState,
    params: AgentParams,
) -> Tuple[chex.Array, AgentState]:
    return params.action_space.sample(key), agent_state


def _update(
    _key: chex.PRNGKey, agent_state: AgentState, _transition: Any, _params: AgentParams
) -> Tuple[AgentState, dict]:
    return agent_state, {}


def make_agent(action_space: Space) -> Agent:
    """Factory function to create an instance of the RandomAgent."""

    # Create the default parameters for this agent
    params = AgentParams(action_space=action_space)

    return Agent(
        params=params,
        init=_init,
        select_action=_select_action,
        update=_update,
    )
