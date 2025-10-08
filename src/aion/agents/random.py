"""A fully functional implementation of a simple random agent."""

from typing import Any, Tuple

import chex
import jax
from flax.struct import dataclass

from .base import Agent, AgentState


@dataclass
class RandomAgentParams:
    """Static parameters for the random agent."""

    num_actions: int


def _init(key: chex.PRNGKey, sample_obs: chex.Array, params: RandomAgentParams) -> AgentState:
    """Initializes the agent's state. For a random agent, there is no state."""
    # The key is unused but required by the protocol.
    _ = key, sample_obs, params
    return None


def _select_action(
    key: chex.PRNGKey,
    observation: chex.Array,
    agent_state: AgentState,
    params: RandomAgentParams,
) -> Tuple[chex.Array, AgentState]:
    """Selects a random action from the discrete action space."""
    # The observation is unused but required by the protocol.
    _ = observation

    action = jax.random.randint(key, shape=(), minval=0, maxval=params.num_actions)
    return action, agent_state


def _update(
    key: chex.PRNGKey, agent_state: AgentState, transition: Any, params: RandomAgentParams
) -> Tuple[AgentState, dict]:
    """The random agent does not learn, so this method does nothing."""
    # All arguments are unused but required by the protocol.
    _ = key, transition, params

    return agent_state, {}


def make_agent(num_actions: int) -> Agent:
    """Factory function to create an instance of the RandomAgent."""

    # Create the default parameters for this agent
    default_params = RandomAgentParams(num_actions=num_actions)

    return Agent(
        init=_init,
        select_action=_select_action,
        update=_update,
        default_params=default_params,
    )
