"""A random agent that samples actions uniformly at random from the action space.

Useful for establishing baseline performance and testing environments.

This is a non-learning agent: the ``update()`` function does nothing.
"""

from typing import Any, Tuple

from flax import struct
from jax import Array

from myriad.core.spaces import Space
from myriad.core.types import Observation, PRNGKey

from ..agent import Agent


@struct.dataclass
class AgentParams:
    """Static parameters for the random agent.

    Attributes:
        action_space: Action space to sample from.
    """

    action_space: Space


@struct.dataclass
class AgentState:
    """State of the random agent.

    The random agent is stateless, so this is an empty dataclass.
    """

    ...


def _init(key: PRNGKey, sample_obs: Observation, params: AgentParams) -> AgentState:
    """Initialize a random agent.

    Args:
        key: Random key (unused).
        sample_obs: Sample observation (unused).
        params: Agent parameters (unused).

    Returns:
        Empty AgentState.
    """
    return AgentState()


def _select_action(
    key: PRNGKey,
    obs: Observation,
    state: AgentState,
    params: AgentParams,
    deterministic: bool,
) -> Tuple[Array, AgentState]:
    """Select a random action from the action space.

    Args:
        key: Random key for sampling.
        obs: Current observation (unused).
        state: Current agent state (unchanged).
        params: Agent parameters containing action space.
        deterministic: Ignored (random agent always samples randomly).

    Returns:
        Tuple of (sampled action, unchanged state).
    """
    return params.action_space.sample(key), state


def _update(key: PRNGKey, state: AgentState, batch: Any, params: AgentParams) -> Tuple[AgentState, dict]:
    """No-op update for random agent (non-learning).

    Args:
        key: Random key (unused).
        state: Current agent state (returned unchanged).
        batch: Batch of experience (unused).
        params: Agent parameters (unused).

    Returns:
        Tuple of (unchanged state, empty metrics dict).
    """
    return state, {}


def make_agent(action_space: Space) -> Agent[AgentState, AgentParams, Observation]:
    """Factory function to create a random agent.

    Args:
        action_space: Action space to sample from (supports any Space type).

    Returns:
        Agent instance with random policy.
    """
    params = AgentParams(action_space=action_space)

    return Agent(
        params=params,
        init=_init,
        select_action=_select_action,
        update=_update,
    )
