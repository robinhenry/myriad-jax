from typing import Any, Callable, NamedTuple

import chex

# AgentState can be any PyTree (e.g., a Flax TrainState or a simple NamedTuple)
AgentState = Any
AgentParams = Any


class Agent(NamedTuple):
    """A container for the pure functions that define a JAX-based agent."""

    init: Callable[[chex.PRNGKey, chex.Array, AgentParams], AgentState]
    select_action: Callable[[chex.PRNGKey, chex.Array, AgentState, AgentParams], tuple[chex.Array, AgentState]]
    update: Callable[[chex.PRNGKey, AgentState, Any, AgentParams], tuple[AgentState, dict]]
    default_params: AgentParams
