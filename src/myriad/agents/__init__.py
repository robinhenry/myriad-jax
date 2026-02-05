from .agent import Agent, AgentParams, AgentState
from .classical import bangbang, pid, random
from .rl import dqn, pqn

# The registry mapping environment IDs to their factory functions
AGENT_REGISTRY = {
    "random": random.make_agent,
    "bangbang": bangbang.make_agent,
    "pid": pid.make_agent,
    "dqn": dqn.make_agent,
    "pqn": pqn.make_agent,
}

__all__ = [
    "make_agent",
    "Agent",
    "AgentParams",
    "AgentState",
    "AGENT_REGISTRY",
]


def make_agent(agent_id: str, **kwargs) -> Agent:
    """Create an agent by its registered ID.

    Args:
        agent_id: The string identifier of the agent to create.
            Available agents: ``"random"``, ``"bangbang"``, ``"pid"``, ``"dqn"``, ``"pqn"``.
        **kwargs: Keyword arguments passed to the agent's factory function.
            See individual agent modules for supported parameters.

    Returns:
        An instance of the requested Agent.

    Raises:
        ValueError: If ``agent_id`` is not found in the registry.

    Example:
        >>> from myriad.agents import make_agent
        >>> from myriad.core.spaces import Discrete
        >>> agent = make_agent("dqn", action_space=Discrete(2), learning_rate=1e-3)
    """
    if agent_id not in AGENT_REGISTRY:
        raise ValueError(
            f"Agent '{agent_id}' not found in the registry. Available agents: {list(AGENT_REGISTRY.keys())}"
        )

    # Look up the factory function and call it with the provided arguments
    make_fn = AGENT_REGISTRY[agent_id]
    return make_fn(**kwargs)  # type: ignore[operator]
