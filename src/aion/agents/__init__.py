from . import random
from .agent import Agent

# The registry mapping environment IDs to their factory functions
AGENT_REGISTRY = {
    "random_agent": random.make_agent,
}


def make_agent(agent_id: str, **kwargs) -> Agent:
    """
    A general factory function to create any registered agent.

    Args:
        agent_id: The string identifier of the agent to create.
        **kwargs: Keyword arguments that will be passed to the specific
                  agent's make_agent function.

    Returns:
        An instance of the requested Agent.

    Raises:
        ValueError: If the agent_id is not found in the registry.
    """
    if agent_id not in AGENT_REGISTRY:
        raise ValueError(
            f"Agent '{agent_id}' not found in the registry. Available agents: {list(AGENT_REGISTRY.keys())}"
        )

    # Look up the factory function and call it with the provided arguments
    make_fn = AGENT_REGISTRY[agent_id]
    return make_fn(**kwargs)
