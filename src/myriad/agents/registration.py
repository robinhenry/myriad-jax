"""Agent registration and metadata.

This module provides a structured registry for agents, allowing the platform
to query agent properties (like on/off policy) without necessarily
instantiating them.
"""

from typing import Any, Callable, NamedTuple


class AgentInfo(NamedTuple):
    """Metadata for a registered agent.

    Attributes:
        name: The unique identifier for the agent.
        make_fn: The factory function to create the agent.
        is_on_policy: Whether the agent is an on-policy RL algorithm (e.g., PQN, PPO).
            On-policy agents typically require rollout collection before updates.
        is_off_policy: Whether the agent is an off-policy RL algorithm (e.g., DQN).
            Off-policy agents typically require a replay buffer.
    """

    name: str
    make_fn: Callable
    is_on_policy: bool = False
    is_off_policy: bool = False


# The global registry of agents
_AGENT_REGISTRY: dict[str, AgentInfo] = {}


def register_agent(
    name: str,
    make_fn: Callable,
    is_on_policy: bool = False,
    is_off_policy: bool = False,
) -> None:
    """Register an agent with metadata.

    Args:
        name: Unique identifier for the agent.
        make_fn: Factory function to create the agent.
        is_on_policy: Whether the agent is on-policy.
        is_off_policy: Whether the agent is off-policy.
    """
    _AGENT_REGISTRY[name] = AgentInfo(
        name=name,
        make_fn=make_fn,
        is_on_policy=is_on_policy,
        is_off_policy=is_off_policy,
    )


def get_agent_info(name: str) -> AgentInfo | None:
    """Get metadata for a registered agent.

    Args:
        name: Unique identifier for the agent.

    Returns:
        AgentInfo if registered, None otherwise.
    """
    return _AGENT_REGISTRY.get(name)


def list_agents() -> list[str]:
    """List all registered agent identifiers.

    Returns:
        List of agent names.
    """
    return list(_AGENT_REGISTRY.keys())


def make_agent(name: str, **kwargs: Any) -> Any:
    """Create an agent instance by name.

    Args:
        name: Unique identifier for the agent.
        **kwargs: Keyword arguments passed to the agent's factory function.

    Returns:
        An instance of the requested Agent.

    Raises:
        ValueError: If the agent name is not found in the registry.
    """
    info = get_agent_info(name)
    if info is None:
        available = ", ".join(list_agents())
        raise ValueError(f"Agent '{name}' not found in the registry. Available agents: {available}")

    return info.make_fn(**kwargs)
