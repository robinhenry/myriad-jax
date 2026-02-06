import pytest

from myriad.agents import list_agents, make_agent
from myriad.agents.agent import Agent
from myriad.core.spaces import Box, Discrete


@pytest.fixture
def continuous_action_space() -> Box:
    return Box(low=-1.0, high=1.0, shape=(2,))


@pytest.fixture
def discrete_action_space() -> Discrete:
    return Discrete(n=4)


def test_unknown_agent_raises():
    """Unknown agent ID raises ValueError with helpful message."""
    with pytest.raises(ValueError, match="not found in the registry"):
        make_agent("nonexistent_agent")


def test_error_message_lists_available_agents():
    """Error message includes list of available agents."""
    with pytest.raises(ValueError) as exc_info:
        make_agent("bad_agent")

    error_msg = str(exc_info.value)
    for agent_id in list_agents():
        assert agent_id in error_msg


def test_registry_completeness():
    """All registered agents can be instantiated."""
    action_space = Box(low=-1.0, high=1.0, shape=(1,))
    discrete_space = Discrete(n=2)

    for agent_id in list_agents():
        # RL agents need discrete spaces
        if agent_id in ("dqn", "pqn"):
            agent = make_agent(agent_id, action_space=discrete_space)
        else:
            agent = make_agent(agent_id, action_space=action_space)

        assert isinstance(agent, Agent), f"{agent_id} did not return an Agent"
