import chex
import jax.numpy as jnp
import jax.random
import pytest

from myriad.agents.agent import Agent
from myriad.agents.classical.random import AgentState, make_agent
from myriad.core.spaces import Box


@pytest.fixture
def action_space() -> Box:
    return Box(low=-3.0, high=2.0, shape=(2,))


@pytest.fixture
def agent(action_space: Box) -> Agent:
    return make_agent(action_space=action_space)


def test_init(key, agent: Agent):
    state = agent.init(key, jnp.array([5.0]), agent.params)
    assert isinstance(state, AgentState)


def test_select_action(key, action_space: Box, agent: Agent):
    obs = jnp.array([0.0])

    a1, _ = agent.select_action(key, obs, None, agent.params, deterministic=False)
    assert a1.shape == action_space.shape
    chex.assert_tree_no_nones(a1)

    # Check that selecting actions with the same params and keys yield
    # the same actions
    a2, _ = agent.select_action(key, obs, None, agent.params, deterministic=False)
    chex.assert_trees_all_close(a1, a2)


def test_select_action_different_keys(key, action_space: Box, agent: Agent):
    """Check that selecting actions with different keys yield different actions"""

    obs = jnp.array([0.0])
    key1, key2 = jax.random.split(key)

    a1, _ = agent.select_action(key1, obs, None, agent.params, deterministic=False)
    a2, _ = agent.select_action(key2, obs, None, agent.params, deterministic=False)

    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(a1, a2)


def test_update(key, agent: Agent):
    agent_state, _ = agent.update(key, None, None, agent.params)
    assert agent_state is None
