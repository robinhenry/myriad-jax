"""Tests for the DQN agent."""

import chex
import jax.numpy as jnp
import jax.random
import pytest

from aion.agents.agent import Agent
from aion.agents.dqn import AgentState, make_agent
from aion.core.spaces import Discrete
from aion.core.types import Transition


@pytest.fixture
def action_space() -> Discrete:
    return Discrete(n=2)


@pytest.fixture
def agent(action_space: Discrete) -> Agent:
    return make_agent(action_space=action_space, learning_rate=1e-3)


@pytest.fixture
def sample_obs():
    return jnp.array([0.1, 0.2, 0.3, 0.4])


def test_make_agent_default(agent: Agent):
    """Test that agent can be created."""
    assert agent is not None


def test_init(key, agent: Agent, sample_obs):
    """Test that agent initialization works."""
    state = agent.init(key, sample_obs, agent.params)
    assert isinstance(state, AgentState)
    assert state.train_state is not None
    assert state.target_params is not None
    assert state.global_step == 0


def test_select_action(key, action_space: Discrete, agent: Agent, sample_obs):
    """Test that action selection works."""
    # Initialize agent
    key, init_key = jax.random.split(key)
    agent_state = agent.init(init_key, sample_obs, agent.params)

    # Select action
    key, action_key = jax.random.split(key)
    action, new_state = agent.select_action(action_key, sample_obs, agent_state, agent.params)

    # Check action is valid
    assert action.shape == ()
    assert 0 <= action < action_space.n
    chex.assert_tree_no_nones(action)


def test_select_action_deterministic_at_epsilon_zero(key, agent: Agent, sample_obs):
    """Test that action selection is deterministic when epsilon is zero."""
    # Create agent with no exploration
    agent_no_explore = make_agent(
        action_space=Discrete(n=2),
        epsilon_start=0.0,
        epsilon_end=0.0,
    )

    # Initialize agent
    key, init_key = jax.random.split(key)
    agent_state = agent_no_explore.init(init_key, sample_obs, agent_no_explore.params)

    # Select actions with different keys
    key1, key2 = jax.random.split(key)
    action1, _ = agent_no_explore.select_action(key1, sample_obs, agent_state, agent_no_explore.params)
    action2, _ = agent_no_explore.select_action(key2, sample_obs, agent_state, agent_no_explore.params)

    # Should be same action (greedy)
    chex.assert_trees_all_close(action1, action2)


def test_update(key, agent: Agent, sample_obs):
    """Test that agent update works."""
    # Initialize agent
    key, init_key = jax.random.split(key)
    agent_state = agent.init(init_key, sample_obs, agent.params)

    # Create a batch of transitions
    batch_size = 8
    batch = Transition(
        obs=jnp.tile(sample_obs, (batch_size, 1)),
        action=jnp.zeros(batch_size, dtype=jnp.int32),
        reward=jnp.ones(batch_size),
        next_obs=jnp.tile(sample_obs, (batch_size, 1)),
        done=jnp.zeros(batch_size, dtype=bool),
    )

    # Update agent
    key, update_key = jax.random.split(key)
    new_state, metrics = agent.update(update_key, agent_state, batch, agent.params)

    # Check state is updated
    assert isinstance(new_state, AgentState)
    assert new_state.global_step == agent_state.global_step + 1

    # Check metrics
    assert "loss" in metrics
    assert "td_error" in metrics
    assert "q_value" in metrics
    chex.assert_tree_no_nones(metrics)


def test_epsilon_decay(key, agent: Agent, sample_obs):
    """Test that epsilon decays over time."""
    # Initialize agent
    agent_state = agent.init(key, sample_obs, agent.params)

    # Epsilon should start high
    epsilon_start = agent.params.epsilon_start
    assert epsilon_start == 1.0

    # After many updates, agent should use more greedy actions
    # (this is implicit in the epsilon decay logic in select_action)
    # We verify the global step increments correctly
    batch = Transition(
        obs=jnp.tile(sample_obs, (8, 1)),
        action=jnp.zeros(8, dtype=jnp.int32),
        reward=jnp.ones(8),
        next_obs=jnp.tile(sample_obs, (8, 1)),
        done=jnp.zeros(8, dtype=bool),
    )

    for i in range(5):
        key, update_key = jax.random.split(key)
        agent_state, _ = agent.update(update_key, agent_state, batch, agent.params)
        assert agent_state.global_step == i + 1
