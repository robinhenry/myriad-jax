"""Tests for the PQN agent implementation."""

import jax
import jax.numpy as jnp
import pytest

from myriad.agents.rl.pqn import AgentState, _compute_lambda_returns, make_agent
from myriad.core.spaces import Discrete
from myriad.core.types import Transition


@pytest.fixture
def action_space():
    """Create a discrete action space."""
    return Discrete(n=4)


@pytest.fixture
def agent(action_space):
    """Create a PQN agent with default parameters."""
    return make_agent(action_space)


@pytest.fixture
def key():
    """Create a random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_obs():
    """Create a sample observation."""
    return jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)


def test_make_agent_default(action_space):
    """Test creating a PQN agent with default parameters."""
    agent = make_agent(action_space)
    assert agent.params.action_space == action_space
    assert agent.params.learning_rate == 2.5e-4
    assert agent.params.gamma == 0.99
    assert agent.params.lambda_ == 0.65
    assert agent.params.hidden_size == 128
    assert agent.params.num_layers == 2


def test_make_agent_custom_params(action_space):
    """Test creating a PQN agent with custom parameters."""
    agent = make_agent(
        action_space,
        learning_rate=1e-3,
        gamma=0.95,
        lambda_=0.8,
        hidden_size=64,
        num_layers=3,
    )
    assert agent.params.learning_rate == 1e-3
    assert agent.params.gamma == 0.95
    assert agent.params.lambda_ == 0.8
    assert agent.params.hidden_size == 64
    assert agent.params.num_layers == 3


def test_init(key, sample_obs, agent):
    """Test agent initialization."""
    agent_state = agent.init(key, sample_obs, agent.params)

    assert isinstance(agent_state, AgentState)
    assert agent_state.train_state is not None
    assert agent_state.global_step == 0


def test_select_action(key, sample_obs, agent):
    """Test action selection."""
    agent_state = agent.init(key, sample_obs, agent.params)
    action, new_state = agent.select_action(key, sample_obs, agent_state, agent.params)

    assert action.shape == ()
    assert 0 <= action < agent.params.action_space.n
    # State should be unchanged (PQN doesn't update state during action selection)
    assert new_state.global_step == agent_state.global_step


def test_select_action_deterministic_greedy(key, sample_obs, agent):
    """Test that action selection is deterministic when epsilon=0."""
    # Create agent with epsilon_end = 0 and decay_steps = 0
    agent = make_agent(agent.params.action_space, epsilon_end=0.0, epsilon_decay_steps=0)
    agent_state = agent.init(key, sample_obs, agent.params)

    # Select action twice with same key
    action1, _ = agent.select_action(key, sample_obs, agent_state, agent.params)
    action2, _ = agent.select_action(key, sample_obs, agent_state, agent.params)

    assert action1 == action2


def test_compute_lambda_returns():
    """Test lambda-return computation."""
    # Simple test case
    rewards = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
    dones = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    next_q_max = jnp.array([0.5, 0.5, 0.0], dtype=jnp.float32)
    gamma = 0.99
    lambda_ = 0.5

    returns = _compute_lambda_returns(rewards, dones, next_q_max, gamma, lambda_)

    assert returns.shape == (3,)
    # Last return should be r + gamma * (1-done) * Q = 1.0 + 0.99 * 0 * 0.0 = 1.0
    assert jnp.isclose(returns[-1], 1.0)
    # Earlier returns should be higher (accumulating discounted future rewards)
    assert returns[0] > returns[1] > returns[2]


def test_update(key, sample_obs, agent):
    """Test agent update with a batch of transitions."""
    agent_state = agent.init(key, sample_obs, agent.params)

    # Create a batch of transitions (must be multiple of num_minibatches)
    batch_size = 16  # Must be divisible by num_minibatches (4)
    batch = Transition(
        obs=jnp.tile(sample_obs, (batch_size, 1)),
        action=jnp.zeros(batch_size, dtype=jnp.int32),
        reward=jnp.ones(batch_size, dtype=jnp.float32),
        next_obs=jnp.tile(sample_obs, (batch_size, 1)),
        done=jnp.zeros(batch_size, dtype=jnp.bool_),
    )

    # Update agent
    new_state, metrics = agent.update(key, agent_state, batch, agent.params)

    assert isinstance(new_state, AgentState)
    assert new_state.global_step == agent_state.global_step + 1
    assert "loss" in metrics
    assert "td_error" in metrics
    assert "q_value" in metrics
    assert "lambda_return_mean" in metrics


def test_epsilon_decay(agent):
    """Test epsilon decay over time."""
    # At step 0, epsilon should be close to epsilon_start
    epsilon_0 = max(
        agent.params.epsilon_end,
        agent.params.epsilon_start
        - (agent.params.epsilon_start - agent.params.epsilon_end) * 0 / agent.params.epsilon_decay_steps,
    )
    assert jnp.isclose(epsilon_0, agent.params.epsilon_start)

    # At final decay step, epsilon should be epsilon_end
    final_step = agent.params.epsilon_decay_steps
    epsilon_final = max(
        agent.params.epsilon_end,
        agent.params.epsilon_start
        - (agent.params.epsilon_start - agent.params.epsilon_end) * final_step / agent.params.epsilon_decay_steps,
    )
    assert jnp.isclose(epsilon_final, agent.params.epsilon_end)


def test_invalid_action_space():
    """Test that PQN raises error for non-Discrete action spaces."""
    from myriad.core.spaces import Box

    box_space = Box(low=-1.0, high=1.0, shape=(2,))

    with pytest.raises(ValueError, match="PQN only supports Discrete action spaces"):
        agent = make_agent(box_space)
        key = jax.random.PRNGKey(42)
        sample_obs = jnp.zeros(4, dtype=jnp.float32)
        agent.init(key, sample_obs, agent.params)


def test_jit_compilation(key, sample_obs, agent):
    """Test that all agent functions can be JIT compiled."""
    # JIT compile init
    jitted_init = jax.jit(agent.init, static_argnames=["params"])
    agent_state = jitted_init(key, sample_obs, params=agent.params)

    # JIT compile select_action
    jitted_select = jax.jit(agent.select_action, static_argnames=["params"])
    action, _ = jitted_select(key, sample_obs, agent_state, params=agent.params)
    assert action.shape == ()

    # JIT compile update
    batch = Transition(
        obs=jnp.tile(sample_obs, (16, 1)),
        action=jnp.zeros(16, dtype=jnp.int32),
        reward=jnp.ones(16, dtype=jnp.float32),
        next_obs=jnp.tile(sample_obs, (16, 1)),
        done=jnp.zeros(16, dtype=jnp.bool_),
    )
    jitted_update = jax.jit(agent.update, static_argnames=["params"])
    new_state, metrics = jitted_update(key, agent_state, batch, params=agent.params)
    assert new_state.global_step == agent_state.global_step + 1
