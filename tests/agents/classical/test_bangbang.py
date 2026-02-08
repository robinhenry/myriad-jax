import chex
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from myriad.agents.classical.bangbang import AgentState, make_agent
from myriad.core.spaces import Box, Discrete
from tests.conftest import MockObs


def test_make_agent_discrete():
    """Test agent creation and validation for Discrete action space."""
    agent = make_agent(action_space=Discrete(n=3), threshold=1.5, obs_field="x")
    assert agent.params.action_space.n == 3  # type: ignore
    assert agent.params.threshold == 1.5
    assert agent.params.obs_field == "x"
    assert agent.params.invert is False
    assert agent.params.low_action == jnp.array(0)
    assert agent.params.high_action == jnp.array(2)


def test_make_agent_box():
    """Test agent creation and validation for Box action space."""
    agent = make_agent(action_space=Box(low=-3.2, high=5.1, shape=(2,)), invert=True)
    assert agent.params.invert is True
    npt.assert_array_equal(agent.params.low_action, jnp.array([-3.2, -3.2]))
    npt.assert_array_equal(agent.params.high_action, jnp.array([5.1, 5.1]))


def test_make_agent_invalid():
    """Test that make_agent raises an error for invalid action spaces."""
    with pytest.raises(ValueError, match="only supports Box and Discrete"):
        make_agent(action_space="invalid")  # type: ignore


def test_make_agent_invalid_obs_field():
    """Test that make_agent raises an error for invalid observation fields."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        make_agent(action_space=Discrete(n=2), obs_field="")


def test_init(key, make_obs):
    """Test agent initialization returns empty state."""
    agent = make_agent(action_space=Discrete(n=2), obs_field="x_dot")
    state = agent.init(key, make_obs(), agent.params)
    assert isinstance(state, AgentState)
    assert state.obs_index == 1


def test_select_action_discrete(key, make_obs):
    """Test discrete action selection: low obs -> action 0, high obs -> action n-1."""
    obs_below = make_obs(theta=-0.1)
    obs_above = make_obs(theta=0.1)

    agent = make_agent(action_space=Discrete(n=3), threshold=0.0, obs_field="theta")
    state = agent.init(key, obs_below, agent.params)
    action_low, _ = agent.select_action(key, obs_below, state, agent.params, deterministic=True)
    action_high, _ = agent.select_action(key, obs_above, state, agent.params, deterministic=True)
    assert action_low == 0
    assert action_high == 2


def test_select_action_discrete_inverted(key, make_obs):
    """Test inverted discrete action: low obs -> action n-1, high obs -> action 0."""
    obs_below = make_obs(theta=-0.1)
    obs_above = make_obs(theta=0.1)

    agent = make_agent(action_space=Discrete(n=4), threshold=0.0, obs_field="theta", invert=True)
    state = agent.init(key, obs_below, agent.params)
    action_low, _ = agent.select_action(key, obs_below, state, agent.params, deterministic=True)
    action_high, _ = agent.select_action(key, obs_above, state, agent.params, deterministic=True)
    assert action_low == 3
    assert action_high == 0


def test_select_action_box(key, make_obs):
    """Test box action selection uses space bounds."""
    obs_below = make_obs(theta=-0.1)
    obs_above = make_obs(theta=0.1)

    agent = make_agent(action_space=Box(low=-2.0, high=3.0, shape=(1,)), threshold=0.0, obs_field="theta")
    state = agent.init(key, obs_below, agent.params)
    action_low, _ = agent.select_action(key, obs_below, state, agent.params, deterministic=True)
    action_high, _ = agent.select_action(key, obs_above, state, agent.params, deterministic=True)
    chex.assert_trees_all_close(action_low, jnp.array([-2.0]))
    chex.assert_trees_all_close(action_high, jnp.array([3.0]))


def test_update(key, make_obs):
    """Test update returns unchanged state and empty metrics."""
    agent = make_agent(action_space=Discrete(n=2))
    state = agent.init(key, make_obs(), agent.params)
    updated_state, metrics = agent.update(key, state, None, agent.params)
    assert isinstance(updated_state, AgentState)
    assert metrics == {}


def test_jax_transforms(key, make_obs):
    """End-to-end test with JIT and vmap."""
    agent = make_agent(action_space=Discrete(n=2), threshold=0.0, obs_field="theta")

    # JIT compatibility
    jitted_init = jax.jit(agent.init)
    jitted_select = jax.jit(agent.select_action, static_argnames=("deterministic",))

    state = jitted_init(key, make_obs(), agent.params)
    assert isinstance(state, AgentState)

    action, _ = jitted_select(key, make_obs(theta=0.1), state, agent.params, False)
    assert action == 1

    # Vmap compatibility
    batch_size = 4
    obs_batch = MockObs(
        x=jnp.zeros(batch_size),
        x_dot=jnp.zeros(batch_size),
        theta=jnp.array([-0.2, -0.1, 0.1, 0.2]),
        theta_dot=jnp.zeros(batch_size),
    )
    keys = jax.random.split(key, batch_size)
    vmapped_select = jax.vmap(agent.select_action, in_axes=(0, 0, None, None, None))
    actions, _ = vmapped_select(keys, obs_batch, state, agent.params, False)

    assert actions.shape == (batch_size,)
    chex.assert_trees_all_close(actions, jnp.array([0, 0, 1, 1]))
