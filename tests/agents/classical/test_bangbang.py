import chex
import jax
import jax.numpy as jnp
import pytest

from myriad.agents.agent import Agent
from myriad.agents.classical.bangbang import AgentState, make_agent
from myriad.core.spaces import Box, Discrete
from myriad.envs.cartpole.physics import PhysicsState


# Fixtures for action spaces
@pytest.fixture
def discrete_action_space() -> Discrete:
    return Discrete(n=2)


@pytest.fixture
def box_action_space() -> Box:
    return Box(low=-2.0, high=2.0, shape=(1,))


@pytest.fixture
def vector_box_action_space() -> Box:
    return Box(low=-1.0, high=1.0, shape=(2,))


# Fixtures for agents
@pytest.fixture
def discrete_agent(discrete_action_space: Discrete) -> Agent:
    return make_agent(action_space=discrete_action_space, threshold=0.0, obs_field="theta")


@pytest.fixture
def box_agent(box_action_space: Box) -> Agent:
    return make_agent(action_space=box_action_space, threshold=0.0, obs_field="theta")


# Fixtures for observations
@pytest.fixture
def obs_below_threshold() -> PhysicsState:
    """Observation with theta = -0.1 (below threshold of 0.0)"""
    return PhysicsState(x=0.0, x_dot=0.0, theta=-0.1, theta_dot=0.0)


@pytest.fixture
def obs_above_threshold() -> PhysicsState:
    """Observation with theta = 0.1 (above threshold of 0.0)"""
    return PhysicsState(x=0.0, x_dot=0.0, theta=0.1, theta_dot=0.0)


@pytest.fixture
def obs_at_threshold() -> PhysicsState:
    """Observation with theta = 0.0 (exactly at threshold)"""
    return PhysicsState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)


# Test agent creation
def test_make_agent_discrete(discrete_agent: Agent):
    """Test creating agent with Discrete action space"""
    assert discrete_agent is not None
    assert isinstance(discrete_agent.params.action_space, Discrete)
    assert discrete_agent.params.threshold == 0.0
    assert discrete_agent.params.obs_field == "theta"


def test_make_agent_box(box_agent: Agent):
    """Test creating agent with Box action space"""
    assert box_agent is not None
    assert isinstance(box_agent.params.action_space, Box)
    assert box_agent.params.threshold == 0.0
    assert box_agent.params.obs_field == "theta"


def test_make_agent_custom_params():
    """Test creating agent with custom parameters"""
    agent = make_agent(
        action_space=Discrete(n=2),
        threshold=1.5,
        obs_field="x",
    )
    assert agent.params.threshold == 1.5
    assert agent.params.obs_field == "x"


# Test validation
def test_make_agent_invalid_space():
    """Test that invalid action spaces are rejected"""
    with pytest.raises(ValueError, match="only supports Box and Discrete"):
        make_agent(action_space="invalid")  # type: ignore


def test_make_agent_empty_obs_field():
    """Test that empty obs_field is rejected"""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        make_agent(action_space=Discrete(n=2), obs_field="")


def test_make_agent_invalid_obs_field():
    """Test that non-string obs_field is rejected"""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        make_agent(action_space=Discrete(n=2), obs_field=123)  # type: ignore


# Test initialization
def test_init(key, discrete_agent: Agent):
    """Test agent initialization returns empty state"""
    sample_obs = PhysicsState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)
    state = discrete_agent.init(key, sample_obs, discrete_agent.params)
    assert isinstance(state, AgentState)


# Test action selection - Discrete action space
def test_select_action_discrete_below_threshold(key, discrete_agent: Agent, obs_below_threshold: PhysicsState):
    """When obs[field] <= threshold, should select action 0"""
    agent_state = discrete_agent.init(key, obs_below_threshold, discrete_agent.params)
    action, _ = discrete_agent.select_action(key, obs_below_threshold, agent_state, discrete_agent.params)
    assert action == 0


def test_select_action_discrete_above_threshold(key, discrete_agent: Agent, obs_above_threshold: PhysicsState):
    """When obs[field] > threshold, should select action 1"""
    agent_state = discrete_agent.init(key, obs_above_threshold, discrete_agent.params)
    action, _ = discrete_agent.select_action(key, obs_above_threshold, agent_state, discrete_agent.params)
    assert action == 1


def test_select_action_discrete_at_threshold(key, discrete_agent: Agent, obs_at_threshold: PhysicsState):
    """When obs[field] == threshold, should select action 0 (low)"""
    agent_state = discrete_agent.init(key, obs_at_threshold, discrete_agent.params)
    action, _ = discrete_agent.select_action(key, obs_at_threshold, agent_state, discrete_agent.params)
    assert action == 0


# Test action selection - Box action space
def test_select_action_box_below_threshold(key, box_agent: Agent, obs_below_threshold: PhysicsState):
    """When obs[field] <= threshold, should select low action"""
    agent_state = box_agent.init(key, obs_below_threshold, box_agent.params)
    action, _ = box_agent.select_action(key, obs_below_threshold, agent_state, box_agent.params)
    expected = jnp.broadcast_to(box_agent.params.action_space.low, box_agent.params.action_space.shape)
    chex.assert_trees_all_close(action, expected)


def test_select_action_box_above_threshold(key, box_agent: Agent, obs_above_threshold: PhysicsState):
    """When obs[field] > threshold, should select high action"""
    agent_state = box_agent.init(key, obs_above_threshold, box_agent.params)
    action, _ = box_agent.select_action(key, obs_above_threshold, agent_state, box_agent.params)
    expected = jnp.broadcast_to(box_agent.params.action_space.high, box_agent.params.action_space.shape)
    chex.assert_trees_all_close(action, expected)


def test_select_action_box_at_threshold(key, box_agent: Agent, obs_at_threshold: PhysicsState):
    """When obs[field] == threshold, should select low action"""
    agent_state = box_agent.init(key, obs_at_threshold, box_agent.params)
    action, _ = box_agent.select_action(key, obs_at_threshold, agent_state, box_agent.params)
    expected = jnp.broadcast_to(box_agent.params.action_space.low, box_agent.params.action_space.shape)
    chex.assert_trees_all_close(action, expected)


# Test different observation fields
def test_select_action_different_obs_fields(key, discrete_action_space: Discrete):
    """Test obs_field parameter works with different fields"""
    # Agent using "x" field
    agent_x = make_agent(action_space=discrete_action_space, threshold=0.0, obs_field="x")
    obs_x_positive = PhysicsState(x=0.5, x_dot=0.0, theta=0.0, theta_dot=0.0)
    state_x = agent_x.init(key, obs_x_positive, agent_x.params)
    action_x, _ = agent_x.select_action(key, obs_x_positive, state_x, agent_x.params)
    assert action_x == 1  # x > 0, should be high

    # Agent using "x_dot" field
    agent_x_dot = make_agent(action_space=discrete_action_space, threshold=0.0, obs_field="x_dot")
    obs_x_dot_negative = PhysicsState(x=0.0, x_dot=-0.5, theta=0.0, theta_dot=0.0)
    state_x_dot = agent_x_dot.init(key, obs_x_dot_negative, agent_x_dot.params)
    action_x_dot, _ = agent_x_dot.select_action(key, obs_x_dot_negative, state_x_dot, agent_x_dot.params)
    assert action_x_dot == 0  # x_dot < 0, should be low

    # Agent using "theta_dot" field
    agent_theta_dot = make_agent(action_space=discrete_action_space, threshold=0.0, obs_field="theta_dot")
    obs_theta_dot_positive = PhysicsState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.5)
    state_theta_dot = agent_theta_dot.init(key, obs_theta_dot_positive, agent_theta_dot.params)
    action_theta_dot, _ = agent_theta_dot.select_action(
        key, obs_theta_dot_positive, state_theta_dot, agent_theta_dot.params
    )
    assert action_theta_dot == 1  # theta_dot > 0, should be high


# Test determinism
def test_select_action_deterministic(key, discrete_agent: Agent, obs_above_threshold: PhysicsState):
    """Same inputs should yield same outputs (deterministic policy)"""
    agent_state = discrete_agent.init(key, obs_above_threshold, discrete_agent.params)
    action1, _ = discrete_agent.select_action(key, obs_above_threshold, agent_state, discrete_agent.params)
    action2, _ = discrete_agent.select_action(key, obs_above_threshold, agent_state, discrete_agent.params)
    chex.assert_trees_all_close(action1, action2)


def test_select_action_deterministic_flag_ignored(key, discrete_agent: Agent, obs_above_threshold: PhysicsState):
    """The deterministic flag should be ignored (always deterministic)"""
    agent_state = discrete_agent.init(key, obs_above_threshold, discrete_agent.params)
    action1, _ = discrete_agent.select_action(
        key, obs_above_threshold, agent_state, discrete_agent.params, deterministic=True
    )
    action2, _ = discrete_agent.select_action(
        key, obs_above_threshold, agent_state, discrete_agent.params, deterministic=False
    )
    chex.assert_trees_all_close(action1, action2)


# Test vector action spaces
def test_select_action_vector_action_space(key, vector_box_action_space: Box, obs_above_threshold: PhysicsState):
    """Test Box with shape=(2,) works correctly"""
    agent = make_agent(action_space=vector_box_action_space, threshold=0.0, obs_field="theta")
    agent_state = agent.init(key, obs_above_threshold, agent.params)
    action, _ = agent.select_action(key, obs_above_threshold, agent_state, agent.params)

    # Should return high action (theta > 0)
    expected = jnp.broadcast_to(vector_box_action_space.high, vector_box_action_space.shape)
    chex.assert_trees_all_close(action, expected)
    assert action.shape == (2,)


# Test update
def test_update(key, discrete_agent: Agent, obs_above_threshold: PhysicsState):
    """Test update returns unchanged state and empty metrics"""
    agent_state = discrete_agent.init(key, obs_above_threshold, discrete_agent.params)
    updated_state, metrics = discrete_agent.update(key, agent_state, None, discrete_agent.params)
    assert isinstance(updated_state, AgentState)
    assert metrics == {}


# Test JAX JIT compatibility
def test_jit_compatibility_discrete(key, discrete_agent: Agent, obs_above_threshold: PhysicsState):
    """Verify agent works with JAX JIT compilation (Discrete)"""
    agent_state = discrete_agent.init(key, obs_above_threshold, discrete_agent.params)
    jitted_select = jax.jit(discrete_agent.select_action, static_argnames=("deterministic",))
    action, state = jitted_select(key, obs_above_threshold, agent_state, discrete_agent.params, False)
    assert action == 1  # Should select high action
    assert isinstance(state, AgentState)


def test_jit_compatibility_box(key, box_agent: Agent, obs_below_threshold: PhysicsState):
    """Verify agent works with JAX JIT compilation (Box)"""
    agent_state = box_agent.init(key, obs_below_threshold, box_agent.params)
    jitted_select = jax.jit(box_agent.select_action, static_argnames=("deterministic",))
    action, state = jitted_select(key, obs_below_threshold, agent_state, box_agent.params, False)
    expected = jnp.broadcast_to(box_agent.params.action_space.low, box_agent.params.action_space.shape)
    chex.assert_trees_all_close(action, expected)  # Should select low action
    assert isinstance(state, AgentState)


def test_jit_compatibility_init(key, discrete_agent: Agent):
    """Verify agent init works with JAX JIT compilation"""
    jitted_init = jax.jit(discrete_agent.init)
    sample_obs = PhysicsState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)
    state = jitted_init(key, sample_obs, discrete_agent.params)
    assert isinstance(state, AgentState)


def test_jit_compatibility_update(key, discrete_agent: Agent, obs_above_threshold: PhysicsState):
    """Verify agent update works with JAX JIT compilation"""
    agent_state = discrete_agent.init(key, obs_above_threshold, discrete_agent.params)
    jitted_update = jax.jit(discrete_agent.update)
    updated_state, metrics = jitted_update(key, agent_state, None, discrete_agent.params)
    assert isinstance(updated_state, AgentState)
    assert metrics == {}


# Test vmap compatibility (for vectorized environments)
def test_vmap_compatibility_discrete(key, discrete_agent: Agent):
    """Verify agent works with vmap (batched observations)"""
    # Create batch of observations
    batch_size = 4
    obs_batch = PhysicsState(
        x=jnp.array([0.0, 0.0, 0.0, 0.0]),
        x_dot=jnp.array([0.0, 0.0, 0.0, 0.0]),
        theta=jnp.array([-0.2, -0.1, 0.1, 0.2]),  # Two below, two above threshold
        theta_dot=jnp.array([0.0, 0.0, 0.0, 0.0]),
    )

    # Initialize agent state (uses first observation)
    sample_obs = PhysicsState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)
    agent_state = discrete_agent.init(key, sample_obs, discrete_agent.params)

    # Generate batch of keys
    keys = jax.random.split(key, batch_size)

    # Vmap select_action over the batch
    vmapped_select = jax.vmap(discrete_agent.select_action, in_axes=(0, 0, None, None, None))  # vmap over keys and obs
    actions, states = vmapped_select(keys, obs_batch, agent_state, discrete_agent.params, False)

    # Check results
    assert actions.shape == (batch_size,)
    assert jnp.array_equal(actions, jnp.array([0, 0, 1, 1]))  # First two low, last two high


def test_vmap_compatibility_box(key, box_agent: Agent):
    """Verify agent works with vmap (batched observations) for Box action space"""
    # Create batch of observations
    batch_size = 3
    obs_batch = PhysicsState(
        x=jnp.array([0.0, 0.0, 0.0]),
        x_dot=jnp.array([0.0, 0.0, 0.0]),
        theta=jnp.array([-0.1, 0.0, 0.1]),  # Below, at, above threshold
        theta_dot=jnp.array([0.0, 0.0, 0.0]),
    )

    # Initialize agent state
    sample_obs = PhysicsState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)
    agent_state = box_agent.init(key, sample_obs, box_agent.params)

    # Generate batch of keys
    keys = jax.random.split(key, batch_size)

    # Vmap select_action over the batch
    vmapped_select = jax.vmap(box_agent.select_action, in_axes=(0, 0, None, None, None))
    actions, states = vmapped_select(keys, obs_batch, agent_state, box_agent.params, False)

    # Check results
    assert actions.shape == (batch_size, 1)
    expected_low = jnp.broadcast_to(box_agent.params.action_space.low, box_agent.params.action_space.shape)
    expected_high = jnp.broadcast_to(box_agent.params.action_space.high, box_agent.params.action_space.shape)

    # First two should be low, last one should be high
    chex.assert_trees_all_close(actions[0], expected_low)
    chex.assert_trees_all_close(actions[1], expected_low)
    chex.assert_trees_all_close(actions[2], expected_high)


# Test invert parameter
def test_make_agent_invert_default(discrete_action_space: Discrete):
    """Test that invert defaults to False"""
    agent = make_agent(action_space=discrete_action_space, threshold=0.0, obs_field="theta")
    assert agent.params.invert is False


def test_make_agent_invert_true(discrete_action_space: Discrete):
    """Test creating agent with invert=True"""
    agent = make_agent(action_space=discrete_action_space, threshold=0.0, obs_field="theta", invert=True)
    assert agent.params.invert is True


def test_select_action_inverted_discrete_below_threshold(
    key, discrete_action_space: Discrete, obs_below_threshold: PhysicsState
):
    """When inverted and obs[field] <= threshold, should select HIGH action (inverted)"""
    agent = make_agent(action_space=discrete_action_space, threshold=0.0, obs_field="theta", invert=True)
    agent_state = agent.init(key, obs_below_threshold, agent.params)
    action, _ = agent.select_action(key, obs_below_threshold, agent_state, agent.params)
    assert action == 1  # High action when inverted and below threshold


def test_select_action_inverted_discrete_above_threshold(
    key, discrete_action_space: Discrete, obs_above_threshold: PhysicsState
):
    """When inverted and obs[field] > threshold, should select LOW action (inverted)"""
    agent = make_agent(action_space=discrete_action_space, threshold=0.0, obs_field="theta", invert=True)
    agent_state = agent.init(key, obs_above_threshold, agent.params)
    action, _ = agent.select_action(key, obs_above_threshold, agent_state, agent.params)
    assert action == 0  # Low action when inverted and above threshold


def test_select_action_inverted_box_below_threshold(key, box_action_space: Box, obs_below_threshold: PhysicsState):
    """When inverted and obs[field] <= threshold, should select HIGH action (inverted)"""
    agent = make_agent(action_space=box_action_space, threshold=0.0, obs_field="theta", invert=True)
    agent_state = agent.init(key, obs_below_threshold, agent.params)
    action, _ = agent.select_action(key, obs_below_threshold, agent_state, agent.params)
    expected = jnp.broadcast_to(box_action_space.high, box_action_space.shape)
    chex.assert_trees_all_close(action, expected)


def test_select_action_inverted_box_above_threshold(key, box_action_space: Box, obs_above_threshold: PhysicsState):
    """When inverted and obs[field] > threshold, should select LOW action (inverted)"""
    agent = make_agent(action_space=box_action_space, threshold=0.0, obs_field="theta", invert=True)
    agent_state = agent.init(key, obs_above_threshold, agent.params)
    action, _ = agent.select_action(key, obs_above_threshold, agent_state, agent.params)
    expected = jnp.broadcast_to(box_action_space.low, box_action_space.shape)
    chex.assert_trees_all_close(action, expected)


def test_select_action_normal_vs_inverted(key, discrete_action_space: Discrete, obs_above_threshold: PhysicsState):
    """Normal and inverted agents should select opposite actions"""
    # Normal agent
    agent_normal = make_agent(action_space=discrete_action_space, threshold=0.0, obs_field="theta", invert=False)
    state_normal = agent_normal.init(key, obs_above_threshold, agent_normal.params)
    action_normal, _ = agent_normal.select_action(key, obs_above_threshold, state_normal, agent_normal.params)

    # Inverted agent
    agent_inverted = make_agent(action_space=discrete_action_space, threshold=0.0, obs_field="theta", invert=True)
    state_inverted = agent_inverted.init(key, obs_above_threshold, agent_inverted.params)
    action_inverted, _ = agent_inverted.select_action(key, obs_above_threshold, state_inverted, agent_inverted.params)

    # Actions should be opposite
    assert action_normal == 1  # High when above threshold (normal)
    assert action_inverted == 0  # Low when above threshold (inverted)
