import chex
import jax
import jax.numpy as jnp
import pytest

from myriad.agents.classical.pid import AgentState, make_agent
from myriad.core.spaces import Box, Discrete
from tests.conftest import MockObs


@pytest.fixture
def box_action_space() -> Box:
    return Box(low=-2.0, high=3.0, shape=(1,))


def test_make_agent_box(box_action_space: Box):
    """Test agent creation and validation for Box action space."""
    agent = make_agent(
        action_space=box_action_space,
        kp=2.0,
        ki=0.5,
        kd=1.0,
        setpoint=1.5,
        obs_field="x",
        dt=0.01,
        anti_windup=5.0,
    )
    assert agent.params.kp == 2.0
    assert agent.params.ki == 0.5
    assert agent.params.kd == 1.0
    assert agent.params.setpoint == 1.5
    assert agent.params.obs_field == "x"
    assert agent.params.dt == 0.01
    assert agent.params.anti_windup == 5.0
    assert agent.params.control_low == -2.0
    assert agent.params.control_high == 3.0
    assert agent.params.bin_edges is None


def test_make_agent_discrete():
    agent = make_agent(
        action_space=Discrete(n=3),
        kp=2.0,
        ki=0.5,
        kd=1.0,
        setpoint=1.5,
        obs_field="x",
        dt=0.01,
        anti_windup=5.0,
        control_low=-3.0,
        control_high=6.0,
    )
    assert agent.params.control_low == -3.0
    assert agent.params.control_high == 6.0
    chex.assert_trees_all_close(agent.params.bin_edges, jnp.array([-3.0, 0.0, 3.0, 6.0]), atol=1e-6)


def test_make_agent_discrete_requires_bounds():
    """Test that Discrete action spaces require control bounds."""
    with pytest.raises(ValueError, match="control_low and control_high must be specified"):
        make_agent(action_space=Discrete(n=2))


def test_make_agent_invalid_obs_field(box_action_space: Box):
    """Test that invalid obs_field is rejected."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        make_agent(action_space=box_action_space, obs_field="")
    with pytest.raises(ValueError, match="must be a non-empty string"):
        make_agent(action_space=box_action_space, obs_field=123)  # type: ignore


def test_init(key, make_obs, box_action_space: Box):
    """Test agent initialization returns state with zero integral and error."""
    agent = make_agent(action_space=box_action_space, obs_field="theta")
    state = agent.init(key, make_obs(), agent.params)
    assert isinstance(state, AgentState)
    assert state.integral_error == 0.0
    assert state.previous_error == 0.0
    assert state.obs_index == 2


def test_select_action_p_controller(key, make_obs, box_action_space: Box):
    """Test P-only controller: error produces proportional action."""
    agent = make_agent(action_space=box_action_space, kp=2.0, ki=0.0, kd=0.0, setpoint=0.0, obs_field="theta")
    state = agent.init(key, make_obs(), agent.params)

    # At setpoint: zero action
    action_zero, _ = agent.select_action(key, make_obs(theta=0.0), state, agent.params, deterministic=True)
    chex.assert_trees_all_close(action_zero, jnp.array([0.0]))

    # Above setpoint: negative action (error = setpoint - obs = -0.5)
    action_neg, _ = agent.select_action(key, make_obs(theta=0.5), state, agent.params, deterministic=True)
    chex.assert_trees_all_close(action_neg, jnp.array([-1.0]), atol=1e-6)

    # Below setpoint: positive action (error = 0.5)
    action_pos, _ = agent.select_action(key, make_obs(theta=-0.5), state, agent.params, deterministic=True)
    chex.assert_trees_all_close(action_pos, jnp.array([1.0]), atol=1e-6)


def test_select_action_pi_controller(key, make_obs, box_action_space: Box):
    """Test PI controller: integral accumulates and anti-windup limits it."""
    agent = make_agent(
        action_space=box_action_space, kp=1.0, ki=0.1, kd=0.0, setpoint=0.0, obs_field="theta", anti_windup=10.0
    )
    state = agent.init(key, make_obs(), agent.params)

    # Run multiple steps with constant error - integral should accumulate
    obs = make_obs(theta=0.5)
    action1, state1 = agent.select_action(key, obs, state, agent.params, deterministic=True)
    assert state1.integral_error == -0.5
    assert action1 == -0.5 + 0.1 * (-0.5)

    action2, state2 = agent.select_action(key, obs, state1, agent.params, deterministic=True)
    assert state2.integral_error == -1.0
    assert action2 == -0.5 + 0.1 * (-1.0)

    # Run many steps with large error - integral should be clamped by anti-windup
    obs_large = make_obs(theta=10.0)
    state_windup = state
    for _ in range(1000):
        _, state_windup = agent.select_action(key, obs_large, state_windup, agent.params, deterministic=True)
    assert jnp.abs(state_windup.integral_error) == agent.params.anti_windup


def test_select_action_pid_controller(key, make_obs, box_action_space: Box):
    """Test full PID controller: derivative responds to error rate of change."""
    agent = make_agent(action_space=box_action_space, kp=1.0, ki=0.1, kd=0.5, setpoint=0.0, obs_field="theta", dt=0.2)
    state = agent.init(key, make_obs(), agent.params)

    # First step with small error
    _, state1 = agent.select_action(key, make_obs(theta=-0.1), state, agent.params, deterministic=True)
    assert state1.previous_error == 0.1

    # Second step with larger error - derivative term should contribute
    action2, state2 = agent.select_action(key, make_obs(theta=-0.5), state1, agent.params, deterministic=True)
    assert state2.previous_error == 0.5
    assert state2.integral_error != 0.0

    # Work out the exact expected action
    p_term = 0.5
    i_term = (0.1 * 0.2) + 0.5 * 0.2
    d_term = (0.5 - 0.1) / 0.2
    chex.assert_trees_all_close(action2, jnp.array([1.0 * p_term + 0.1 * i_term + 0.5 * d_term]))


def test_select_action_clamping(key, make_obs, box_action_space: Box):
    """Test that control output is clamped to action space bounds."""
    agent = make_agent(action_space=box_action_space, kp=100.0, setpoint=0.0, obs_field="theta")
    state = agent.init(key, make_obs(), agent.params)

    # Large error should saturate at bounds [-2, 2]
    action, _ = agent.select_action(key, make_obs(theta=10.0), state, agent.params, deterministic=True)
    chex.assert_trees_all_close(action, box_action_space.low)

    action, _ = agent.select_action(key, make_obs(theta=-10.0), state, agent.params, deterministic=True)
    chex.assert_trees_all_close(action, box_action_space.high)


def test_select_action_different_obs_fields(key, make_obs, box_action_space: Box):
    """Test obs_field parameter works with different fields."""
    agent_x = make_agent(action_space=box_action_space, kp=1.0, setpoint=0.0, obs_field="x")
    state = agent_x.init(key, make_obs(), agent_x.params)

    action, _ = agent_x.select_action(key, make_obs(x=0.5), state, agent_x.params, deterministic=True)
    chex.assert_trees_all_close(action, jnp.array([-0.5]), atol=1e-6)


@pytest.mark.parametrize(
    ("theta", "action"),
    [
        (-1.5, 1),  # control=1.5, middle bin [0, 3)
        (-4.5, 2),  # control=4.5, upper bin [3, 6]
        (1.5, 0),  # control=-1.5, lower bin [-3, 0)
        (-100.0, 2),  # control clamped to 6.0, upper bin
        (100.0, 0),  # control clamped to -3.0, lower bin
    ],
)
def test_select_action_discrete(key, make_obs, theta, action):
    """Test that continuous control is discretized into bins for Discrete action space.

    Discrete(n=3) with control_low=-3.0, control_high=6.0
    bin_edges: [-3.0, 0.0, 3.0, 6.0] (approximately, due to FP precision)
    action 0: control in [-3.0, ~0.0)
    action 1: control in [~0.0, 3.0)
    action 2: control in [3.0, 6.0]
    """
    agent = make_agent(
        action_space=Discrete(n=3),
        kp=1.0,
        ki=0.0,
        kd=0.0,
        setpoint=0.0,
        obs_field="theta",
        control_low=-3.0,
        control_high=6.0,
    )
    state = agent.init(key, make_obs(), agent.params)
    result, _ = agent.select_action(key, make_obs(theta=theta), state, agent.params, deterministic=True)
    assert result == action


def test_update(key, make_obs, box_action_space: Box):
    """Test update returns unchanged state and empty metrics."""
    agent = make_agent(action_space=box_action_space)
    state = agent.init(key, make_obs(), agent.params)
    updated_state, metrics = agent.update(key, state, None, agent.params)
    assert isinstance(updated_state, AgentState)
    assert metrics == {}


def test_jax_transforms(key, make_obs, box_action_space: Box):
    """End-to-end test with JIT and vmap."""
    agent = make_agent(action_space=box_action_space, kp=1.0, setpoint=0.0, obs_field="theta")

    # JIT compatibility
    jitted_init = jax.jit(agent.init)
    jitted_select = jax.jit(agent.select_action, static_argnames=("deterministic",))

    state = jitted_init(key, make_obs(), agent.params)
    assert isinstance(state, AgentState)

    action, _ = jitted_select(key, make_obs(theta=0.5), state, agent.params, False)
    chex.assert_trees_all_close(action, jnp.array([-0.5]), atol=1e-6)

    # Vmap compatibility
    batch_size = 4
    obs_batch = MockObs(
        x=jnp.zeros(batch_size),
        x_dot=jnp.zeros(batch_size),
        theta=jnp.array([-0.5, 0.0, 0.5, 1.0]),
        theta_dot=jnp.zeros(batch_size),
    )
    keys = jax.random.split(key, batch_size)
    vmapped_select = jax.vmap(agent.select_action, in_axes=(0, 0, None, None, None))
    actions, _ = vmapped_select(keys, obs_batch, state, agent.params, False)

    assert actions.shape == (batch_size, 1)
    chex.assert_trees_all_close(actions, jnp.array([[0.5], [0.0], [-0.5], [-1.0]]), atol=1e-6)
