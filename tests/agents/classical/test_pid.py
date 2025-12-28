import chex
import jax
import jax.numpy as jnp
import pytest

from myriad.agents.agent import Agent
from myriad.agents.classical.pid import AgentState, make_agent
from myriad.core.spaces import Box, Discrete
from myriad.envs.cartpole.physics import PhysicsState


# Fixtures for action spaces
@pytest.fixture
def box_action_space() -> Box:
    return Box(low=-2.0, high=2.0, shape=(1,))


@pytest.fixture
def vector_box_action_space() -> Box:
    return Box(low=-1.0, high=1.0, shape=(2,))


# Fixtures for agents
@pytest.fixture
def p_controller(box_action_space: Box) -> Agent:
    """Pure proportional controller (P-only)"""
    return make_agent(action_space=box_action_space, kp=1.0, ki=0.0, kd=0.0, setpoint=0.0, obs_field="theta")


@pytest.fixture
def pi_controller(box_action_space: Box) -> Agent:
    """Proportional-Integral controller (PI)"""
    return make_agent(action_space=box_action_space, kp=1.0, ki=0.1, kd=0.0, setpoint=0.0, obs_field="theta")


@pytest.fixture
def pid_controller(box_action_space: Box) -> Agent:
    """Full PID controller"""
    return make_agent(action_space=box_action_space, kp=1.0, ki=0.1, kd=0.5, setpoint=0.0, obs_field="theta")


# Fixtures for observations
@pytest.fixture
def obs_at_setpoint() -> PhysicsState:
    """Observation at setpoint (zero error)"""
    return PhysicsState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)


@pytest.fixture
def obs_above_setpoint() -> PhysicsState:
    """Observation above setpoint (positive error)"""
    return PhysicsState(x=0.0, x_dot=0.0, theta=0.5, theta_dot=0.0)


@pytest.fixture
def obs_below_setpoint() -> PhysicsState:
    """Observation below setpoint (negative error)"""
    return PhysicsState(x=0.0, x_dot=0.0, theta=-0.5, theta_dot=0.0)


# Test agent creation
def test_make_agent_default(box_action_space: Box):
    """Test creating agent with default parameters"""
    agent = make_agent(action_space=box_action_space)
    assert agent is not None
    assert isinstance(agent.params.action_space, Box)
    assert agent.params.kp == 1.0
    assert agent.params.ki == 0.0
    assert agent.params.kd == 0.0
    assert agent.params.setpoint == 0.0
    assert agent.params.obs_field == "theta"
    assert agent.params.dt == 0.02
    assert agent.params.anti_windup == 10.0


def test_make_agent_custom_params(box_action_space: Box):
    """Test creating agent with custom parameters"""
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


# Test validation
def test_make_agent_invalid_space():
    """Test that Discrete action spaces require control bounds"""
    with pytest.raises(ValueError, match="control_low and control_high must be specified"):
        make_agent(action_space=Discrete(n=2))


def test_make_agent_empty_obs_field(box_action_space: Box):
    """Test that empty obs_field is rejected"""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        make_agent(action_space=box_action_space, obs_field="")


def test_make_agent_invalid_obs_field(box_action_space: Box):
    """Test that non-string obs_field is rejected"""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        make_agent(action_space=box_action_space, obs_field=123)  # type: ignore


# Test initialization
def test_init(key, p_controller: Agent):
    """Test agent initialization returns state with zero integral and error"""
    sample_obs = PhysicsState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)
    state = p_controller.init(key, sample_obs, p_controller.params)
    assert isinstance(state, AgentState)
    assert state.integral_error == 0.0
    assert state.previous_error == 0.0
    assert state.obs_index >= 0


# Test P-only controller (proportional term)
def test_p_controller_at_setpoint(key, p_controller: Agent, obs_at_setpoint: PhysicsState):
    """P controller at setpoint should output zero action"""
    agent_state = p_controller.init(key, obs_at_setpoint, p_controller.params)
    action, new_state = p_controller.select_action(key, obs_at_setpoint, agent_state, p_controller.params)

    # Zero error -> zero action
    chex.assert_trees_all_close(action, jnp.array([0.0]))
    # State should remain zero
    assert new_state.integral_error == 0.0
    assert new_state.previous_error == 0.0


def test_p_controller_above_setpoint(key, p_controller: Agent, obs_above_setpoint: PhysicsState):
    """P controller above setpoint should output negative action"""
    agent_state = p_controller.init(key, obs_above_setpoint, p_controller.params)
    action, new_state = p_controller.select_action(key, obs_above_setpoint, agent_state, p_controller.params)

    # Positive observation (theta=0.5) with setpoint=0 -> error = -0.5
    # Control = kp * error = 1.0 * (-0.5) = -0.5
    expected_action = jnp.array([-0.5])
    chex.assert_trees_all_close(action, expected_action, atol=1e-6)

    # State should track error
    assert new_state.previous_error == -0.5


def test_p_controller_below_setpoint(key, p_controller: Agent, obs_below_setpoint: PhysicsState):
    """P controller below setpoint should output positive action"""
    agent_state = p_controller.init(key, obs_below_setpoint, p_controller.params)
    action, new_state = p_controller.select_action(key, obs_below_setpoint, agent_state, p_controller.params)

    # Negative observation (theta=-0.5) with setpoint=0 -> error = 0.5
    # Control = kp * error = 1.0 * 0.5 = 0.5
    expected_action = jnp.array([0.5])
    chex.assert_trees_all_close(action, expected_action, atol=1e-6)


# Test PI controller (integral term)
def test_pi_controller_accumulates_integral(key, pi_controller: Agent, obs_above_setpoint: PhysicsState):
    """PI controller should accumulate integral error over multiple steps"""
    agent_state = pi_controller.init(key, obs_above_setpoint, pi_controller.params)

    # First step
    action1, state1 = pi_controller.select_action(key, obs_above_setpoint, agent_state, pi_controller.params)

    # Second step with same error
    action2, state2 = pi_controller.select_action(key, obs_above_setpoint, state1, pi_controller.params)

    # Integral should accumulate in magnitude (error is negative, so integral becomes more negative)
    assert jnp.abs(state2.integral_error) > jnp.abs(state1.integral_error)

    # Action should increase in magnitude due to integral term
    assert jnp.abs(action2[0]) > jnp.abs(action1[0])


def test_pi_controller_integral_reset_on_sign_change(key, pi_controller: Agent):
    """Test integral accumulation with changing error"""
    agent_state = pi_controller.init(key, PhysicsState(0, 0, 0, 0), pi_controller.params)

    # Step with positive error
    obs_pos = PhysicsState(x=0.0, x_dot=0.0, theta=-0.5, theta_dot=0.0)  # error = +0.5
    _, state1 = pi_controller.select_action(key, obs_pos, agent_state, pi_controller.params)
    integral_after_pos = state1.integral_error

    # Step with negative error
    obs_neg = PhysicsState(x=0.0, x_dot=0.0, theta=0.5, theta_dot=0.0)  # error = -0.5
    _, state2 = pi_controller.select_action(key, obs_neg, state1, pi_controller.params)

    # Integral should decrease (opposite sign error)
    assert state2.integral_error < integral_after_pos


# Test PID controller (derivative term)
def test_pid_controller_derivative_term(key, pid_controller: Agent):
    """Test that derivative term responds to rate of change of error"""
    agent_state = pid_controller.init(key, PhysicsState(0, 0, 0, 0), pid_controller.params)

    # First step: error = 0.1
    obs1 = PhysicsState(x=0.0, x_dot=0.0, theta=-0.1, theta_dot=0.0)
    action1, state1 = pid_controller.select_action(key, obs1, agent_state, pid_controller.params)

    # Second step: error increases to 0.5 (rapid change)
    obs2 = PhysicsState(x=0.0, x_dot=0.0, theta=-0.5, theta_dot=0.0)
    action2, state2 = pid_controller.select_action(key, obs2, state1, pid_controller.params)

    # Derivative term should contribute to larger action
    # (error increased, so derivative is positive, adding to control)
    assert state2.previous_error == 0.5


def test_pid_controller_all_terms_contribute(key, pid_controller: Agent, obs_above_setpoint: PhysicsState):
    """Test that P, I, and D terms all contribute to control output"""
    agent_state = pid_controller.init(key, obs_above_setpoint, pid_controller.params)

    # Run for a few steps
    state = agent_state
    for _ in range(3):
        _, state = pid_controller.select_action(key, obs_above_setpoint, state, pid_controller.params)

    # After multiple steps with constant error:
    # - Integral should have accumulated
    # - Previous error should be set
    assert state.integral_error != 0.0
    assert state.previous_error != 0.0


# Test action clamping
def test_action_clamping_to_bounds(key, box_action_space: Box):
    """Test that control output is clamped to action space bounds"""
    # Create controller with very high gain to force saturation
    agent = make_agent(action_space=box_action_space, kp=100.0, setpoint=0.0, obs_field="theta")
    agent_state = agent.init(key, PhysicsState(0, 0, 0, 0), agent.params)

    # Large error should saturate
    obs_large_error = PhysicsState(x=0.0, x_dot=0.0, theta=10.0, theta_dot=0.0)
    action, _ = agent.select_action(key, obs_large_error, agent_state, agent.params)

    # Action should be clamped to bounds [-2, 2]
    assert jnp.all(action >= box_action_space.low)
    assert jnp.all(action <= box_action_space.high)


# Test anti-windup
def test_anti_windup_limits_integral(key, pi_controller: Agent):
    """Test that anti-windup prevents integral from growing unbounded"""
    agent_state = pi_controller.init(key, PhysicsState(0, 0, 0, 0), pi_controller.params)

    # Run many steps with constant large error to accumulate integral
    obs_large_error = PhysicsState(x=0.0, x_dot=0.0, theta=10.0, theta_dot=0.0)
    state = agent_state
    for _ in range(1000):
        _, state = pi_controller.select_action(key, obs_large_error, state, pi_controller.params)

    # Integral should be clamped to anti_windup limit
    assert jnp.abs(state.integral_error) <= pi_controller.params.anti_windup


# Test different observation fields
def test_select_action_different_obs_fields(key, box_action_space: Box):
    """Test obs_field parameter works with different fields"""
    # Agent controlling "x" field
    agent_x = make_agent(action_space=box_action_space, kp=1.0, setpoint=0.0, obs_field="x")
    obs = PhysicsState(x=0.5, x_dot=0.0, theta=0.0, theta_dot=0.0)
    state_x = agent_x.init(key, obs, agent_x.params)
    action_x, _ = agent_x.select_action(key, obs, state_x, agent_x.params)

    # error = 0 - 0.5 = -0.5, control = 1.0 * (-0.5) = -0.5
    chex.assert_trees_all_close(action_x, jnp.array([-0.5]), atol=1e-6)

    # Agent controlling "x_dot" field
    agent_x_dot = make_agent(action_space=box_action_space, kp=1.0, setpoint=0.0, obs_field="x_dot")
    obs2 = PhysicsState(x=0.0, x_dot=0.3, theta=0.0, theta_dot=0.0)
    state_x_dot = agent_x_dot.init(key, obs2, agent_x_dot.params)
    action_x_dot, _ = agent_x_dot.select_action(key, obs2, state_x_dot, agent_x_dot.params)

    # error = 0 - 0.3 = -0.3
    chex.assert_trees_all_close(action_x_dot, jnp.array([-0.3]), atol=1e-6)


# Test non-zero setpoint
def test_non_zero_setpoint(key, box_action_space: Box):
    """Test PID controller with non-zero setpoint"""
    agent = make_agent(action_space=box_action_space, kp=1.0, setpoint=1.0, obs_field="theta")
    agent_state = agent.init(key, PhysicsState(0, 0, 0, 0), agent.params)

    # Observation at 0.5, setpoint at 1.0 -> error = 0.5
    obs = PhysicsState(x=0.0, x_dot=0.0, theta=0.5, theta_dot=0.0)
    action, _ = agent.select_action(key, obs, agent_state, agent.params)

    # Control = kp * error = 1.0 * 0.5 = 0.5
    chex.assert_trees_all_close(action, jnp.array([0.5]), atol=1e-6)


# Test determinism
def test_select_action_deterministic(key, pid_controller: Agent, obs_above_setpoint: PhysicsState):
    """Same inputs should yield same outputs (deterministic policy)"""
    agent_state = pid_controller.init(key, obs_above_setpoint, pid_controller.params)
    action1, state1 = pid_controller.select_action(key, obs_above_setpoint, agent_state, pid_controller.params)
    action2, state2 = pid_controller.select_action(key, obs_above_setpoint, agent_state, pid_controller.params)

    chex.assert_trees_all_close(action1, action2)
    chex.assert_trees_all_close(state1, state2)


def test_select_action_deterministic_flag_ignored(key, pid_controller: Agent, obs_above_setpoint: PhysicsState):
    """The deterministic flag should be ignored (always deterministic)"""
    agent_state = pid_controller.init(key, obs_above_setpoint, pid_controller.params)
    action1, _ = pid_controller.select_action(
        key, obs_above_setpoint, agent_state, pid_controller.params, deterministic=True
    )
    action2, _ = pid_controller.select_action(
        key, obs_above_setpoint, agent_state, pid_controller.params, deterministic=False
    )
    chex.assert_trees_all_close(action1, action2)


# Test vector action spaces
def test_select_action_vector_action_space(key, vector_box_action_space: Box, obs_above_setpoint: PhysicsState):
    """Test Box with shape=(2,) works correctly"""
    agent = make_agent(action_space=vector_box_action_space, kp=1.0, setpoint=0.0, obs_field="theta")
    agent_state = agent.init(key, obs_above_setpoint, agent.params)
    action, _ = agent.select_action(key, obs_above_setpoint, agent_state, agent.params)

    # Should broadcast to action shape
    assert action.shape == (2,)
    # Both dimensions should have same value (broadcasted)
    chex.assert_trees_all_close(action, jnp.array([-0.5, -0.5]), atol=1e-6)


# Test update
def test_update(key, pid_controller: Agent, obs_above_setpoint: PhysicsState):
    """Test update returns unchanged state and empty metrics"""
    agent_state = pid_controller.init(key, obs_above_setpoint, pid_controller.params)
    updated_state, metrics = pid_controller.update(key, agent_state, None, pid_controller.params)
    assert isinstance(updated_state, AgentState)
    assert metrics == {}


# Test JAX JIT compatibility
def test_jit_compatibility_select_action(key, pid_controller: Agent, obs_above_setpoint: PhysicsState):
    """Verify select_action works with JAX JIT compilation"""
    agent_state = pid_controller.init(key, obs_above_setpoint, pid_controller.params)
    jitted_select = jax.jit(pid_controller.select_action, static_argnames=("deterministic",))
    action, state = jitted_select(key, obs_above_setpoint, agent_state, pid_controller.params, False)

    assert action.shape == (1,)
    assert isinstance(state, AgentState)


def test_jit_compatibility_init(key, pid_controller: Agent):
    """Verify agent init works with JAX JIT compilation"""
    jitted_init = jax.jit(pid_controller.init)
    sample_obs = PhysicsState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)
    state = jitted_init(key, sample_obs, pid_controller.params)
    assert isinstance(state, AgentState)


def test_jit_compatibility_update(key, pid_controller: Agent, obs_above_setpoint: PhysicsState):
    """Verify agent update works with JAX JIT compilation"""
    agent_state = pid_controller.init(key, obs_above_setpoint, pid_controller.params)
    jitted_update = jax.jit(pid_controller.update)
    updated_state, metrics = jitted_update(key, agent_state, None, pid_controller.params)
    assert isinstance(updated_state, AgentState)
    assert metrics == {}


# Test vmap compatibility (for vectorized environments)
def test_vmap_compatibility(key, pid_controller: Agent):
    """Verify agent works with vmap (batched observations)"""
    # Create batch of observations
    batch_size = 3
    obs_batch = PhysicsState(
        x=jnp.array([0.0, 0.0, 0.0]),
        x_dot=jnp.array([0.0, 0.0, 0.0]),
        theta=jnp.array([-0.5, 0.0, 0.5]),  # Below, at, above setpoint
        theta_dot=jnp.array([0.0, 0.0, 0.0]),
    )

    # Initialize agent state (uses first observation)
    sample_obs = PhysicsState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)
    agent_state = pid_controller.init(key, sample_obs, pid_controller.params)

    # Generate batch of keys
    keys = jax.random.split(key, batch_size)

    # Vmap select_action over the batch
    vmapped_select = jax.vmap(pid_controller.select_action, in_axes=(0, 0, None, None, None))
    actions, states = vmapped_select(keys, obs_batch, agent_state, pid_controller.params, False)

    # Check results
    assert actions.shape == (batch_size, 1)
    # First action should be positive (theta < 0), last should be negative (theta > 0)
    assert actions[0, 0] > 0
    assert actions[2, 0] < 0
