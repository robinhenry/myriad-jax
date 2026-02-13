"""Tests for the Pendulum control task.

This module tests the control task wrapper including:
- Environment creation and configuration
- Reset and Step interface compliance
- Reward function correctness
- JAX transformations (JIT/VMAP)
"""

import jax
import jax.numpy as jnp
import pytest

from myriad.core import spaces
from myriad.envs import make_env as make_env_from_registry
from myriad.envs.classic.pendulum.tasks.base import PendulumObservation
from myriad.envs.classic.pendulum.tasks.control import (
    ControlTaskConfig,
    ControlTaskParams,
    ControlTaskState,
    _reset,
    _step,
    make_env,
)
from myriad.envs.environment import Environment


@pytest.fixture
def env():
    """Create a default control task environment."""
    return make_env()


def test_env_config_and_factory():
    """Test environment configuration defaults and factory overrides."""
    # Defaults
    env = make_env()
    assert isinstance(env.config, ControlTaskConfig)
    assert isinstance(env.params, ControlTaskParams)
    assert env.config.max_steps > 0
    assert env.config.physics.gravity > 0

    # Overrides
    env_custom = make_env(max_steps=500, gravity=10.0, max_torque=3.0)
    assert env_custom.config.task.max_steps == 500
    assert env_custom.config.physics.gravity == 10.0
    assert env_custom.config.physics.max_torque == 3.0


def test_env_specs(env):
    """Test action space and observation shape."""
    # Action Space
    action_space = env.get_action_space(env.config)
    assert isinstance(action_space, spaces.Box)
    assert action_space.shape == (1,)
    assert action_space.low == -env.config.physics.max_torque
    assert action_space.high == env.config.physics.max_torque

    # Observation Shape
    assert env.get_obs_shape(env.config) == (3,)


def test_reset(env):
    """Test reset initializes state correctly (t=0, valid physics)."""
    key = jax.random.key(0)
    obs, state = _reset(key, env.params, env.config)

    assert isinstance(state, ControlTaskState)
    assert state.t == 0
    assert isinstance(obs, PendulumObservation)

    # Check bounds
    assert -1 <= obs.cos_theta <= 1
    assert -1 <= obs.sin_theta <= 1


def test_step_logic(env):
    """Test basic step mechanics: reward sign, time increment, info."""
    key = jax.random.key(0)
    obs, state = _reset(key, env.params, env.config)

    action = jnp.array([0.0])  # Zero torque
    next_obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Time increments
    assert next_state.t == state.t + 1

    # Reward should be negative (cost function)
    assert reward <= 0

    # Info is empty
    assert info == {}
    assert isinstance(done, (float, jnp.ndarray))


def test_reward_function(env):
    """Test reward function components."""
    key = jax.random.key(0)

    # Create state at upright position (theta=pi)
    from myriad.envs.classic.pendulum.physics import PhysicsState

    upright_state = ControlTaskState(
        physics=PhysicsState(theta=jnp.array(jnp.pi), theta_dot=jnp.array(0.0)),
        t=jnp.array(0),
    )

    # Step with zero torque
    action = jnp.array([0.0])
    _, _, reward_upright, _, _ = _step(key, upright_state, action, env.params, env.config)

    # At upright with zero velocity and zero torque, reward should be close to 0
    # (only numerical noise from angle normalization)
    assert reward_upright > -0.1

    # Create state hanging down (theta=0)
    hanging_state = ControlTaskState(
        physics=PhysicsState(theta=jnp.array(0.0), theta_dot=jnp.array(0.0)),
        t=jnp.array(0),
    )

    _, _, reward_hanging, _, _ = _step(key, hanging_state, action, env.params, env.config)

    # Hanging down is far from upright, reward should be significantly negative
    # theta_from_up = -pi, so cost = pi^2 ~ 9.87
    assert reward_hanging < -5.0

    # Upright should have better reward than hanging
    assert reward_upright > reward_hanging


def test_torque_penalty(env):
    """Test that torque contributes to cost in the reward function."""
    # The reward includes a torque penalty term: -0.001 * torque^2
    # We verify this by comparing rewards with different torques applied to the same initial state
    # The torque penalty should make high-torque actions less rewarding, all else being equal

    key = jax.random.key(0)

    # Use upright state where theta_from_up is zero (no angle cost)
    from myriad.envs.classic.pendulum.physics import PhysicsState

    upright_state = ControlTaskState(
        physics=PhysicsState(theta=jnp.array(jnp.pi), theta_dot=jnp.array(0.0)),
        t=jnp.array(0),
    )

    # Zero torque - no torque penalty
    _, _, reward_zero, _, _ = _step(key, upright_state, jnp.array([0.0]), env.params, env.config)

    # Max torque - has torque penalty of 0.001 * max_torque^2
    max_torque = env.config.physics.max_torque
    _, _, reward_max, _, _ = _step(key, upright_state, jnp.array([max_torque]), env.params, env.config)

    # The torque penalty should make the high-torque reward lower
    # Expected difference: -0.001 * max_torque^2 = -0.001 * 4 = -0.004
    expected_penalty = 0.001 * max_torque**2
    assert reward_zero - reward_max > 0, "Torque should add cost to reward"
    # The difference should be close to the torque penalty (plus small state-change effects)
    assert reward_zero - reward_max > expected_penalty * 0.5, "Torque penalty should be noticeable"


def test_termination_only_at_max_steps(env):
    """Test that termination only occurs at max_steps (no early termination)."""
    key = jax.random.key(0)

    from myriad.envs.classic.pendulum.physics import PhysicsState

    # Create state with extreme values but not at max_steps
    extreme_state = ControlTaskState(
        physics=PhysicsState(theta=jnp.array(100.0), theta_dot=jnp.array(100.0)),
        t=jnp.array(0),
    )

    _, _, _, done, _ = _step(key, extreme_state, jnp.array([0.0]), env.params, env.config)
    assert done == 0.0

    # At max_steps - 1
    near_end_state = ControlTaskState(
        physics=PhysicsState(theta=jnp.array(0.0), theta_dot=jnp.array(0.0)),
        t=jnp.array(env.config.max_steps - 1),
    )

    _, _, _, done, _ = _step(key, near_end_state, jnp.array([0.0]), env.params, env.config)
    assert done == 1.0


def test_jax_transforms(env):
    """Test JIT and VMAP compatibility for reset and step."""
    key = jax.random.key(0)
    keys = jax.random.split(key, 3)

    # JIT Reset & Step
    jitted_reset = jax.jit(_reset, static_argnames=["config"])
    jitted_step = jax.jit(_step, static_argnames=["config"])

    obs, state = jitted_reset(key, env.params, env.config)
    _, _, reward, _, _ = jitted_step(key, state, jnp.array([0.0]), env.params, env.config)
    assert reward <= 0

    # VMAP Reset
    vmap_reset = jax.vmap(_reset, in_axes=(0, None, None))
    obs_batch, state_batch = vmap_reset(keys, env.params, env.config)
    assert obs_batch.cos_theta.shape == (3,)

    # VMAP Step
    actions = jnp.zeros((3, 1))
    vmap_step = jax.vmap(_step, in_axes=(0, 0, 0, None, None))
    _, next_states, rewards, dones, _ = vmap_step(keys, state_batch, actions, env.params, env.config)

    assert rewards.shape == (3,)
    assert jnp.all(rewards <= 0)


def test_config_dt_property():
    """ControlTaskConfig.dt delegates to physics.dt."""
    env = make_env()
    assert env.config.dt == env.config.physics.dt
    assert env.config.dt > 0


def test_env_registry_integration():
    """Verify registry loading."""
    env = make_env_from_registry("pendulum-control")
    assert isinstance(env, Environment)
    assert isinstance(env.config, ControlTaskConfig)
