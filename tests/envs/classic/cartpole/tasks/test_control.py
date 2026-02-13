"""Tests for the CartPole control task.

This module tests the control task wrapper including:
- Environment creation and configuration
- Reset and Step interface compliance
- Integration of termination and reward logic
- JAX transformations (JIT/VMAP)
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from myriad.core import spaces
from myriad.envs import make_env as make_env_from_registry
from myriad.envs.classic.cartpole.physics import PhysicsState
from myriad.envs.classic.cartpole.tasks.control import (
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
    env_custom = make_env(max_steps=1000, gravity=10.0, theta_threshold=0.3)
    assert env_custom.config.task.max_steps == 1000
    assert env_custom.config.physics.gravity == 10.0
    assert env_custom.config.task.theta_threshold == 0.3


def test_env_specs(env):
    """Test action space and observation shape."""
    # Action Space
    action_space = env.get_action_space(env.config)
    assert isinstance(action_space, spaces.Discrete)
    assert action_space.n == 2

    # Observation Shape
    assert env.get_obs_shape(env.config) == (4,)


def test_reset(env):
    """Test reset initializes state correctly (t=0, valid physics)."""
    key = jax.random.key(0)
    obs, state = _reset(key, env.params, env.config)

    assert isinstance(state, ControlTaskState)
    assert state.t == 0
    assert isinstance(obs, PhysicsState)

    # Check consistency between state and observation
    chex.assert_trees_all_equal(obs, state.physics)

    # Check bounds (integration check for sample_initial_physics)
    assert jnp.all(jnp.abs(obs.to_array()) <= 0.05)


def test_step_logic(env):
    """Test basic step mechanics: reward, time increment, info."""
    key = jax.random.key(0)
    obs, state = _reset(key, env.params, env.config)

    action = jnp.array(1)
    next_obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Time increments
    assert next_state.t == state.t + 1

    # Reward is always 1.0 for Control task
    assert reward == 1.0

    # Info is empty
    assert info == {}
    assert isinstance(done, (float, jnp.ndarray))


def test_termination_integration(env):
    """Integration test: Verify termination logic triggers within the step function."""
    key = jax.random.key(0)

    # 1. Max Steps
    # Start at max_steps - 1
    state = ControlTaskState(
        physics=PhysicsState(x=jnp.array(0.0), x_dot=jnp.array(0.0), theta=jnp.array(0.0), theta_dot=jnp.array(0.0)),
        t=jnp.array(env.config.max_steps - 1),
    )
    _, next_state, _, done, _ = _step(key, state, jnp.array(0), env.params, env.config)
    assert done == 1.0
    assert next_state.t == env.config.max_steps

    # 2. Bounds (Theta)
    # Start near threshold
    state = ControlTaskState(
        physics=PhysicsState(
            x=jnp.array(0.0),
            x_dot=jnp.array(0.0),
            theta=jnp.array(env.config.task.theta_threshold * 0.99),
            theta_dot=jnp.array(10.0),  # High velocity to cross threshold
        ),
        t=jnp.array(0),
    )
    _, _, _, done, _ = _step(key, state, jnp.array(1), env.params, env.config)
    assert done == 1.0


def test_jax_transforms(env):
    """Test JIT and VMAP compatibility for reset and step."""
    key = jax.random.key(0)
    keys = jax.random.split(key, 3)

    # JIT Reset & Step
    jitted_reset = jax.jit(_reset, static_argnames=["config"])
    jitted_step = jax.jit(_step, static_argnames=["config"])

    obs, state = jitted_reset(key, env.params, env.config)
    _, _, reward, _, _ = jitted_step(key, state, jnp.array(0), env.params, env.config)
    assert reward == 1.0

    # VMAP Reset
    vmap_reset = jax.vmap(_reset, in_axes=(0, None, None))
    obs_batch, state_batch = vmap_reset(keys, env.params, env.config)
    assert obs_batch.x.shape == (3,)

    # VMAP Step
    actions = jnp.zeros(3, dtype=jnp.int32)
    vmap_step = jax.vmap(_step, in_axes=(0, 0, 0, None, None))
    _, next_states, rewards, dones, _ = vmap_step(keys, state_batch, actions, env.params, env.config)

    assert rewards.shape == (3,)
    assert jnp.all(rewards == 1.0)


def test_config_dt_property():
    """ControlTaskConfig.dt delegates to physics.dt."""
    env = make_env()
    assert env.config.dt == env.config.physics.dt
    assert env.config.dt > 0


def test_env_registry_integration():
    """Verify registry loading."""
    env = make_env_from_registry("cartpole-control")
    assert isinstance(env, Environment)
    assert isinstance(env.config, ControlTaskConfig)
