"""Tests for the CartPole system identification task.

This module tests the SysID task wrapper including:
- Environment creation with randomization
- Parameter randomization
- Reward computation (information-seeking)
- Info dict with true parameters
- Integration with base physics
"""

from typing import cast

import chex
import jax
import jax.numpy as jnp
import pytest

from aion.core import spaces
from aion.envs import make_env as make_env_from_registry
from aion.envs.cartpole.physics import PhysicsState
from aion.envs.cartpole.tasks.sysid import (
    SysIDTaskConfig,
    SysIDTaskParams,
    SysIDTaskState,
    _reset,
    _step,
    compute_sysid_reward,
    create_randomized_params,
    make_env,
)
from aion.envs.environment import Environment


@pytest.fixture
def env_config():
    """Default SysID environment configuration."""
    return SysIDTaskConfig()


@pytest.fixture
def env_params():
    """Default SysID environment parameters."""
    return SysIDTaskParams()


@pytest.fixture
def env() -> Environment:
    """Create a default SysID task environment."""
    return make_env()


def test_default_env_config(env_config: SysIDTaskConfig):
    """Test that default config has sensible values."""
    assert env_config.physics.gravity > 0
    assert env_config.physics.cart_mass > 0
    assert env_config.physics.pole_mass > 0
    assert env_config.physics.pole_length > 0
    assert env_config.physics.force_magnitude > 0
    assert env_config.physics.dt > 0
    assert env_config.task.theta_threshold > 0
    assert env_config.task.x_threshold > 0
    assert env_config.max_steps > 0

    # Check SysID-specific config
    assert env_config.reward_type in ["state_change", "action_diversity", "sparse"]
    assert env_config.reward_scale > 0
    assert env_config.pole_mass_min > 0
    assert env_config.pole_mass_max > env_config.pole_mass_min
    assert env_config.pole_length_min > 0
    assert env_config.pole_length_max > env_config.pole_length_min


def test_create_env_params(env_params: SysIDTaskParams):
    """Test environment parameters creation."""
    assert isinstance(env_params, SysIDTaskParams)
    # Default params should have nominal values
    assert env_params.pole_mass == 0.1
    assert env_params.pole_length == 0.5


def test_make_default_env(env_config: SysIDTaskConfig):
    """Test making environment with defaults."""
    env = make_env()

    assert env.config == env_config
    assert isinstance(env.params, SysIDTaskParams)


def test_make_env_with_custom_reward_type():
    """Test making environment with different reward types."""
    env_state_change = make_env(reward_type="state_change")
    env_action_div = make_env(reward_type="action_diversity")
    env_sparse = make_env(reward_type="sparse")

    assert env_state_change.config.reward_type == "state_change"
    assert env_action_div.config.reward_type == "action_diversity"
    assert env_sparse.config.reward_type == "sparse"


def test_make_env_with_custom_randomization_ranges():
    """Test making environment with custom parameter ranges."""
    env = make_env(
        pole_mass_min=0.08,
        pole_mass_max=0.12,
        pole_length_min=0.4,
        pole_length_max=0.6,
    )

    assert env.config.pole_mass_min == 0.08
    assert env.config.pole_mass_max == 0.12
    assert env.config.pole_length_min == 0.4
    assert env.config.pole_length_max == 0.6


def test_create_randomized_params(env_config: SysIDTaskConfig):
    """Test parameter randomization."""
    key = jax.random.key(42)
    params = create_randomized_params(key, env_config)

    # Parameters should be within configured ranges
    assert env_config.pole_mass_min <= params.pole_mass <= env_config.pole_mass_max
    assert env_config.pole_length_min <= params.pole_length <= env_config.pole_length_max


def test_randomized_params_are_different():
    """Test that different keys produce different parameters."""
    config = SysIDTaskConfig()
    key1 = jax.random.key(0)
    key2 = jax.random.key(1)

    params1 = create_randomized_params(key1, config)
    params2 = create_randomized_params(key2, config)

    # Should be different
    assert not jnp.allclose(params1.pole_mass, params2.pole_mass)
    assert not jnp.allclose(params1.pole_length, params2.pole_length)


def test_randomized_params_are_deterministic():
    """Test that same key produces same parameters."""
    config = SysIDTaskConfig()
    key = jax.random.key(123)

    params1 = create_randomized_params(key, config)
    params2 = create_randomized_params(key, config)

    # Same key should produce identical results
    chex.assert_trees_all_equal(params1, params2)


def test_get_action_space(env: Environment):
    """Test action space is discrete with 2 actions."""
    space = env.get_action_space(env.config)
    space = cast(spaces.Discrete, space)

    assert isinstance(space, spaces.Discrete)
    assert space.n == 2
    assert space.shape == ()


def test_get_obs_shape(env: Environment):
    """Test observation shape is (4,)."""
    assert env.get_obs_shape(env.config) == (4,)


def test_reset(key: chex.PRNGKey, env: Environment):
    """Test reset produces valid initial state."""
    obs, state = _reset(key, env.params, env.config)

    # Check state is properly initialized
    assert isinstance(state, SysIDTaskState)
    assert state.t == 0

    # All state values should be small (between -0.05 and 0.05)
    assert -0.05 <= state.physics.x <= 0.05
    assert -0.05 <= state.physics.x_dot <= 0.05
    assert -0.05 <= state.physics.theta <= 0.05
    assert -0.05 <= state.physics.theta_dot <= 0.05

    # Observation should match state
    assert obs.shape == env.get_obs_shape(env.config)
    chex.assert_trees_all_close(
        obs, jnp.array([state.physics.x, state.physics.x_dot, state.physics.theta, state.physics.theta_dot])
    )


def test_reset_is_random(env: Environment):
    """Test that reset produces different initial states."""
    key1 = jax.random.key(0)
    key2 = jax.random.key(1)

    obs1, state1 = _reset(key1, env.params, env.config)
    obs2, state2 = _reset(key2, env.params, env.config)

    # States should be different
    assert not jnp.allclose(obs1, obs2)


def test_step_basic(key: chex.PRNGKey, env: Environment):
    """Test basic step function."""
    state = SysIDTaskState(
        physics=PhysicsState(
            x=jnp.array(0.0),
            x_dot=jnp.array(0.0),
            theta=jnp.array(0.01),
            theta_dot=jnp.array(0.0),
        ),
        t=jnp.array(0),
    )

    action = jnp.array(1)
    obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Check output shapes and types
    assert obs.shape == (4,)
    assert isinstance(next_state, SysIDTaskState)
    assert reward.shape == ()
    assert done.shape == ()
    assert isinstance(info, dict)

    # Time should increment
    assert next_state.t == 1

    # Should still be running (not done)
    assert done == 0.0

    # Observation should match state
    chex.assert_trees_all_close(
        obs,
        jnp.array(
            [next_state.physics.x, next_state.physics.x_dot, next_state.physics.theta, next_state.physics.theta_dot]
        ),
    )


def test_step_info_contains_true_params(key: chex.PRNGKey, env: Environment):
    """Test that step returns true physics parameters in info dict."""
    obs, state = env.reset(key, env.params, env.config)
    action = jnp.array(1)

    obs, state, reward, done, info = env.step(key, state, action, env.params, env.config)

    # Info should contain true parameters
    assert "true_pole_mass" in info
    assert "true_pole_length" in info

    # Values should match the params
    assert jnp.allclose(info["true_pole_mass"], env.params.pole_mass)
    assert jnp.allclose(info["true_pole_length"], env.params.pole_length)


def test_step_with_randomized_params(key: chex.PRNGKey):
    """Test stepping with randomized parameters."""
    env = make_env()
    config = env.config

    # Create randomized params
    param_key, reset_key, step_key = jax.random.split(key, 3)
    params = create_randomized_params(param_key, config)

    # Reset and step
    obs, state = _reset(reset_key, params, config)
    obs, state, reward, done, info = _step(step_key, state, jnp.array(1), params, config)

    # Should work fine
    assert obs.shape == (4,)
    assert state.t == 1

    # Info should contain the randomized parameters
    assert jnp.allclose(info["true_pole_mass"], params.pole_mass)
    assert jnp.allclose(info["true_pole_length"], params.pole_length)


def test_compute_sysid_reward_state_change():
    """Test state_change reward computation."""
    config = SysIDTaskConfig(reward_type="state_change", reward_scale=1.0)

    prev_state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )

    next_state = PhysicsState(
        x=jnp.array(0.1),
        x_dot=jnp.array(0.2),
        theta=jnp.array(0.05),
        theta_dot=jnp.array(0.1),
    )

    reward = compute_sysid_reward(prev_state, next_state, jnp.array(1), config)

    # Reward should be positive (state changed)
    assert reward > 0

    # Should be approximately the L2 norm of state changes
    expected_diff = jnp.linalg.norm(jnp.array([0.1, 0.2, 0.05, 0.1]))
    assert jnp.allclose(reward, expected_diff, atol=1e-5)


def test_compute_sysid_reward_no_change():
    """Test that zero state change gives zero reward."""
    config = SysIDTaskConfig(reward_type="state_change", reward_scale=1.0)

    state = PhysicsState(
        x=jnp.array(0.5),
        x_dot=jnp.array(0.3),
        theta=jnp.array(0.1),
        theta_dot=jnp.array(0.2),
    )

    reward = compute_sysid_reward(state, state, jnp.array(1), config)

    # No change should give zero reward
    assert reward == 0.0


def test_compute_sysid_reward_action_diversity():
    """Test action_diversity reward type."""
    config = SysIDTaskConfig(reward_type="action_diversity", reward_scale=2.0)

    prev_state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )

    next_state = PhysicsState(
        x=jnp.array(0.1),
        x_dot=jnp.array(0.2),
        theta=jnp.array(0.05),
        theta_dot=jnp.array(0.1),
    )

    reward = compute_sysid_reward(prev_state, next_state, jnp.array(1), config)

    # Should return constant scaled reward
    assert reward == 2.0


def test_compute_sysid_reward_sparse():
    """Test sparse reward type."""
    config = SysIDTaskConfig(reward_type="sparse", reward_scale=1.0)

    prev_state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )

    next_state = PhysicsState(
        x=jnp.array(0.1),
        x_dot=jnp.array(0.2),
        theta=jnp.array(0.05),
        theta_dot=jnp.array(0.1),
    )

    reward = compute_sysid_reward(prev_state, next_state, jnp.array(1), config)

    # Sparse reward should always be zero
    assert reward == 0.0


def test_reward_scale_parameter():
    """Test that reward_scale parameter scales rewards correctly."""
    config1 = SysIDTaskConfig(reward_type="state_change", reward_scale=1.0)
    config2 = SysIDTaskConfig(reward_type="state_change", reward_scale=2.0)

    prev_state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )

    next_state = PhysicsState(
        x=jnp.array(0.1),
        x_dot=jnp.array(0.2),
        theta=jnp.array(0.05),
        theta_dot=jnp.array(0.1),
    )

    reward1 = compute_sysid_reward(prev_state, next_state, jnp.array(1), config1)
    reward2 = compute_sysid_reward(prev_state, next_state, jnp.array(1), config2)

    # reward2 should be twice reward1
    assert jnp.allclose(reward2, reward1 * 2.0)


def test_step_termination_theta(key: chex.PRNGKey, env: Environment):
    """Test termination when pole angle exceeds threshold."""
    state = SysIDTaskState(
        physics=PhysicsState(
            x=jnp.array(0.0),
            x_dot=jnp.array(0.0),
            theta=jnp.array(env.config.task.theta_threshold * 0.95),
            theta_dot=jnp.array(5.0),
        ),
        t=jnp.array(0),
    )

    action = jnp.array(0)
    obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Should be terminated
    assert done == 1.0


def test_step_termination_x(key: chex.PRNGKey, env: Environment):
    """Test termination when cart position exceeds threshold."""
    state = SysIDTaskState(
        physics=PhysicsState(
            x=jnp.array(env.config.task.x_threshold * 0.95),
            x_dot=jnp.array(10.0),
            theta=jnp.array(0.0),
            theta_dot=jnp.array(0.0),
        ),
        t=jnp.array(0),
    )

    action = jnp.array(1)
    obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Should be terminated
    assert done == 1.0


def test_step_termination_max_steps(key: chex.PRNGKey, env: Environment):
    """Test termination at max steps."""
    state = SysIDTaskState(
        physics=PhysicsState(
            x=jnp.array(0.0),
            x_dot=jnp.array(0.0),
            theta=jnp.array(0.0),
            theta_dot=jnp.array(0.0),
        ),
        t=jnp.array(env.config.max_steps - 1),
    )

    action = jnp.array(0)
    obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Should be done
    assert next_state.t == env.config.max_steps
    assert done == 1.0


def test_episode_rollout(env: Environment):
    """Test a full episode rollout."""
    key = jax.random.key(42)

    # Reset
    obs, state = env.reset(key, env.params, env.config)

    total_reward = 0.0
    steps = 0
    max_episode_steps = 100

    # Run episode
    for i in range(max_episode_steps):
        # Random action
        key, action_key = jax.random.split(key)
        action = env.get_action_space(env.config).sample(action_key)

        # Step
        key, step_key = jax.random.split(key)
        obs, state, reward, done, info = env.step(step_key, state, action, env.params, env.config)

        total_reward += reward
        steps += 1

        # Check info dict
        assert "true_pole_mass" in info
        assert "true_pole_length" in info

        if done == 1.0:
            break

    # Should have run at least one step
    assert steps > 0

    # If terminated early, either pole fell or cart went out
    if steps < max_episode_steps:
        theta_failed = jnp.abs(state.physics.theta) > env.config.task.theta_threshold
        x_failed = jnp.abs(state.physics.x) > env.config.task.x_threshold
        assert theta_failed or x_failed or state.t >= env.config.max_steps


def test_env_registry_integration():
    """Test that CartPole SysID can be created via ENV_REGISTRY."""
    env = make_env_from_registry("cartpole-sysid")

    # Verify it's a valid Environment
    assert isinstance(env, Environment)
    assert isinstance(env.config, SysIDTaskConfig)
    assert isinstance(env.params, SysIDTaskParams)

    # Verify it can be used
    key = jax.random.key(0)
    obs, state = env.reset(key, env.params, env.config)
    assert obs.shape == (4,)
    assert isinstance(state, SysIDTaskState)


def test_config_max_steps_property(env_config: SysIDTaskConfig):
    """Test that max_steps property works correctly."""
    assert env_config.max_steps == env_config.task.max_steps


def test_vmap_step(key: chex.PRNGKey, env: Environment):
    """Test that step can be vectorized."""
    batch_size = 4

    # Create batch of states
    states = SysIDTaskState(
        physics=PhysicsState(
            x=jnp.zeros(batch_size),
            x_dot=jnp.zeros(batch_size),
            theta=jnp.linspace(0.0, 0.1, batch_size),
            theta_dot=jnp.zeros(batch_size),
        ),
        t=jnp.zeros(batch_size, dtype=jnp.int32),
    )

    actions = jnp.ones(batch_size, dtype=jnp.int32)
    keys = jax.random.split(key, batch_size)

    # Vectorize step
    vmap_step = jax.vmap(_step, in_axes=(0, 0, 0, None, None))
    obs_batch, next_states, rewards, dones, infos = vmap_step(keys, states, actions, env.params, env.config)

    # Check shapes
    assert obs_batch.shape == (batch_size, 4)
    assert next_states.physics.x.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert dones.shape == (batch_size,)


def test_vmap_reset(key: chex.PRNGKey, env: Environment):
    """Test that reset can be vectorized."""
    batch_size = 5
    keys = jax.random.split(key, batch_size)

    # Vectorize reset
    vmap_reset = jax.vmap(_reset, in_axes=(0, None, None))
    obs_batch, states = vmap_reset(keys, env.params, env.config)

    # Check shapes
    assert obs_batch.shape == (batch_size, 4)
    assert states.physics.x.shape == (batch_size,)
    assert states.t.shape == (batch_size,)

    # All timesteps should be 0
    assert jnp.all(states.t == 0)


def test_different_params_produce_different_dynamics(key: chex.PRNGKey):
    """Test that different physics parameters produce different dynamics."""
    config = SysIDTaskConfig()

    # Create two different parameter sets
    key1, key2, reset_key, step_key = jax.random.split(key, 4)
    params1 = create_randomized_params(key1, config)
    params2 = create_randomized_params(key2, config)

    # Same initial state
    state = SysIDTaskState(
        physics=PhysicsState(
            x=jnp.array(0.0),
            x_dot=jnp.array(0.0),
            theta=jnp.array(0.1),
            theta_dot=jnp.array(0.0),
        ),
        t=jnp.array(0),
    )

    # Same action
    action = jnp.array(1)

    # Step with different params
    obs1, state1, _, _, _ = _step(step_key, state, action, params1, config)
    obs2, state2, _, _, _ = _step(step_key, state, action, params2, config)

    # Should produce different next states
    assert not jnp.allclose(obs1, obs2, atol=1e-6)
