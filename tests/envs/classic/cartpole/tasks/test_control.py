"""Tests for the CartPole control task.

This module tests the control task wrapper including:
- Environment creation
- Reset functionality
- Step functionality
- Reward computation
- Termination conditions
- Integration with base physics
"""

from typing import cast

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
def env_config():
    """Default environment configuration."""
    return ControlTaskConfig()


@pytest.fixture
def env_params():
    """Default environment parameters."""
    return ControlTaskParams()


@pytest.fixture
def env() -> Environment:
    """Create a default control task environment."""
    return make_env()


def test_default_env_config(env_config: ControlTaskConfig):
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


def test_create_env_params(env_params: ControlTaskParams):
    """Test environment parameters creation."""
    assert isinstance(env_params, ControlTaskParams)


def test_make_default_env(env_config: ControlTaskConfig):
    """Test making environment with defaults."""
    env = make_env()

    assert env.config == env_config
    assert isinstance(env.params, ControlTaskParams)


def test_make_env_with_custom_config():
    """Test making environment with custom config."""
    env = make_env(max_steps=1000, gravity=10.0)

    assert env.config.task.max_steps == 1000
    assert env.config.physics.gravity == 10.0
    assert isinstance(env.params, ControlTaskParams)


def test_make_env_with_custom_physics_params():
    """Test making environment with custom physics parameters."""
    env = make_env(force_magnitude=20.0, dt=0.01, pole_length=0.8)

    assert env.config.physics.force_magnitude == 20.0
    assert env.config.physics.dt == 0.01
    assert env.config.physics.pole_length == 0.8


def test_make_env_with_custom_task_params():
    """Test making environment with custom task parameters."""
    env = make_env(theta_threshold=0.3, x_threshold=3.0)

    assert env.config.task.theta_threshold == 0.3
    assert env.config.task.x_threshold == 3.0


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

    # Check observation type
    assert isinstance(obs, PhysicsState)

    # Check state is properly initialized
    assert isinstance(state, ControlTaskState)
    assert state.t == 0

    # All state values should be small (between -0.05 and 0.05)
    assert -0.05 <= state.physics.x <= 0.05
    assert -0.05 <= state.physics.x_dot <= 0.05
    assert -0.05 <= state.physics.theta <= 0.05
    assert -0.05 <= state.physics.theta_dot <= 0.05

    # Observation should match state (check named fields)
    assert obs.x == state.physics.x
    assert obs.x_dot == state.physics.x_dot
    assert obs.theta == state.physics.theta
    assert obs.theta_dot == state.physics.theta_dot

    # Check array conversion works
    assert obs.to_array().shape == env.get_obs_shape(env.config)


def test_reset_is_random(env: Environment):
    """Test that reset produces different initial states."""
    key1 = jax.random.key(0)
    key2 = jax.random.key(1)

    obs1, state1 = _reset(key1, env.params, env.config)
    obs2, state2 = _reset(key2, env.params, env.config)

    # States should be different (convert to arrays for comparison)
    assert not jnp.allclose(obs1.to_array(), obs2.to_array())


def test_reset_is_deterministic_with_same_key(env: Environment):
    """Test that same key produces same initial state."""
    key = jax.random.key(42)

    obs1, state1 = _reset(key, env.params, env.config)
    obs2, state2 = _reset(key, env.params, env.config)

    # Same key should produce identical results
    chex.assert_trees_all_equal(obs1, obs2)
    chex.assert_trees_all_equal(state1, state2)


def test_step_basic(key: chex.PRNGKey, env: Environment):
    """Test basic step function."""
    # Start from a simple state
    state = ControlTaskState(
        physics=PhysicsState(
            x=jnp.array(0.0),
            x_dot=jnp.array(0.0),
            theta=jnp.array(0.01),  # Small angle
            theta_dot=jnp.array(0.0),
        ),
        t=jnp.array(0),
    )

    # Take action 1 (push right)
    action = jnp.array(1)
    obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Check output types
    assert isinstance(obs, PhysicsState)
    assert isinstance(next_state, ControlTaskState)
    assert reward.shape == ()
    assert done.shape == ()
    assert isinstance(info, dict)

    # Check array conversion has correct shape
    assert obs.to_array().shape == (4,)

    # Time should increment
    assert next_state.t == 1

    # Should still be running (not done)
    assert done == 0.0

    # Should get reward
    assert reward == 1.0

    # Observation should match state (check named fields)
    assert obs.x == next_state.physics.x
    assert obs.x_dot == next_state.physics.x_dot
    assert obs.theta == next_state.physics.theta
    assert obs.theta_dot == next_state.physics.theta_dot


@pytest.mark.parametrize("action", [0, 1])
def test_step_actions(key: chex.PRNGKey, env: Environment, action: int):
    """Test that different actions produce different results."""
    state = ControlTaskState(
        physics=PhysicsState(x=jnp.array(0.0), x_dot=jnp.array(0.0), theta=jnp.array(0.01), theta_dot=jnp.array(0.0)),
        t=jnp.array(0),
    )

    obs, next_state, reward, done, info = _step(key, state, jnp.array(action), env.params, env.config)

    # Both actions should work
    assert isinstance(obs, PhysicsState)
    assert obs.to_array().shape == (4,)
    assert reward == 1.0
    assert done == 0.0


def test_step_reward_is_always_one(key: chex.PRNGKey, env: Environment):
    """Test that control task always returns reward of 1.0."""
    state = ControlTaskState(
        physics=PhysicsState(x=jnp.array(0.0), x_dot=jnp.array(0.0), theta=jnp.array(0.0), theta_dot=jnp.array(0.0)),
        t=jnp.array(0),
    )

    # Try multiple steps
    for _ in range(5):
        obs, state, reward, done, info = _step(key, state, jnp.array(1), env.params, env.config)
        assert reward == 1.0


def test_step_physics(key: chex.PRNGKey, env: Environment):
    """Test that physics evolves correctly."""
    state = ControlTaskState(
        physics=PhysicsState(x=jnp.array(0.0), x_dot=jnp.array(0.0), theta=jnp.array(0.0), theta_dot=jnp.array(0.0)),
        t=jnp.array(0),
    )

    # Push right (action=1) should move cart to the right
    action = jnp.array(1)
    obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Cart should have positive velocity (moving right)
    assert next_state.physics.x_dot > 0

    # After multiple steps, position should clearly increase
    for _ in range(5):
        key, step_key = jax.random.split(key)
        obs, state, reward, done, info = _step(step_key, next_state, action, env.params, env.config)
        next_state = state

    assert next_state.physics.x > 0.01  # Position should have increased after multiple steps


def test_step_termination_theta(key: chex.PRNGKey, env: Environment):
    """Test termination when pole angle exceeds threshold."""
    # Start with pole just past threshold to guarantee termination
    state = ControlTaskState(
        physics=PhysicsState(
            x=jnp.array(0.0),
            x_dot=jnp.array(0.0),
            theta=jnp.array(env.config.task.theta_threshold * 0.95),  # Near threshold
            theta_dot=jnp.array(5.0),  # Positive angular velocity to push over
        ),
        t=jnp.array(0),
    )

    action = jnp.array(0)
    obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Should be terminated (either already over threshold or pushed over by velocity)
    assert done == 1.0
    # CartPole gives +1 reward even on terminal step
    assert reward == 1.0


def test_step_termination_x(key: chex.PRNGKey, env: Environment):
    """Test termination when cart position exceeds threshold."""
    # Start with cart near boundary with high velocity to push it over
    state = ControlTaskState(
        physics=PhysicsState(
            x=jnp.array(env.config.task.x_threshold * 0.95),  # Very close to threshold
            x_dot=jnp.array(10.0),  # High velocity to push over
            theta=jnp.array(0.0),
            theta_dot=jnp.array(0.0),
        ),
        t=jnp.array(0),
    )

    action = jnp.array(1)  # Push right to help exceed threshold
    obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Should be terminated (either already over threshold or pushed over by velocity)
    assert done == 1.0
    # CartPole gives +1 reward even on terminal step
    assert reward == 1.0


def test_step_termination_max_steps(key: chex.PRNGKey, env: Environment):
    """Test termination at max steps."""
    # Start at the last step
    state = ControlTaskState(
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
    # CartPole gives +1 reward even on terminal step
    assert reward == 1.0


def test_max_episode_reward(key: chex.PRNGKey):
    """Test that maximum achievable reward equals max_steps."""
    # Create environment with very small max_steps to make testing easier
    # Use large thresholds so the pole won't fall during our short test
    env = make_env(max_steps=3, theta_threshold=10.0, x_threshold=100.0)
    config = env.config

    # Start from initial state
    state = ControlTaskState(
        physics=PhysicsState(
            x=jnp.array(0.0),
            x_dot=jnp.array(0.0),
            theta=jnp.array(0.0),
            theta_dot=jnp.array(0.0),
        ),
        t=jnp.array(0),
    )

    total_reward = 0.0
    action = jnp.array(0)

    # Step through entire episode
    for step_num in range(config.max_steps + 2):  # Try to go beyond max_steps
        key, step_key = jax.random.split(key)
        obs, state, reward, done, info = _step(step_key, state, action, env.params, config)
        total_reward += reward

        if done == 1.0:
            # Should terminate at exactly max_steps (3)
            assert state.t == config.max_steps
            # Total reward should be exactly max_steps
            assert total_reward == config.max_steps
            break
    else:
        # Should never get here - episode should have terminated
        assert False, "Episode did not terminate"


def test_step_jit_compilation(key: chex.PRNGKey, env: Environment):
    """Test that step function can be JIT compiled and works correctly."""
    obs, state = env.reset(key, env.params, env.config)

    # Step should be JIT compiled
    action = jnp.array(1)
    obs_next, next_state, reward, done, info = env.step(key, state, action, env.params, env.config)

    assert isinstance(obs_next, PhysicsState)
    assert obs_next.to_array().shape == (4,)
    assert reward == 1.0


def test_reset_jit_compilation(key: chex.PRNGKey, env: Environment):
    """Test that reset function can be JIT compiled."""
    obs, state = env.reset(key, env.params, env.config)

    assert isinstance(obs, PhysicsState)
    assert obs.to_array().shape == (4,)
    assert state.t == 0


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

        if done == 1.0:
            break

    # Should have run at least one step
    assert steps > 0
    assert total_reward > 0

    # If terminated early, either pole fell or cart went out
    if steps < max_episode_steps:
        theta_failed = jnp.abs(state.physics.theta) > env.config.task.theta_threshold
        x_failed = jnp.abs(state.physics.x) > env.config.task.x_threshold
        assert theta_failed or x_failed or state.t >= env.config.max_steps


def test_observation_bounds(key: chex.PRNGKey, env: Environment):
    """Test that observations are within reasonable bounds during normal operation."""
    obs, state = env.reset(key, env.params, env.config)

    # Run a few steps
    for _ in range(10):
        key, action_key, step_key = jax.random.split(key, 3)
        action = env.get_action_space(env.config).sample(action_key)
        obs, state, reward, done, info = env.step(step_key, state, action, env.params, env.config)

        if done == 0.0:
            # While episode is running, observations should be reasonable
            assert jnp.all(jnp.isfinite(obs.to_array()))


def test_env_registry_integration():
    """Test that CartPole can be created via ENV_REGISTRY."""
    env = make_env_from_registry("cartpole-control")

    # Verify it's a valid Environment
    assert isinstance(env, Environment)
    assert isinstance(env.config, ControlTaskConfig)
    assert isinstance(env.params, ControlTaskParams)

    # Verify it can be used
    key = jax.random.key(0)
    obs, state = env.reset(key, env.params, env.config)
    assert isinstance(obs, PhysicsState)
    assert obs.to_array().shape == (4,)
    assert isinstance(state, ControlTaskState)


def test_step_info_dict_is_empty(key: chex.PRNGKey, env: Environment):
    """Test that control task returns empty info dict."""
    obs, state = env.reset(key, env.params, env.config)
    action = jnp.array(1)

    obs, state, reward, done, info = env.step(key, state, action, env.params, env.config)

    assert isinstance(info, dict)
    assert len(info) == 0


def test_config_max_steps_property(env_config: ControlTaskConfig):
    """Test that max_steps property works correctly."""
    assert env_config.max_steps == env_config.task.max_steps


def test_vmap_step(key: chex.PRNGKey, env: Environment):
    """Test that step can be vectorized."""
    batch_size = 4

    # Create batch of states
    states = ControlTaskState(
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

    # Check types and shapes
    # vmap over NamedTuples gives NamedTuple with batched fields
    assert isinstance(obs_batch, PhysicsState)
    assert obs_batch.x.shape == (batch_size,)
    assert obs_batch.x_dot.shape == (batch_size,)
    assert obs_batch.theta.shape == (batch_size,)
    assert obs_batch.theta_dot.shape == (batch_size,)
    assert next_states.physics.x.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert dones.shape == (batch_size,)

    # All rewards should be 1.0
    assert jnp.all(rewards == 1.0)


def test_vmap_reset(key: chex.PRNGKey, env: Environment):
    """Test that reset can be vectorized."""
    batch_size = 5
    keys = jax.random.split(key, batch_size)

    # Vectorize reset
    vmap_reset = jax.vmap(_reset, in_axes=(0, None, None))
    obs_batch, states = vmap_reset(keys, env.params, env.config)

    # Check types and shapes
    # vmap over NamedTuples gives NamedTuple with batched fields
    assert isinstance(obs_batch, PhysicsState)
    assert obs_batch.x.shape == (batch_size,)
    assert obs_batch.x_dot.shape == (batch_size,)
    assert obs_batch.theta.shape == (batch_size,)
    assert obs_batch.theta_dot.shape == (batch_size,)
    assert states.physics.x.shape == (batch_size,)
    assert states.t.shape == (batch_size,)

    # All timesteps should be 0
    assert jnp.all(states.t == 0)
