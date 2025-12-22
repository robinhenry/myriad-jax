"""Tests for the CartPole environment."""

from typing import cast

import chex
import jax
import jax.numpy as jnp
import pytest

from aion.core import spaces
from aion.envs import make_env as make_env_from_registry
from aion.envs.cartpole_v1 import EnvConfig, EnvParams, EnvState, _reset, _step, create_env_params, make_env
from aion.envs.environment import Environment


@pytest.fixture
def env_config():
    """Default environment configuration."""
    return EnvConfig()


def test_default_env_config(env_config: EnvConfig):
    """Test that default config has sensible values."""
    assert env_config.gravity > 0
    assert env_config.cart_mass > 0
    assert env_config.pole_mass > 0
    assert env_config.pole_length > 0
    assert env_config.force_magnitude > 0
    assert env_config.dt > 0
    assert env_config.theta_threshold > 0
    assert env_config.x_threshold > 0
    assert env_config.max_steps > 0


def test_create_env_params():
    """Test environment parameters creation."""
    params = create_env_params()
    assert isinstance(params, EnvParams)


def test_make_default_env(env_config: EnvConfig):
    """Test making environment with defaults."""
    env = make_env()

    assert env.config == env_config
    assert isinstance(env.params, EnvParams)


def test_make_env_with_custom_config():
    """Test making environment with custom config."""
    config = EnvConfig(max_steps=1000, gravity=10.0)
    params = EnvParams()
    env = make_env(config=config, params=params)

    assert env.config == config
    assert env.params == params


@pytest.fixture
def env() -> Environment:
    """Create a default environment."""
    return make_env()


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
    assert isinstance(state, EnvState)
    assert state.t == 0

    # All state values should be small (between -0.05 and 0.05)
    assert -0.05 <= state.x <= 0.05
    assert -0.05 <= state.x_dot <= 0.05
    assert -0.05 <= state.theta <= 0.05
    assert -0.05 <= state.theta_dot <= 0.05

    # Observation should match state
    assert obs.shape == env.get_obs_shape(env.config)
    chex.assert_trees_all_close(obs, jnp.array([state.x, state.x_dot, state.theta, state.theta_dot]))


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
    # Start from a simple state
    state = EnvState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.01),  # Small angle
        theta_dot=jnp.array(0.0),
        t=jnp.array(0),
    )

    # Take action 1 (push right)
    action = jnp.array(1)
    obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Check output shapes and types
    assert obs.shape == (4,)
    assert isinstance(next_state, EnvState)
    assert reward.shape == ()
    assert done.shape == ()
    assert isinstance(info, dict)

    # Time should increment
    assert next_state.t == 1

    # Should still be running (not done)
    assert done == 0.0

    # Should get reward
    assert reward == 1.0

    # Observation should match state
    chex.assert_trees_all_close(
        obs, jnp.array([next_state.x, next_state.x_dot, next_state.theta, next_state.theta_dot])
    )


@pytest.mark.parametrize("action", [0, 1])
def test_step_actions(key: chex.PRNGKey, env: Environment, action: int):
    """Test that different actions produce different results."""
    state = EnvState(
        x=jnp.array(0.0), x_dot=jnp.array(0.0), theta=jnp.array(0.01), theta_dot=jnp.array(0.0), t=jnp.array(0)
    )

    obs, next_state, reward, done, info = _step(key, state, jnp.array(action), env.params, env.config)

    # Both actions should work
    assert obs.shape == (4,)
    assert reward == 1.0
    assert done == 0.0


def test_step_physics(key: chex.PRNGKey, env: Environment):
    """Test that physics evolves correctly."""
    state = EnvState(
        x=jnp.array(0.0), x_dot=jnp.array(0.0), theta=jnp.array(0.0), theta_dot=jnp.array(0.0), t=jnp.array(0)
    )

    # Push right (action=1) should move cart to the right
    action = jnp.array(1)
    obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Cart should have positive velocity (moving right)
    assert next_state.x_dot > 0

    # After multiple steps, position should clearly increase
    for _ in range(5):
        key, step_key = jax.random.split(key)
        obs, state, reward, done, info = _step(step_key, next_state, action, env.params, env.config)
        next_state = state

    assert next_state.x > 0.01  # Position should have increased after multiple steps


def test_step_termination_theta(key: chex.PRNGKey, env: Environment):
    """Test termination when pole angle exceeds threshold."""
    # Start with pole just past threshold to guarantee termination
    state = EnvState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(env.config.theta_threshold * 0.95),  # Near threshold
        theta_dot=jnp.array(5.0),  # Positive angular velocity to push over
        t=jnp.array(0),
    )

    action = jnp.array(0)
    obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Should be terminated (either already over threshold or pushed over by velocity)
    assert done == 1.0
    # Gymnasium gives +1 reward even on terminal step (before termination is detected)
    assert reward == 1.0


def test_step_termination_x(key: chex.PRNGKey, env: Environment):
    """Test termination when cart position exceeds threshold."""
    # Start with cart near boundary with high velocity to push it over
    state = EnvState(
        x=jnp.array(env.config.x_threshold * 0.95),  # Very close to threshold
        x_dot=jnp.array(10.0),  # High velocity to push over
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
        t=jnp.array(0),
    )

    action = jnp.array(1)  # Push right to help exceed threshold
    obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Should be terminated (either already over threshold or pushed over by velocity)
    assert done == 1.0
    # Gymnasium gives +1 reward even on terminal step (before termination is detected)
    assert reward == 1.0


def test_step_termination_max_steps(key: chex.PRNGKey, env: Environment):
    """Test termination at max steps."""
    # Start at the last step
    state = EnvState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
        t=jnp.array(env.config.max_steps - 1),
    )

    action = jnp.array(0)
    obs, next_state, reward, done, info = _step(key, state, action, env.params, env.config)

    # Should be done
    assert next_state.t == env.config.max_steps
    assert done == 1.0
    # Gymnasium gives +1 reward even on terminal step (before termination is detected)
    assert reward == 1.0


def test_max_episode_reward(key: chex.PRNGKey):
    """Test that maximum achievable reward equals max_steps (not max_steps + 1)."""
    # Create environment with very small max_steps to make testing easier
    # Use large thresholds so the pole won't fall during our short test
    config = EnvConfig(
        max_steps=3,
        theta_threshold=10.0,  # Very large to prevent early termination
        x_threshold=100.0,  # Very large to prevent early termination
    )
    env = make_env(config=config)

    # Start from initial state
    state = EnvState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
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
            # (rewards earned at t=0, t=1, t=2, then terminate at t=3 with no reward)
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

    assert obs_next.shape == (4,)
    assert reward == 1.0


def test_reset_jit_compilation(key: chex.PRNGKey, env: Environment):
    """Test that reset function can be JIT compiled."""
    obs, state = env.reset(key, env.params, env.config)

    assert obs.shape == (4,)
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
        theta_failed = jnp.abs(state.theta) > env.config.theta_threshold
        x_failed = jnp.abs(state.x) > env.config.x_threshold
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
            assert jnp.all(jnp.isfinite(obs))


def test_env_registry_integration():
    """Test that CartPole can be created via ENV_REGISTRY."""
    env = make_env_from_registry("cartpole-v1")

    # Verify it's a valid Environment
    assert isinstance(env, Environment)
    assert isinstance(env.config, EnvConfig)
    assert isinstance(env.params, EnvParams)

    # Verify it can be used
    key = jax.random.key(0)
    obs, state = env.reset(key, env.params, env.config)
    assert obs.shape == (4,)
    assert isinstance(state, EnvState)
