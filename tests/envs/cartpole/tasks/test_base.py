"""Tests for shared CartPole task utilities.

This module tests the base task utilities used by all CartPole tasks:
- Termination checking
- Observation extraction
- Action space definition
- Initial state sampling
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from myriad.core.spaces import Discrete
from myriad.envs.cartpole.physics import PhysicsState
from myriad.envs.cartpole.tasks.base import (
    TaskConfig,
    check_termination,
    get_cartpole_action_space,
    get_cartpole_obs,
    get_cartpole_obs_shape,
    sample_initial_physics,
)


@pytest.fixture
def task_config():
    """Default task configuration."""
    return TaskConfig()


@pytest.fixture
def physics_state():
    """Sample physics state for testing."""
    return PhysicsState(
        x=jnp.array(0.5),
        x_dot=jnp.array(0.3),
        theta=jnp.array(0.1),
        theta_dot=jnp.array(0.2),
    )


def test_task_config_defaults(task_config: TaskConfig):
    """Test that default task config has reasonable values."""
    assert task_config.max_steps > 0
    assert task_config.theta_threshold > 0
    assert task_config.x_threshold > 0
    # Check that theta threshold is approximately 12 degrees
    assert jnp.isclose(task_config.theta_threshold, jnp.deg2rad(12), atol=0.01)


def test_check_termination_theta_exceeds(task_config: TaskConfig):
    """Test termination when pole angle exceeds threshold."""
    # Angle just over threshold
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(task_config.theta_threshold * 1.01),
        theta_dot=jnp.array(0.0),
    )
    t = jnp.array(0)

    done = check_termination(state, t, task_config)
    assert done == 1.0


def test_check_termination_x_exceeds(task_config: TaskConfig):
    """Test termination when cart position exceeds threshold."""
    # Position just over threshold
    state = PhysicsState(
        x=jnp.array(task_config.x_threshold * 1.01),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )
    t = jnp.array(0)

    done = check_termination(state, t, task_config)
    assert done == 1.0


def test_check_termination_negative_x_exceeds(task_config: TaskConfig):
    """Test termination when cart position exceeds negative threshold."""
    # Position just under negative threshold
    state = PhysicsState(
        x=jnp.array(-task_config.x_threshold * 1.01),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )
    t = jnp.array(0)

    done = check_termination(state, t, task_config)
    assert done == 1.0


def test_check_termination_negative_theta_exceeds(task_config: TaskConfig):
    """Test termination when pole angle exceeds negative threshold."""
    # Angle just under negative threshold
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(-task_config.theta_threshold * 1.01),
        theta_dot=jnp.array(0.0),
    )
    t = jnp.array(0)

    done = check_termination(state, t, task_config)
    assert done == 1.0


def test_check_termination_max_steps(task_config: TaskConfig):
    """Test termination when max steps reached."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )
    t = jnp.array(task_config.max_steps)

    done = check_termination(state, t, task_config)
    assert done == 1.0


def test_check_termination_not_done(task_config: TaskConfig):
    """Test that episode continues when conditions are not met."""
    state = PhysicsState(
        x=jnp.array(0.5),
        x_dot=jnp.array(0.1),
        theta=jnp.array(0.05),
        theta_dot=jnp.array(0.1),
    )
    t = jnp.array(10)

    done = check_termination(state, t, task_config)
    assert done == 0.0


def test_check_termination_boundary_theta(task_config: TaskConfig):
    """Test termination at exact theta threshold boundary."""
    # Exactly at threshold (should not terminate)
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(task_config.theta_threshold),
        theta_dot=jnp.array(0.0),
    )
    t = jnp.array(0)

    done = check_termination(state, t, task_config)
    # At boundary, should not terminate (only > threshold terminates)
    assert done == 0.0


def test_check_termination_boundary_x(task_config: TaskConfig):
    """Test termination at exact x threshold boundary."""
    # Exactly at threshold (should not terminate)
    state = PhysicsState(
        x=jnp.array(task_config.x_threshold),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )
    t = jnp.array(0)

    done = check_termination(state, t, task_config)
    # At boundary, should not terminate (only > threshold terminates)
    assert done == 0.0


def test_check_termination_one_step_before_max(task_config: TaskConfig):
    """Test that episode continues one step before max."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )
    t = jnp.array(task_config.max_steps - 1)

    done = check_termination(state, t, task_config)
    assert done == 0.0


def test_get_cartpole_obs(physics_state: PhysicsState):
    """Test observation extraction from physics state."""
    obs = get_cartpole_obs(physics_state)

    # Should be 1D array of shape (4,)
    assert obs.shape == (4,)

    # Should contain [x, x_dot, theta, theta_dot]
    expected_obs = jnp.array(
        [
            physics_state.x,
            physics_state.x_dot,
            physics_state.theta,
            physics_state.theta_dot,
        ]
    )
    chex.assert_trees_all_close(obs, expected_obs)


def test_get_cartpole_obs_shape():
    """Test observation shape getter."""
    shape = get_cartpole_obs_shape()
    assert shape == (4,)
    assert isinstance(shape, tuple)


def test_get_cartpole_action_space():
    """Test action space definition."""
    action_space = get_cartpole_action_space()

    assert isinstance(action_space, Discrete)
    assert action_space.n == 2
    assert action_space.shape == ()


def test_action_space_sampling():
    """Test that action space can sample valid actions."""
    action_space = get_cartpole_action_space()
    key = jax.random.key(0)

    # Sample multiple actions
    for i in range(10):
        key, subkey = jax.random.split(key)
        action = action_space.sample(subkey)

        # Action should be 0 or 1
        assert action in [0, 1]


def test_sample_initial_physics():
    """Test initial physics state sampling."""
    key = jax.random.key(42)
    physics = sample_initial_physics(key)

    # Should be a PhysicsState
    assert isinstance(physics, PhysicsState)

    # All values should be in range [-0.05, 0.05]
    assert -0.05 <= physics.x <= 0.05
    assert -0.05 <= physics.x_dot <= 0.05
    assert -0.05 <= physics.theta <= 0.05
    assert -0.05 <= physics.theta_dot <= 0.05


def test_sample_initial_physics_randomness():
    """Test that initial physics sampling is random."""
    key1 = jax.random.key(0)
    key2 = jax.random.key(1)

    physics1 = sample_initial_physics(key1)
    physics2 = sample_initial_physics(key2)

    # Should produce different initial states
    assert not jnp.allclose(physics1.x, physics2.x)


def test_sample_initial_physics_repeatability():
    """Test that same key produces same initial state."""
    key = jax.random.key(123)

    physics1 = sample_initial_physics(key)
    physics2 = sample_initial_physics(key)

    # Same key should produce identical results
    chex.assert_trees_all_equal(physics1, physics2)


def test_check_termination_jit_compilation(task_config: TaskConfig):
    """Test that termination check can be JIT compiled."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.1),
        theta_dot=jnp.array(0.0),
    )
    t = jnp.array(10)

    # JIT compile
    jitted_check = jax.jit(check_termination, static_argnames=["task_config"])

    # Should work
    done = jitted_check(state, t, task_config)
    assert done == 0.0


def test_check_termination_vmap_compatibility(task_config: TaskConfig):
    """Test that termination check can be vectorized."""
    batch_size = 5

    # Create batch of states with different x positions
    states = PhysicsState(
        x=jnp.linspace(-3.0, 3.0, batch_size),  # Some will exceed threshold
        x_dot=jnp.zeros(batch_size),
        theta=jnp.zeros(batch_size),
        theta_dot=jnp.zeros(batch_size),
    )
    t = jnp.zeros(batch_size, dtype=jnp.int32)

    # Vectorize
    vmap_check = jax.vmap(check_termination, in_axes=(0, 0, None))
    dones = vmap_check(states, t, task_config)

    # Should be batch of termination flags
    assert dones.shape == (batch_size,)

    # States beyond threshold should be done
    assert dones[0] == 1.0  # x = -3.0 exceeds threshold
    assert dones[-1] == 1.0  # x = 3.0 exceeds threshold
    assert dones[batch_size // 2] == 0.0  # x = 0.0 is fine


def test_get_cartpole_obs_vmap_compatibility():
    """Test that observation extraction can be vectorized."""
    batch_size = 3

    states = PhysicsState(
        x=jnp.array([0.0, 0.5, 1.0]),
        x_dot=jnp.array([0.1, 0.2, 0.3]),
        theta=jnp.array([0.01, 0.02, 0.03]),
        theta_dot=jnp.array([0.1, 0.2, 0.3]),
    )

    # Vectorize
    vmap_obs = jax.vmap(get_cartpole_obs)
    obs_batch = vmap_obs(states)

    # Should be batch of observations
    assert obs_batch.shape == (batch_size, 4)

    # Check first observation
    expected_first = jnp.array([states.x[0], states.x_dot[0], states.theta[0], states.theta_dot[0]])
    chex.assert_trees_all_close(obs_batch[0], expected_first)


def test_sample_initial_physics_vmap_compatibility():
    """Test that initial sampling can be vectorized."""
    batch_size = 10
    key = jax.random.key(0)
    keys = jax.random.split(key, batch_size)

    # Vectorize
    vmap_sample = jax.vmap(sample_initial_physics)
    physics_batch = vmap_sample(keys)

    # Should be batch of states
    assert physics_batch.x.shape == (batch_size,)
    assert physics_batch.x_dot.shape == (batch_size,)
    assert physics_batch.theta.shape == (batch_size,)
    assert physics_batch.theta_dot.shape == (batch_size,)

    # All should be in valid range
    assert jnp.all(jnp.abs(physics_batch.x) <= 0.05)
    assert jnp.all(jnp.abs(physics_batch.x_dot) <= 0.05)
    assert jnp.all(jnp.abs(physics_batch.theta) <= 0.05)
    assert jnp.all(jnp.abs(physics_batch.theta_dot) <= 0.05)


def test_termination_return_type(task_config: TaskConfig):
    """Test that termination returns float32."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )
    t = jnp.array(0)

    done = check_termination(state, t, task_config)

    # Should be float32 for JAX compatibility
    assert done.dtype == jnp.float32
    assert done.shape == ()
