import chex
import jax
import jax.numpy as jnp
import pytest

from myriad.core.spaces import Box
from myriad.envs.classic.pendulum.physics import PhysicsConfig, PhysicsState
from myriad.envs.classic.pendulum.tasks.base import (
    PendulumObservation,
    TaskConfig,
    get_pendulum_action_space,
    get_pendulum_obs,
    get_pendulum_obs_shape,
    sample_initial_physics,
)


@pytest.fixture
def task_config():
    """Default task configuration."""
    return TaskConfig()


@pytest.fixture
def physics_config():
    """Default physics configuration."""
    return PhysicsConfig()


def test_task_config_defaults(task_config: TaskConfig):
    """Test that default task config has reasonable values."""
    assert task_config.max_steps > 0
    assert task_config.max_steps == 200


def test_pendulum_observation():
    """Test observation extraction and shape."""
    state = PhysicsState(
        theta=jnp.array(0.5),
        theta_dot=jnp.array(0.2),
    )
    obs = get_pendulum_obs(state)

    assert isinstance(obs, PendulumObservation)
    chex.assert_trees_all_close(obs.cos_theta, jnp.cos(0.5))
    chex.assert_trees_all_close(obs.sin_theta, jnp.sin(0.5))
    chex.assert_trees_all_close(obs.theta_dot, 0.2)

    # Check shape helper
    assert get_pendulum_obs_shape() == (3,)

    # Check array conversion
    obs_array = obs.to_array()
    expected = jnp.array([jnp.cos(0.5), jnp.sin(0.5), 0.2])
    chex.assert_trees_all_close(obs_array, expected)


def test_observation_bounds():
    """Test that observation values are bounded."""
    # Test at various angles
    for theta in [0.0, jnp.pi / 2, jnp.pi, -jnp.pi / 2]:
        state = PhysicsState(theta=jnp.array(theta), theta_dot=jnp.array(0.0))
        obs = get_pendulum_obs(state)

        # cos and sin should be in [-1, 1]
        assert -1 <= obs.cos_theta <= 1
        assert -1 <= obs.sin_theta <= 1


def test_pendulum_action_space(physics_config: PhysicsConfig):
    """Test action space definition and sampling."""
    action_space = get_pendulum_action_space(physics_config)
    assert isinstance(action_space, Box)
    assert action_space.shape == (1,)
    assert action_space.low == -physics_config.max_torque
    assert action_space.high == physics_config.max_torque

    key = jax.random.key(0)
    action = action_space.sample(key)
    assert action.shape == (1,)
    assert action_space.contains(action)


def test_sample_initial_physics():
    """Test initial physics state sampling: range, randomness, repeatability."""
    key = jax.random.key(42)

    # Test Range
    physics = sample_initial_physics(key)
    assert isinstance(physics, PhysicsState)
    assert -jnp.pi <= physics.theta <= jnp.pi
    assert -1.0 <= physics.theta_dot <= 1.0

    # Test Randomness
    p2 = sample_initial_physics(jax.random.key(43))
    assert not jnp.allclose(physics.to_array(), p2.to_array())

    # Test Repeatability
    p3 = sample_initial_physics(key)
    chex.assert_trees_all_equal(physics, p3)


def test_jax_transforms(physics_config: PhysicsConfig):
    """Test JIT and vmap compatibility for core functions."""
    # 1. Observation
    state = PhysicsState(
        theta=jnp.array([0.0, 0.5]),
        theta_dot=jnp.zeros(2),
    )

    vmap_obs = jax.vmap(get_pendulum_obs)
    obs_batch = vmap_obs(state)
    assert obs_batch.cos_theta.shape == (2,)

    # 2. Sampling
    keys = jax.random.split(jax.random.key(0), 5)
    vmap_sample = jax.vmap(sample_initial_physics)
    batch_physics = vmap_sample(keys)
    assert batch_physics.theta.shape == (5,)

    # 3. Action space - just verify it works (not jittable since Box is not a JAX type)
    action_space = get_pendulum_action_space(physics_config)
    assert isinstance(action_space, Box)
