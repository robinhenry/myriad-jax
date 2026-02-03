import chex
import jax
import jax.numpy as jnp
import pytest

from myriad.core.spaces import Discrete
from myriad.envs.classic.cartpole.physics import PhysicsState
from myriad.envs.classic.cartpole.tasks.base import (
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


def test_task_config_defaults(task_config: TaskConfig):
    """Test that default task config has reasonable values."""
    assert task_config.max_steps > 0
    assert task_config.theta_threshold > 0
    assert task_config.x_threshold > 0
    # Check that theta threshold is approximately 12 degrees
    assert jnp.isclose(task_config.theta_threshold, jnp.deg2rad(12), atol=0.01)


@pytest.mark.parametrize(
    ("condition_type", "expected_done"),
    [
        ("normal", 0.0),
        ("x_high", 1.0),
        ("x_low", 1.0),
        ("theta_high", 1.0),
        ("theta_low", 1.0),
        ("max_steps", 1.0),
        ("almost_max_steps", 0.0),
    ],
)
def test_check_termination(task_config, condition_type, expected_done):
    """Test termination conditions using parametrization."""
    x, theta, t = 0.0, 0.0, 0

    if condition_type == "x_high":
        x = task_config.x_threshold * 1.01
    elif condition_type == "x_low":
        x = -task_config.x_threshold * 1.01
    elif condition_type == "theta_high":
        theta = task_config.theta_threshold * 1.01
    elif condition_type == "theta_low":
        theta = -task_config.theta_threshold * 1.01
    elif condition_type == "max_steps":
        t = task_config.max_steps
    elif condition_type == "almost_max_steps":
        t = task_config.max_steps - 1

    state = PhysicsState(
        x=jnp.array(x),
        x_dot=jnp.array(0.0),
        theta=jnp.array(theta),
        theta_dot=jnp.array(0.0),
    )
    done = check_termination(state, jnp.array(t), task_config)
    assert done == expected_done


def test_cartpole_obs():
    """Test observation extraction and shape."""
    state = PhysicsState(
        x=jnp.array(0.5),
        x_dot=jnp.array(0.3),
        theta=jnp.array(0.1),
        theta_dot=jnp.array(0.2),
    )
    obs = get_cartpole_obs(state)

    assert isinstance(obs, PhysicsState)
    chex.assert_trees_all_equal(obs, state)

    # Check shape helper
    assert get_cartpole_obs_shape() == (4,)

    # Check array conversion
    obs_array = obs.to_array()
    expected = jnp.array([0.5, 0.3, 0.1, 0.2])
    chex.assert_trees_all_close(obs_array, expected)


def test_cartpole_action_space():
    """Test action space definition and sampling."""
    action_space = get_cartpole_action_space()
    assert isinstance(action_space, Discrete)
    assert action_space.n == 2

    key = jax.random.key(0)
    action = action_space.sample(key)
    assert action in [0, 1]


def test_sample_initial_physics():
    """Test initial physics state sampling: range, randomness, repeatability."""
    key = jax.random.key(42)

    # Test Range
    physics = sample_initial_physics(key)
    assert isinstance(physics, PhysicsState)
    assert jnp.all(jnp.abs(physics.to_array()) <= 0.05)

    # Test Randomness
    p2 = sample_initial_physics(jax.random.key(43))
    assert not jnp.allclose(physics.to_array(), p2.to_array())

    # Test Repeatability
    p3 = sample_initial_physics(key)
    chex.assert_trees_all_equal(physics, p3)


def test_jax_transforms(task_config):
    """Test JIT and vmap compatibility for core functions."""
    # 1. Termination Check
    state = PhysicsState(
        x=jnp.array([0.0, 2.5]),  # One safe, one done
        x_dot=jnp.zeros(2),
        theta=jnp.zeros(2),
        theta_dot=jnp.zeros(2),
    )
    t = jnp.array([0, 0])

    # JIT check
    jitted_check = jax.jit(check_termination, static_argnames=["task_config"])
    assert (
        jitted_check(
            PhysicsState(x=jnp.array(0.0), x_dot=jnp.array(0.0), theta=jnp.array(0.0), theta_dot=jnp.array(0.0)),
            jnp.array(0),
            task_config,
        )
        == 0.0
    )

    # VMAP check
    vmap_check = jax.vmap(check_termination, in_axes=(0, 0, None))
    dones = vmap_check(state, t, task_config)
    chex.assert_trees_all_close(dones, jnp.array([0.0, 1.0]))

    # 2. Observation
    vmap_obs = jax.vmap(get_cartpole_obs)
    obs_batch = vmap_obs(state)
    assert obs_batch.x.shape == (2,)

    # 3. Sampling
    keys = jax.random.split(jax.random.key(0), 5)
    vmap_sample = jax.vmap(sample_initial_physics)
    batch_physics = vmap_sample(keys)
    assert batch_physics.x.shape == (5,)
