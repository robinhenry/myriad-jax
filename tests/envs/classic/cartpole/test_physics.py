"""Tests for CartPole physics dynamics."""

import chex
import jax
import jax.numpy as jnp
import pytest

from myriad.envs.classic.cartpole.physics import PhysicsConfig, PhysicsParams, PhysicsState, step_physics


@pytest.fixture
def config() -> PhysicsConfig:
    return PhysicsConfig()


@pytest.fixture
def params() -> PhysicsParams:
    return PhysicsParams()


def test_action_to_force_mapping(params: PhysicsParams, config: PhysicsConfig):
    """Actions 0/1 produce symmetric opposite forces."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )

    left = step_physics(state, jnp.array(0), params, config)
    right = step_physics(state, jnp.array(1), params, config)

    # From rest at origin, velocities should be exactly opposite
    chex.assert_trees_all_close(left.x_dot, -right.x_dot)
    chex.assert_trees_all_close(left.theta_dot, -right.theta_dot)

    # Positions unchanged (Euler: x_next = x + dt * x_dot, where x_dot was 0)
    chex.assert_trees_all_close(left.x, 0.0)
    chex.assert_trees_all_close(right.x, 0.0)


def test_step_from_rest(params: PhysicsParams, config: PhysicsConfig):
    """Single step from rest produces expected values."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )

    next_state = step_physics(state, jnp.array(1), params, config)

    # Pre-computed values for default config (g=9.8, mc=1.0, mp=0.1, l=0.5, F=10, dt=0.02)
    chex.assert_trees_all_close(next_state.x, 0.0)
    chex.assert_trees_all_close(next_state.x_dot, 0.19512196, rtol=1e-5)
    chex.assert_trees_all_close(next_state.theta, 0.0)
    chex.assert_trees_all_close(next_state.theta_dot, -0.29268292, rtol=1e-5)


def test_step_with_initial_tilt(params: PhysicsParams, config: PhysicsConfig):
    """Step from tilted position shows gravity effect."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.1),  # Tilted right
        theta_dot=jnp.array(0.0),
    )

    next_state = step_physics(state, jnp.array(1), params, config)

    # Pre-computed: gravity accelerates fall, force counteracts
    chex.assert_trees_all_close(next_state.x, 0.0)
    chex.assert_trees_all_close(next_state.x_dot, 0.193556, rtol=1e-5)
    chex.assert_trees_all_close(next_state.theta, 0.1)
    chex.assert_trees_all_close(next_state.theta_dot, -0.259533, rtol=1e-5)


def test_step_with_velocity(params: PhysicsParams, config: PhysicsConfig):
    """Step with initial velocity integrates position correctly."""
    state = PhysicsState(
        x=jnp.array(1.0),
        x_dot=jnp.array(2.0),
        theta=jnp.array(0.05),
        theta_dot=jnp.array(-0.5),
    )

    next_state = step_physics(state, jnp.array(0), params, config)

    # Euler: x_next = x + dt * x_dot = 1.0 + 0.02 * 2.0 = 1.04
    chex.assert_trees_all_close(next_state.x, 1.04)
    # Euler: theta_next = theta + dt * theta_dot = 0.05 + 0.02 * (-0.5) = 0.04
    chex.assert_trees_all_close(next_state.theta, 0.04)


@pytest.mark.parametrize(
    ("pole_mass", "expected_theta_dot"),
    [
        (0.05, -0.296296),  # Light pole
        (0.2, -0.285714),  # Heavy pole
    ],
)
def test_pole_mass_affects_dynamics(pole_mass: float, expected_theta_dot: float, params: PhysicsParams):
    """Different pole masses produce different angular accelerations."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )

    config = PhysicsConfig(pole_mass=pole_mass)
    next_state = step_physics(state, jnp.array(1), params, config)

    chex.assert_trees_all_close(next_state.theta_dot, expected_theta_dot, rtol=1e-5)


def test_jax_transforms(params: PhysicsParams, config: PhysicsConfig):
    """JIT compilation and vmap work correctly."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.1),
        theta_dot=jnp.array(0.0),
    )

    # JIT
    jitted_step = jax.jit(step_physics, static_argnames=["config"])
    jit_result = jitted_step(state, jnp.array(1), params, config)
    eager_result = step_physics(state, jnp.array(1), params, config)
    chex.assert_trees_all_close(jit_result, eager_result)

    # Vmap over batch of states
    batch_states = PhysicsState(
        x=jnp.zeros(3),
        x_dot=jnp.zeros(3),
        theta=jnp.array([0.0, 0.1, 0.2]),
        theta_dot=jnp.zeros(3),
    )
    actions = jnp.ones(3, dtype=jnp.int32)

    vmap_step = jax.vmap(step_physics, in_axes=(0, 0, None, None))
    batch_result = vmap_step(batch_states, actions, params, config)

    assert batch_result.x.shape == (3,)
    assert batch_result.theta.shape == (3,)
