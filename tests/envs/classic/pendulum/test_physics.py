import chex
import jax
import jax.numpy as jnp
import pytest

from myriad.envs.classic.pendulum.physics import PhysicsConfig, PhysicsParams, PhysicsState, step_physics


@pytest.fixture
def config() -> PhysicsConfig:
    return PhysicsConfig()


@pytest.fixture
def params() -> PhysicsParams:
    return PhysicsParams()


def test_torque_direction(params: PhysicsParams, config: PhysicsConfig):
    """Positive and negative torques produce opposite angular accelerations."""
    state = PhysicsState(
        theta=jnp.array(0.0),  # Hanging down
        theta_dot=jnp.array(0.0),
    )

    pos_torque = step_physics(state, jnp.array(1.0), params, config)
    neg_torque = step_physics(state, jnp.array(-1.0), params, config)

    # Velocities should be opposite
    chex.assert_trees_all_close(pos_torque.theta_dot, -neg_torque.theta_dot)


def test_step_from_hanging(params: PhysicsParams, config: PhysicsConfig):
    """Step from hanging position with zero torque stays at equilibrium."""
    state = PhysicsState(
        theta=jnp.array(0.0),  # Hanging down (stable equilibrium)
        theta_dot=jnp.array(0.0),
    )

    next_state = step_physics(state, jnp.array(0.0), params, config)

    # Should stay at equilibrium (no gravity torque at theta=0)
    chex.assert_trees_all_close(next_state.theta, 0.0, atol=1e-6)
    chex.assert_trees_all_close(next_state.theta_dot, 0.0, atol=1e-6)


def test_gravity_effect(params: PhysicsParams, config: PhysicsConfig):
    """Gravity accelerates fall from tilted position."""
    state = PhysicsState(
        theta=jnp.array(0.5),  # Tilted from hanging
        theta_dot=jnp.array(0.0),
    )

    next_state = step_physics(state, jnp.array(0.0), params, config)

    # Gravity should create angular acceleration (theta_dot should increase)
    # sin(0.5) > 0, so theta_ddot > 0, so theta_dot should increase
    assert next_state.theta_dot > 0


def test_clipping(params: PhysicsParams, config: PhysicsConfig):
    """Torque and velocity are clipped to configured limits."""
    state = PhysicsState(theta=jnp.array(0.0), theta_dot=jnp.array(0.0))

    # Torque clipping: large torque should equal max torque
    large_torque = step_physics(state, jnp.array(10.0), params, config)
    max_torque = step_physics(state, jnp.array(config.max_torque), params, config)
    chex.assert_trees_all_close(large_torque.theta_dot, max_torque.theta_dot)

    # Velocity clipping: can't exceed max_speed
    fast_state = PhysicsState(theta=jnp.array(0.0), theta_dot=jnp.array(config.max_speed - 0.1))
    next_state = step_physics(fast_state, jnp.array(config.max_torque), params, config)
    assert next_state.theta_dot <= config.max_speed


def test_step_with_velocity(params: PhysicsParams, config: PhysicsConfig):
    """Step with initial velocity integrates position correctly."""
    state = PhysicsState(
        theta=jnp.array(1.0),
        theta_dot=jnp.array(2.0),
    )

    next_state = step_physics(state, jnp.array(0.0), params, config)

    # Position should change due to velocity
    # Semi-implicit Euler: theta_next = theta + dt * theta_dot_next
    assert next_state.theta != state.theta


@pytest.mark.parametrize(
    ("mass", "expected_slower"),
    [
        (0.5, True),  # Lighter mass -> faster response
        (2.0, False),  # Heavier mass -> slower response (relative to m=1)
    ],
)
def test_mass_affects_dynamics(mass: float, expected_slower: bool, params: PhysicsParams):
    """Different masses produce different angular accelerations from torque."""
    state = PhysicsState(
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )

    config_default = PhysicsConfig()
    config_custom = PhysicsConfig(mass=mass)

    default_result = step_physics(state, jnp.array(1.0), params, config_default)
    custom_result = step_physics(state, jnp.array(1.0), params, config_custom)

    if expected_slower:
        # Lighter mass should have larger theta_dot
        assert jnp.abs(custom_result.theta_dot) > jnp.abs(default_result.theta_dot)
    else:
        # Heavier mass should have smaller theta_dot
        assert jnp.abs(custom_result.theta_dot) < jnp.abs(default_result.theta_dot)


def test_jax_transforms(params: PhysicsParams, config: PhysicsConfig):
    """JIT compilation and vmap work correctly."""
    state = PhysicsState(
        theta=jnp.array(0.5),
        theta_dot=jnp.array(0.0),
    )

    # JIT
    jitted_step = jax.jit(step_physics, static_argnames=["config"])
    jit_result = jitted_step(state, jnp.array(1.0), params, config)
    eager_result = step_physics(state, jnp.array(1.0), params, config)
    chex.assert_trees_all_close(jit_result, eager_result)

    # Vmap over batch of states
    batch_states = PhysicsState(
        theta=jnp.array([0.0, 0.5, 1.0]),
        theta_dot=jnp.zeros(3),
    )
    actions = jnp.ones(3)

    vmap_step = jax.vmap(step_physics, in_axes=(0, 0, None, None))
    batch_result = vmap_step(batch_states, actions, params, config)

    assert batch_result.theta.shape == (3,)
    assert batch_result.theta_dot.shape == (3,)
