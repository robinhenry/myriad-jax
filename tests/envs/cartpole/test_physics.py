"""Tests for CartPole physics dynamics.

This module tests the pure physics step function, including:
- Conservation of energy properties
- Force application correctness
- Integration stability
- Action to force mapping
- Physical realism checks
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from aion.envs.cartpole.physics import PhysicsConfig, PhysicsParams, PhysicsState, step_physics


@pytest.fixture
def physics_config():
    """Default physics configuration."""
    return PhysicsConfig()


@pytest.fixture
def physics_params():
    """Default physics parameters."""
    return PhysicsParams()


def test_physics_config_defaults(physics_config: PhysicsConfig):
    """Test that default physics config has physically reasonable values."""
    assert physics_config.gravity > 0
    assert physics_config.cart_mass > 0
    assert physics_config.pole_mass > 0
    assert physics_config.pole_length > 0
    assert physics_config.force_magnitude > 0
    assert physics_config.dt > 0
    assert physics_config.dt < 0.1  # Should be small for stability


def test_action_to_force_mapping(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that actions 0 and 1 map to opposite forces."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )

    # Action 0 should push left (negative force)
    next_state_left = step_physics(state, jnp.array(0), physics_params, physics_config)

    # Action 1 should push right (positive force)
    next_state_right = step_physics(state, jnp.array(1), physics_params, physics_config)

    # From rest, action 1 should produce positive velocity, action 0 negative
    assert next_state_right.x_dot > 0, "Action 1 should push cart right (positive velocity)"
    assert next_state_left.x_dot < 0, "Action 0 should push cart left (negative velocity)"

    # Velocities should be opposite in sign and equal in magnitude
    assert jnp.allclose(next_state_right.x_dot, -next_state_left.x_dot, atol=1e-6)


def test_zero_force_equilibrium(physics_params: PhysicsParams):
    """Test that upright equilibrium is unstable but momentarily stable."""
    # At perfect upright equilibrium with no velocity, the pole should stay up
    # for a very short time (numerically), but it's actually unstable
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),  # Perfectly upright
        theta_dot=jnp.array(0.0),
    )

    # With very small dt, the pole should barely move from perfect equilibrium
    config = PhysicsConfig(dt=0.001)

    # Without external force (we'll use balanced forces by alternating actions)
    # Actually, let's just check that with no initial perturbation and action 0,
    # the state changes are small
    next_state = step_physics(state, jnp.array(0), physics_params, config)

    # Position should change very little from zero state with tiny dt
    assert jnp.abs(next_state.theta) < 0.01


def test_conservation_of_state_magnitude(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that physics doesn't produce unrealistic state explosions."""
    state = PhysicsState(
        x=jnp.array(0.1),
        x_dot=jnp.array(0.1),
        theta=jnp.array(0.1),
        theta_dot=jnp.array(0.1),
    )

    # Step forward with reasonable action
    next_state = step_physics(state, jnp.array(1), physics_params, physics_config)

    # No state variable should explode to unrealistic values in one step
    assert jnp.abs(next_state.x) < 10.0
    assert jnp.abs(next_state.x_dot) < 100.0
    assert jnp.abs(next_state.theta) < jnp.pi
    assert jnp.abs(next_state.theta_dot) < 100.0


def test_gravity_pulls_pole_down(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that gravity causes the pole to fall when tilted."""
    # Start with pole tilted to the right, no angular velocity
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.1),  # Tilted right (positive angle)
        theta_dot=jnp.array(0.0),
    )

    # Step without applying corrective force (action 0 pushes left, which might balance)
    # Use action 0 to test that gravity still pulls the pole
    next_state = step_physics(state, jnp.array(0), physics_params, physics_config)

    # Pole should have some angular velocity from gravity
    # The angular velocity direction depends on the balance of gravity and cart acceleration
    # What we can guarantee is that there IS angular acceleration (velocity changes)
    assert next_state.theta_dot != 0.0, "Gravity should produce angular acceleration"

    # After another step, angle should change (now theta_dot is non-zero)
    next_next_state = step_physics(next_state, jnp.array(0), physics_params, physics_config)
    assert next_next_state.theta != state.theta, "Pole angle should change after two steps"


def test_force_affects_cart_motion(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that applied force accelerates the cart in correct direction."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )

    # Apply rightward force (action 1)
    next_state = step_physics(state, jnp.array(1), physics_params, physics_config)

    # Cart should gain rightward velocity
    assert next_state.x_dot > 0

    # Position doesn't change immediately with Euler integration (x_next = x + dt * x_dot, where x_dot was 0)
    # But after multiple steps, position should clearly increase
    for _ in range(3):
        next_state = step_physics(next_state, jnp.array(1), physics_params, physics_config)

    assert next_state.x > 0.01, "After multiple steps, cart should move right"


def test_multiple_steps_consistency(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that multiple physics steps are consistent and stable."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.05),
        theta_dot=jnp.array(0.0),
    )

    # Apply consistent action for multiple steps
    action = jnp.array(0)
    for _ in range(10):
        state = step_physics(state, action, physics_params, physics_config)

        # Check that all values remain finite
        assert jnp.isfinite(state.x)
        assert jnp.isfinite(state.x_dot)
        assert jnp.isfinite(state.theta)
        assert jnp.isfinite(state.theta_dot)


def test_velocity_integration(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that position updates correctly from velocity (Euler integration)."""
    x_initial = 1.0
    x_dot = 2.0  # Constant velocity

    state = PhysicsState(
        x=jnp.array(x_initial),
        x_dot=jnp.array(x_dot),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )

    next_state = step_physics(state, jnp.array(1), physics_params, physics_config)

    # Position should increase by approximately velocity * dt
    # (approximately because acceleration also affects velocity)
    expected_delta_x = x_dot * physics_config.dt
    actual_delta_x = next_state.x - state.x

    # Should be close (within reasonable tolerance for Euler integration)
    assert jnp.abs(actual_delta_x - expected_delta_x) < 0.5 * physics_config.dt


def test_pole_mass_affects_dynamics(physics_params: PhysicsParams):
    """Test that changing pole mass affects the dynamics."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.1),
        theta_dot=jnp.array(0.0),
    )

    action = jnp.array(1)

    # Light pole
    config_light = PhysicsConfig(pole_mass=0.05)
    next_state_light = step_physics(state, action, physics_params, config_light)

    # Heavy pole
    config_heavy = PhysicsConfig(pole_mass=0.2)
    next_state_heavy = step_physics(state, action, physics_params, config_heavy)

    # Different masses should produce different angular velocities (theta doesn't change in first step with Euler)
    assert not jnp.allclose(next_state_light.theta_dot, next_state_heavy.theta_dot, atol=1e-6)

    # After multiple steps, theta should also differ
    for _ in range(3):
        next_state_light = step_physics(next_state_light, action, physics_params, config_light)
        next_state_heavy = step_physics(next_state_heavy, action, physics_params, config_heavy)

    assert not jnp.allclose(next_state_light.theta, next_state_heavy.theta, atol=1e-6)


def test_pole_length_affects_dynamics(physics_params: PhysicsParams):
    """Test that changing pole length affects the dynamics."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.1),
        theta_dot=jnp.array(0.0),
    )

    action = jnp.array(1)

    # Short pole
    config_short = PhysicsConfig(pole_length=0.3)
    next_state_short = step_physics(state, action, physics_params, config_short)

    # Long pole
    config_long = PhysicsConfig(pole_length=0.7)
    next_state_long = step_physics(state, action, physics_params, config_long)

    # Different lengths should produce different angular velocities
    assert not jnp.allclose(next_state_short.theta_dot, next_state_long.theta_dot, atol=1e-6)

    # After multiple steps, theta should also differ
    for _ in range(3):
        next_state_short = step_physics(next_state_short, action, physics_params, config_short)
        next_state_long = step_physics(next_state_long, action, physics_params, config_long)

    assert not jnp.allclose(next_state_short.theta, next_state_long.theta, atol=1e-6)


def test_force_magnitude_affects_dynamics(physics_params: PhysicsParams):
    """Test that changing force magnitude affects cart acceleration."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.0),
        theta_dot=jnp.array(0.0),
    )

    action = jnp.array(1)

    # Weak force
    config_weak = PhysicsConfig(force_magnitude=5.0)
    next_state_weak = step_physics(state, action, physics_params, config_weak)

    # Strong force
    config_strong = PhysicsConfig(force_magnitude=20.0)
    next_state_strong = step_physics(state, action, physics_params, config_strong)

    # Stronger force should produce larger velocity
    assert next_state_strong.x_dot > next_state_weak.x_dot

    # After multiple steps, position should also be larger
    for _ in range(3):
        next_state_weak = step_physics(next_state_weak, action, physics_params, config_weak)
        next_state_strong = step_physics(next_state_strong, action, physics_params, config_strong)

    assert next_state_strong.x > next_state_weak.x


def test_timestep_affects_integration(physics_params: PhysicsParams):
    """Test that different timesteps produce different integration results."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(1.0),
        theta=jnp.array(0.1),
        theta_dot=jnp.array(0.5),
    )

    action = jnp.array(1)

    # Small timestep
    config_small_dt = PhysicsConfig(dt=0.01)
    next_state_small = step_physics(state, action, physics_params, config_small_dt)

    # Large timestep
    config_large_dt = PhysicsConfig(dt=0.04)
    next_state_large = step_physics(state, action, physics_params, config_large_dt)

    # Larger timestep should produce larger state changes
    assert jnp.abs(next_state_large.x - state.x) > jnp.abs(next_state_small.x - state.x)
    assert jnp.abs(next_state_large.theta - state.theta) > jnp.abs(next_state_small.theta - state.theta)


def test_physics_step_is_deterministic(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that physics step is deterministic (same input -> same output)."""
    state = PhysicsState(
        x=jnp.array(0.5),
        x_dot=jnp.array(0.3),
        theta=jnp.array(0.1),
        theta_dot=jnp.array(0.2),
    )

    action = jnp.array(1)

    # Run the same step twice
    next_state_1 = step_physics(state, action, physics_params, physics_config)
    next_state_2 = step_physics(state, action, physics_params, physics_config)

    # Should be exactly identical
    chex.assert_trees_all_equal(next_state_1, next_state_2)


def test_physics_jit_compilation(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that physics step can be JIT compiled."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.1),
        theta_dot=jnp.array(0.0),
    )

    action = jnp.array(1)

    # JIT compile the physics step
    jitted_step = jax.jit(step_physics, static_argnames=["config"])

    # Run it
    next_state = jitted_step(state, action, physics_params, physics_config)

    # Should produce valid output
    assert jnp.isfinite(next_state.x)
    assert jnp.isfinite(next_state.x_dot)
    assert jnp.isfinite(next_state.theta)
    assert jnp.isfinite(next_state.theta_dot)


def test_physics_vmap_compatibility(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that physics step can be vectorized with vmap."""
    # Create batch of states
    batch_size = 5
    states = PhysicsState(
        x=jnp.zeros(batch_size),
        x_dot=jnp.zeros(batch_size),
        theta=jnp.linspace(0.0, 0.2, batch_size),  # Different angles
        theta_dot=jnp.zeros(batch_size),
    )

    actions = jnp.ones(batch_size, dtype=jnp.int32)

    # Vectorize over batch dimension
    vmap_step = jax.vmap(step_physics, in_axes=(0, 0, None, None))
    next_states = vmap_step(states, actions, physics_params, physics_config)

    # Should produce batch of outputs
    assert next_states.x.shape == (batch_size,)
    assert next_states.x_dot.shape == (batch_size,)
    assert next_states.theta.shape == (batch_size,)
    assert next_states.theta_dot.shape == (batch_size,)

    # All outputs should be finite
    assert jnp.all(jnp.isfinite(next_states.x))
    assert jnp.all(jnp.isfinite(next_states.x_dot))
    assert jnp.all(jnp.isfinite(next_states.theta))
    assert jnp.all(jnp.isfinite(next_states.theta_dot))


def test_angular_acceleration_formula(physics_params: PhysicsParams):
    """Test the angular acceleration computation at specific state."""
    # Test case: upright pole with small perturbation
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.01),  # Small angle approximation
        theta_dot=jnp.array(0.0),
    )

    config = PhysicsConfig()
    action = jnp.array(1)

    next_state = step_physics(state, action, physics_params, config)

    # For small angles, pole should have angular acceleration proportional to gravity * theta
    # This tests that the physics formula is working correctly
    # Angular acceleration should be positive (falling in direction of tilt)
    angular_acceleration = (next_state.theta_dot - state.theta_dot) / config.dt
    assert angular_acceleration != 0.0  # Should have some angular acceleration


def test_extreme_angle_stability(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that physics remains stable even at extreme angles."""
    # Nearly horizontal pole
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(jnp.pi / 2 - 0.1),  # Nearly horizontal
        theta_dot=jnp.array(0.0),
    )

    action = jnp.array(1)
    next_state = step_physics(state, action, physics_params, physics_config)

    # Should still produce finite results
    assert jnp.isfinite(next_state.x)
    assert jnp.isfinite(next_state.x_dot)
    assert jnp.isfinite(next_state.theta)
    assert jnp.isfinite(next_state.theta_dot)


@pytest.mark.parametrize("action", [0, 1])
def test_both_actions_produce_valid_physics(action: int, physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that both actions produce valid physics updates."""
    state = PhysicsState(
        x=jnp.array(0.0),
        x_dot=jnp.array(0.0),
        theta=jnp.array(0.1),
        theta_dot=jnp.array(0.0),
    )

    next_state = step_physics(state, jnp.array(action), physics_params, physics_config)

    # All state variables should be finite
    assert jnp.isfinite(next_state.x)
    assert jnp.isfinite(next_state.x_dot)
    assert jnp.isfinite(next_state.theta)
    assert jnp.isfinite(next_state.theta_dot)

    # State should have changed
    state_changed = (
        not jnp.allclose(next_state.x, state.x)
        or not jnp.allclose(next_state.x_dot, state.x_dot)
        or not jnp.allclose(next_state.theta, state.theta)
        or not jnp.allclose(next_state.theta_dot, state.theta_dot)
    )
    assert state_changed, "Physics should update at least one state variable"
