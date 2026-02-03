"""Tests for CcaS-CcaR gene circuit physics dynamics.

This module tests the Gillespie algorithm implementation, including:
- Propensity calculation correctness
- Reaction application correctness
- Stochastic simulation stability
- Action to protein production mapping
- Parameter sensitivity
"""

import jax
import jax.numpy as jnp
import pytest

from myriad.envs.ccas_ccar.physics import (
    PhysicsConfig,
    PhysicsParams,
    PhysicsState,
    apply_reaction,
    compute_propensities,
    step_physics,
)


@pytest.fixture
def physics_config():
    """Default physics configuration."""
    return PhysicsConfig()


@pytest.fixture
def physics_params():
    """Default physics parameters."""
    return PhysicsParams()


def test_physics_config_defaults(physics_config: PhysicsConfig):
    """Test that default physics config has biologically reasonable values."""
    assert physics_config.eta > 0  # Production rate
    assert physics_config.nu > 0  # Dilution rate
    assert physics_config.a > 0  # Promoter activity
    assert physics_config.Kh > 0  # Hill coefficient
    assert physics_config.nh > 0  # Hill cooperativity
    assert physics_config.Kf > 0  # Self-activation Hill coefficient
    assert physics_config.nf > 0  # Self-activation cooperativity
    assert physics_config.timestep_minutes > 0
    assert physics_config.max_gillespie_steps > 0


def test_propensities_all_positive(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that propensities are always non-negative."""
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(50.0),
        F=jnp.array(30.0),
    )

    for action in [0, 1]:
        propensities = compute_propensities(state, jnp.array(action), physics_config)

        # All propensities should be non-negative
        assert jnp.all(propensities >= 0), f"Propensities should be non-negative, got {propensities}"
        assert propensities.shape == (5,), "Should have 5 reactions"


def test_propensities_light_dependence(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that CcaSR activation depends on light input (action)."""
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(50.0),
        F=jnp.array(30.0),
    )

    # Light off (action=0)
    prop_dark = compute_propensities(state, jnp.array(0), physics_config)

    # Light on (action=1)
    prop_light = compute_propensities(state, jnp.array(1), physics_config)

    # Reaction 1 (CcaSR activation) should be higher with light on
    assert prop_light[0] > prop_dark[0], "Light should increase CcaSR activation"

    # Other reactions should be unchanged (independent of light)
    assert jnp.allclose(prop_light[1:], prop_dark[1:])


def test_propensities_concentration_dependence(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that propensities depend on protein concentrations."""
    action = jnp.array(1)

    # Low concentrations
    state_low = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(10.0),
        F=jnp.array(5.0),
    )

    # High concentrations
    state_high = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(100.0),
        F=jnp.array(50.0),
    )

    prop_low = compute_propensities(state_low, action, physics_config)
    prop_high = compute_propensities(state_high, action, physics_config)

    # Reaction 2 (H deactivation) should scale with H
    assert prop_high[1] > prop_low[1], "H deactivation should increase with H"

    # Reaction 5 (F dilution) should scale with F
    assert prop_high[4] > prop_low[4], "F dilution should increase with F"


def test_reaction_0_increases_H(physics_params: PhysicsParams):
    """Test that reaction 0 (CcaSR activation) increases H."""
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(50.0),
        F=jnp.array(30.0),
    )

    next_state = apply_reaction(state, jnp.array(0))

    assert next_state.H == state.H + 1, "Reaction 0 should add one H molecule"
    assert next_state.F == state.F, "Reaction 0 should not change F"


def test_reaction_1_decreases_H(physics_params: PhysicsParams):
    """Test that reaction 1 (CcaSR deactivation) decreases H."""
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(50.0),
        F=jnp.array(30.0),
    )

    next_state = apply_reaction(state, jnp.array(1))

    assert next_state.H == state.H - 1, "Reaction 1 should remove one H molecule"
    assert next_state.F == state.F, "Reaction 1 should not change F"


def test_reaction_2_increases_F(physics_params: PhysicsParams):
    """Test that reaction 2 (F creation from H) increases F."""
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(50.0),
        F=jnp.array(30.0),
    )

    next_state = apply_reaction(state, jnp.array(2))

    assert next_state.F == state.F + 1, "Reaction 2 should add one F molecule"
    assert next_state.H == state.H, "Reaction 2 should not change H"


def test_reaction_3_increases_F(physics_params: PhysicsParams):
    """Test that reaction 3 (F self-activation) increases F."""
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(50.0),
        F=jnp.array(30.0),
    )

    next_state = apply_reaction(state, jnp.array(3))

    assert next_state.F == state.F + 1, "Reaction 3 should add one F molecule"
    assert next_state.H == state.H, "Reaction 3 should not change H"


def test_reaction_4_decreases_F(physics_params: PhysicsParams):
    """Test that reaction 4 (F dilution) decreases F."""
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(50.0),
        F=jnp.array(30.0),
    )

    next_state = apply_reaction(state, jnp.array(4))

    assert next_state.F == state.F - 1, "Reaction 4 should remove one F molecule"
    assert next_state.H == state.H, "Reaction 4 should not change H"


def test_reactions_dont_go_negative(physics_params: PhysicsParams):
    """Test that reactions cannot produce negative concentrations."""
    # Start with zero concentrations
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(0.0),
        F=jnp.array(0.0),
    )

    # Apply reactions that would decrease concentrations
    next_state_H = apply_reaction(state, jnp.array(1))  # H deactivation
    next_state_F = apply_reaction(state, jnp.array(4))  # F dilution

    assert next_state_H.H >= 0, "H should not go negative"
    assert next_state_F.F >= 0, "F should not go negative"


def test_step_physics_advances_time(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that physics step advances time by timestep_minutes."""
    key = jax.random.PRNGKey(0)
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(50.0),
        F=jnp.array(30.0),
    )

    next_state = step_physics(key, state, jnp.array(1), physics_params, physics_config)

    expected_time = state.time + physics_config.timestep_minutes
    assert jnp.allclose(next_state.time, expected_time), f"Time should advance by {physics_config.timestep_minutes}"


def test_step_physics_light_on_increases_H(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that light on (action=1) tends to increase H over time."""
    key = jax.random.PRNGKey(0)
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(10.0),
        F=jnp.array(5.0),
    )

    # Run multiple steps with light on
    action = jnp.array(1)
    for i in range(20):
        key, subkey = jax.random.split(key)
        state = step_physics(subkey, state, action, physics_params, physics_config)

    # H should have increased significantly with light on
    # (stochastic, but with high probability)
    assert state.H > 10.0, "Light on should increase H over time"


def test_step_physics_light_off_decreases_H(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that light off (action=0) tends to decrease H over time."""
    key = jax.random.PRNGKey(0)
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(100.0),
        F=jnp.array(50.0),
    )

    # Run multiple steps with light off
    action = jnp.array(0)
    for i in range(50):
        key, subkey = jax.random.split(key)
        state = step_physics(subkey, state, action, physics_params, physics_config)

    # H should have decreased with light off
    # (stochastic, but with high probability)
    assert state.H < 100.0, "Light off should decrease H over time"


def test_step_physics_produces_finite_values(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that physics step always produces finite protein concentrations."""
    key = jax.random.PRNGKey(0)
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(50.0),
        F=jnp.array(30.0),
    )

    # Run multiple steps with random actions
    for i in range(50):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        action = jax.random.choice(subkey1, jnp.array([0, 1]))
        state = step_physics(subkey2, state, action, physics_params, physics_config)

        # All values should remain finite and non-negative
        assert jnp.isfinite(state.time)
        assert jnp.isfinite(state.H)
        assert jnp.isfinite(state.F)
        assert state.H >= 0
        assert state.F >= 0


def test_physics_is_stochastic(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that physics produces different results with different random seeds."""
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(50.0),
        F=jnp.array(30.0),
    )

    action = jnp.array(1)

    # Run with different seeds
    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)

    next_state_1 = step_physics(key1, state, action, physics_params, physics_config)
    next_state_2 = step_physics(key2, state, action, physics_params, physics_config)

    # Results should likely be different (stochastic)
    # Time should be the same (deterministic advancement)
    assert jnp.allclose(next_state_1.time, next_state_2.time)

    # Concentrations might differ (run multiple steps to ensure difference)
    state1, state2 = next_state_1, next_state_2
    for i in range(10):
        key1, subkey1 = jax.random.split(key1)
        key2, subkey2 = jax.random.split(key2)
        state1 = step_physics(subkey1, state1, action, physics_params, physics_config)
        state2 = step_physics(subkey2, state2, action, physics_params, physics_config)

    # After multiple steps, trajectories should diverge
    concentrations_differ = not jnp.allclose(state1.H, state2.H) or not jnp.allclose(state1.F, state2.F)
    assert concentrations_differ, "Stochastic simulation should produce different trajectories with different seeds"


def test_physics_jit_compilation(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that physics step can be JIT compiled."""
    key = jax.random.PRNGKey(0)
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(50.0),
        F=jnp.array(30.0),
    )

    action = jnp.array(1)

    # JIT compile the physics step
    jitted_step = jax.jit(step_physics, static_argnames=["config"])

    # Run it
    next_state = jitted_step(key, state, action, physics_params, physics_config)

    # Should produce valid output
    assert jnp.isfinite(next_state.time)
    assert jnp.isfinite(next_state.H)
    assert jnp.isfinite(next_state.F)


def test_physics_vmap_compatibility(physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that physics step can be vectorized with vmap."""
    # Create batch of states
    batch_size = 5
    states = PhysicsState(
        time=jnp.zeros(batch_size),
        H=jnp.linspace(10.0, 100.0, batch_size),  # Different H levels
        F=jnp.full(batch_size, 30.0),
    )

    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    actions = jnp.ones(batch_size, dtype=jnp.int32)

    # Vectorize over batch dimension
    vmap_step = jax.vmap(step_physics, in_axes=(0, 0, 0, None, None))
    next_states = vmap_step(keys, states, actions, physics_params, physics_config)

    # Should produce batch of outputs
    assert next_states.time.shape == (batch_size,)
    assert next_states.H.shape == (batch_size,)
    assert next_states.F.shape == (batch_size,)

    # All outputs should be finite
    assert jnp.all(jnp.isfinite(next_states.time))
    assert jnp.all(jnp.isfinite(next_states.H))
    assert jnp.all(jnp.isfinite(next_states.F))


def test_parameter_sensitivity_Kh(physics_params: PhysicsParams):
    """Test that changing Kh affects F production dynamics."""
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(90.0),  # At nominal Kh value
        F=jnp.array(10.0),
    )

    action = jnp.array(1)

    # Low Kh (easier to activate F production)
    config_low = PhysicsConfig(Kh=60.0)
    prop_low = compute_propensities(state, action, config_low)

    # High Kh (harder to activate F production)
    config_high = PhysicsConfig(Kh=120.0)
    prop_high = compute_propensities(state, action, config_high)

    # Reaction 3 (F production from H) should be higher with lower Kh
    assert prop_low[2] > prop_high[2], "Lower Kh should increase F production from H"


def test_parameter_sensitivity_eta(physics_params: PhysicsParams):
    """Test that changing eta affects H production."""
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(50.0),
        F=jnp.array(30.0),
    )

    action = jnp.array(1)

    # Low eta (slow H production)
    config_low = PhysicsConfig(eta=0.5)
    prop_low = compute_propensities(state, action, config_low)

    # High eta (fast H production)
    config_high = PhysicsConfig(eta=2.0)
    prop_high = compute_propensities(state, action, config_high)

    # Reaction 1 (H activation) should scale with eta
    assert prop_high[0] > prop_low[0], "Higher eta should increase H production rate"
    assert jnp.allclose(prop_high[0] / prop_low[0], 2.0 / 0.5), "H production should scale linearly with eta"


@pytest.mark.parametrize("action", [0, 1])
def test_both_actions_produce_valid_physics(action: int, physics_params: PhysicsParams, physics_config: PhysicsConfig):
    """Test that both actions produce valid physics updates."""
    key = jax.random.PRNGKey(0)
    state = PhysicsState(
        time=jnp.array(0.0),
        H=jnp.array(50.0),
        F=jnp.array(30.0),
    )

    next_state = step_physics(key, state, jnp.array(action), physics_params, physics_config)

    # All state variables should be finite and non-negative
    assert jnp.isfinite(next_state.time)
    assert jnp.isfinite(next_state.H)
    assert jnp.isfinite(next_state.F)
    assert next_state.H >= 0
    assert next_state.F >= 0
