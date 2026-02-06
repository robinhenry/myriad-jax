"""Tests for CcaS-CcaR gene circuit physics dynamics.

This module tests the Gillespie algorithm implementation, including:
- Propensity calculation correctness
- Reaction application correctness
- Stochastic simulation stability
- JIT and vmap compatibility
"""

import jax
import jax.numpy as jnp
import pytest

from myriad.envs.bio.ccas_ccar.physics import (
    PhysicsConfig,
    PhysicsParams,
    PhysicsState,
    apply_reaction,
    compute_propensities,
    step_physics,
)


@pytest.fixture
def config():
    """Default physics configuration."""
    return PhysicsConfig()


@pytest.fixture
def params():
    """Default physics parameters."""
    return PhysicsParams()


def test_physics_config_defaults(config: PhysicsConfig):
    """Test that default physics config has biologically reasonable values."""
    assert config.eta > 0  # Production rate
    assert config.nu > 0  # Dilution rate
    assert config.a > 0  # Promoter activity
    assert config.Kh > 0  # Hill coefficient
    assert config.nh > 0  # Hill cooperativity
    assert config.Kf > 0  # Self-activation Hill coefficient
    assert config.nf > 0  # Self-activation cooperativity
    assert config.timestep_minutes > 0
    assert config.max_gillespie_steps > 0


def test_propensities_all_positive(config: PhysicsConfig):
    """Test that propensities are always non-negative."""
    state = PhysicsState(time=jnp.array(0.0), H=jnp.array(50.0), F=jnp.array(30.0))

    for action in [0, 1]:
        propensities = compute_propensities(state, jnp.array(action), config)
        assert jnp.all(propensities >= 0), f"Propensities should be non-negative, got {propensities}"
        assert propensities.shape == (5,), "Should have 5 reactions"


def test_propensities_light_dependence(config: PhysicsConfig):
    """Test that CcaSR activation depends on light input (action)."""
    state = PhysicsState(time=jnp.array(0.0), H=jnp.array(50.0), F=jnp.array(30.0))

    prop_dark = compute_propensities(state, jnp.array(0), config)
    prop_light = compute_propensities(state, jnp.array(1), config)

    # Reaction 1 (CcaSR activation) should be higher with light on
    assert prop_light[0] > prop_dark[0], "Light should increase CcaSR activation"
    # Other reactions should be unchanged
    assert jnp.allclose(prop_light[1:], prop_dark[1:])


def test_propensities_concentration_dependence(config: PhysicsConfig):
    """Test that propensities depend on protein concentrations."""
    action = jnp.array(1)

    state_low = PhysicsState(time=jnp.array(0.0), H=jnp.array(10.0), F=jnp.array(5.0))
    state_high = PhysicsState(time=jnp.array(0.0), H=jnp.array(100.0), F=jnp.array(50.0))

    prop_low = compute_propensities(state_low, action, config)
    prop_high = compute_propensities(state_high, action, config)

    # Reaction 2 (H deactivation) should scale with H
    assert prop_high[1] > prop_low[1], "H deactivation should increase with H"
    # Reaction 5 (F dilution) should scale with F
    assert prop_high[4] > prop_low[4], "F dilution should increase with F"


@pytest.mark.parametrize(
    "reaction_idx,H_delta,F_delta",
    [
        (0, +1, 0),  # CcaSR activation: ∅ → H
        (1, -1, 0),  # CcaSR deactivation: H → ∅
        (2, 0, +1),  # F creation from H: ∅ → F
        (3, 0, +1),  # F self-activation: ∅ → F
        (4, 0, -1),  # F dilution: F → ∅
    ],
)
def test_reaction_effects(reaction_idx: int, H_delta: int, F_delta: int):
    """Test that each reaction modifies state correctly."""
    state = PhysicsState(time=jnp.array(0.0), H=jnp.array(50.0), F=jnp.array(30.0))
    next_state = apply_reaction(state, jnp.array(reaction_idx))

    expected_H = state.H + H_delta
    expected_F = state.F + F_delta
    assert next_state.H == expected_H, f"Reaction {reaction_idx} should change H by {H_delta}"
    assert next_state.F == expected_F, f"Reaction {reaction_idx} should change F by {F_delta}"


def test_reactions_dont_go_negative():
    """Test that reactions cannot produce negative concentrations."""
    state = PhysicsState(time=jnp.array(0.0), H=jnp.array(0.0), F=jnp.array(0.0))

    next_state_H = apply_reaction(state, jnp.array(1))  # H deactivation
    next_state_F = apply_reaction(state, jnp.array(4))  # F dilution

    assert next_state_H.H >= 0, "H should not go negative"
    assert next_state_F.F >= 0, "F should not go negative"


def test_step_physics_advances_time(params: PhysicsParams, config: PhysicsConfig):
    """Test that physics step advances time by timestep_minutes."""
    key = jax.random.PRNGKey(0)
    state = PhysicsState(time=jnp.array(0.0), H=jnp.array(50.0), F=jnp.array(30.0))

    next_state = step_physics(key, state, jnp.array(1), params, config)

    expected_time = state.time + config.timestep_minutes
    assert jnp.allclose(next_state.time, expected_time)


def test_step_physics_light_effects(params: PhysicsParams, config: PhysicsConfig):
    """Test that light on/off affects H dynamics over multiple steps."""
    # Light on should increase H
    key = jax.random.PRNGKey(0)
    state = PhysicsState(time=jnp.array(0.0), H=jnp.array(10.0), F=jnp.array(5.0))
    for _ in range(20):
        key, subkey = jax.random.split(key)
        state = step_physics(subkey, state, jnp.array(1), params, config)
    assert state.H > 10.0, "Light on should increase H over time"

    # Light off should decrease H
    key = jax.random.PRNGKey(0)
    state = PhysicsState(time=jnp.array(0.0), H=jnp.array(100.0), F=jnp.array(50.0))
    for _ in range(50):
        key, subkey = jax.random.split(key)
        state = step_physics(subkey, state, jnp.array(0), params, config)
    assert state.H < 100.0, "Light off should decrease H over time"


def test_step_physics_produces_finite_values(params: PhysicsParams, config: PhysicsConfig):
    """Test that physics step always produces finite protein concentrations."""
    key = jax.random.PRNGKey(0)
    state = PhysicsState(time=jnp.array(0.0), H=jnp.array(50.0), F=jnp.array(30.0))

    for _ in range(50):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        action = jax.random.choice(subkey1, jnp.array([0, 1]))
        state = step_physics(subkey2, state, action, params, config)

        assert jnp.isfinite(state.time)
        assert jnp.isfinite(state.H) and state.H >= 0
        assert jnp.isfinite(state.F) and state.F >= 0


def test_physics_is_stochastic(params: PhysicsParams, config: PhysicsConfig):
    """Test that physics produces different results with different random seeds."""
    state = PhysicsState(time=jnp.array(0.0), H=jnp.array(50.0), F=jnp.array(30.0))
    action = jnp.array(1)

    # Run with different seeds for multiple steps
    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)
    state1 = state
    state2 = state

    for _ in range(10):
        key1, subkey1 = jax.random.split(key1)
        key2, subkey2 = jax.random.split(key2)
        state1 = step_physics(subkey1, state1, action, params, config)
        state2 = step_physics(subkey2, state2, action, params, config)

    # Time advances deterministically
    assert jnp.allclose(state1.time, state2.time)
    # Concentrations should diverge
    concentrations_differ = not jnp.allclose(state1.H, state2.H) or not jnp.allclose(state1.F, state2.F)
    assert concentrations_differ, "Stochastic simulation should produce different trajectories"


def test_physics_jit_compilation(params: PhysicsParams, config: PhysicsConfig):
    """Test that physics step can be JIT compiled."""
    key = jax.random.PRNGKey(0)
    state = PhysicsState(time=jnp.array(0.0), H=jnp.array(50.0), F=jnp.array(30.0))

    jitted_step = jax.jit(step_physics, static_argnames=["config"])
    next_state = jitted_step(key, state, jnp.array(1), params, config)

    assert jnp.isfinite(next_state.time)
    assert jnp.isfinite(next_state.H)
    assert jnp.isfinite(next_state.F)


def test_physics_vmap_compatibility(params: PhysicsParams, config: PhysicsConfig):
    """Test that physics step can be vectorized with vmap."""
    batch_size = 5
    states = PhysicsState(
        time=jnp.zeros(batch_size),
        H=jnp.linspace(10.0, 100.0, batch_size),
        F=jnp.full(batch_size, 30.0),
    )

    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    actions = jnp.ones(batch_size, dtype=jnp.int32)

    vmap_step = jax.vmap(step_physics, in_axes=(0, 0, 0, None, None))
    next_states = vmap_step(keys, states, actions, params, config)

    assert next_states.time.shape == (batch_size,)
    assert next_states.H.shape == (batch_size,)
    assert next_states.F.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(next_states.time))
    assert jnp.all(jnp.isfinite(next_states.H))
    assert jnp.all(jnp.isfinite(next_states.F))


@pytest.mark.parametrize(
    "param,value_low,value_high,reaction_idx,expected_higher",
    [
        ("Kh", 60.0, 120.0, 2, "low"),  # Lower Kh → higher F production from H
        ("eta", 0.5, 2.0, 0, "high"),  # Higher eta → higher H production
    ],
)
def test_parameter_sensitivity(
    param: str, value_low: float, value_high: float, reaction_idx: int, expected_higher: str
):
    """Test that changing parameters affects reaction propensities as expected."""
    state = PhysicsState(time=jnp.array(0.0), H=jnp.array(90.0), F=jnp.array(30.0))
    action = jnp.array(1)

    config_low = PhysicsConfig(**{param: value_low})
    config_high = PhysicsConfig(**{param: value_high})

    prop_low = compute_propensities(state, action, config_low)
    prop_high = compute_propensities(state, action, config_high)

    if expected_higher == "low":
        assert prop_low[reaction_idx] > prop_high[reaction_idx]
    else:
        assert prop_high[reaction_idx] > prop_low[reaction_idx]


@pytest.mark.parametrize("action", [0, 1])
def test_both_actions_produce_valid_physics(action: int, params: PhysicsParams, config: PhysicsConfig):
    """Test that both actions produce valid physics updates."""
    key = jax.random.PRNGKey(0)
    state = PhysicsState(time=jnp.array(0.0), H=jnp.array(50.0), F=jnp.array(30.0))

    next_state = step_physics(key, state, jnp.array(action), params, config)

    assert jnp.isfinite(next_state.time)
    assert jnp.isfinite(next_state.H) and next_state.H >= 0
    assert jnp.isfinite(next_state.F) and next_state.F >= 0


def test_state_to_array_from_array():
    """Test PhysicsState array conversion methods."""
    state = PhysicsState(time=jnp.array(10.0), H=jnp.array(50.0), F=jnp.array(30.0))

    arr = state.to_array()
    assert arr.shape == (3,)
    assert jnp.allclose(arr, jnp.array([10.0, 50.0, 30.0]))

    restored = PhysicsState.from_array(arr)
    assert jnp.allclose(restored.time, state.time)
    assert jnp.allclose(restored.H, state.H)
    assert jnp.allclose(restored.F, state.F)
