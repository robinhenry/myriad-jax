"""Tests for opto_hill_1d physics dynamics.

Covers propensities, reaction application, Gillespie stability,
JIT/vmap compatibility, and steady-state behavior.
"""

import jax
import jax.numpy as jnp
import pytest

from myriad.envs.bio.opto_hill_1d.physics import (
    PhysicsConfig,
    PhysicsParams,
    PhysicsState,
    apply_reaction,
    compute_propensities,
    step_physics,
)


@pytest.fixture
def config():
    return PhysicsConfig()


@pytest.fixture
def params():
    return PhysicsParams()


def test_physics_config_defaults(config: PhysicsConfig):
    assert config.timestep_minutes > 0
    assert config.max_gillespie_steps > 0


def test_physics_params_defaults(params: PhysicsParams):
    assert params.k_prod > 0
    assert params.K > 0
    assert params.n > 0
    assert params.k_deg > 0


def test_propensities_non_negative(params: PhysicsParams):
    state = PhysicsState.create(time=jnp.array(0.0), X=jnp.array(50.0))
    for U in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p = compute_propensities(state, jnp.array(U, dtype=jnp.float32), params)
        assert p.shape == (2,)
        assert jnp.all(p >= 0), f"propensities must be >= 0, got {p} at U={U}"


def test_propensity_light_dependence(params: PhysicsParams):
    """r_prod monotonically increasing in U; r_deg independent of U."""
    state = PhysicsState.create(time=jnp.array(0.0), X=jnp.array(50.0))

    p_dark = compute_propensities(state, jnp.array(0.0, dtype=jnp.float32), params)
    p_half = compute_propensities(state, jnp.array(0.5, dtype=jnp.float32), params)
    p_full = compute_propensities(state, jnp.array(1.0, dtype=jnp.float32), params)

    assert p_dark[0] == 0.0, "Hill(0, K, n) must be exactly 0"
    assert p_half[0] > p_dark[0]
    assert p_full[0] > p_half[0]

    # Degradation propensity unchanged by U
    assert jnp.allclose(p_dark[1], p_half[1])
    assert jnp.allclose(p_half[1], p_full[1])


def test_propensity_degradation_is_linear_in_x(params: PhysicsParams):
    """r_deg = k_deg * X (strictly linear)."""
    U = jnp.array(0.5, dtype=jnp.float32)
    r_deg_10 = compute_propensities(PhysicsState.create(time=jnp.array(0.0), X=jnp.array(10.0)), U, params)[1]
    r_deg_100 = compute_propensities(PhysicsState.create(time=jnp.array(0.0), X=jnp.array(100.0)), U, params)[1]
    assert jnp.isclose(r_deg_100, 10.0 * r_deg_10)


def test_hill_half_max_at_K():
    """At U = K, r_prod ≈ k_prod / 2."""
    params = PhysicsParams(k_prod=10.0, K=0.4, n=2.0, k_deg=0.05)
    state = PhysicsState.create(time=jnp.array(0.0), X=jnp.array(0.0))
    p = compute_propensities(state, jnp.array(0.4, dtype=jnp.float32), params)
    assert jnp.isclose(p[0], 5.0, atol=1e-5)


def test_hill_saturation_large_n():
    """At U=1.0 and K << 1 with large n, r_prod saturates near k_prod."""
    params = PhysicsParams(k_prod=10.0, K=0.1, n=6.0, k_deg=0.05)
    state = PhysicsState.create(time=jnp.array(0.0), X=jnp.array(0.0))
    p_full = compute_propensities(state, jnp.array(1.0, dtype=jnp.float32), params)
    assert 9.9 < p_full[0] <= 10.0


@pytest.mark.parametrize(
    "reaction_idx,X_delta",
    [(0, +1), (1, -1)],
)
def test_reaction_effects(reaction_idx: int, X_delta: int):
    state = PhysicsState.create(time=jnp.array(0.0), X=jnp.array(50.0))
    next_state = apply_reaction(state, jnp.array(reaction_idx))
    assert next_state.X == state.X + X_delta


def test_degradation_floor_at_zero():
    state = PhysicsState.create(time=jnp.array(0.0), X=jnp.array(0.0))
    next_state = apply_reaction(state, jnp.array(1))  # degradation
    assert next_state.X >= 0


def test_step_physics_advances_time(params: PhysicsParams, config: PhysicsConfig):
    key = jax.random.PRNGKey(0)
    state = PhysicsState.create(time=jnp.array(0.0), X=jnp.array(20.0))
    action = jnp.array(1.0, dtype=jnp.float32)
    previous_action = jnp.array(0.0, dtype=jnp.float32)

    next_state = step_physics(key, state, action, params, config, previous_action, jnp.array(0.0))

    target_time = config.timestep_minutes
    assert next_state.time >= state.time
    assert next_state.time <= target_time


def test_step_physics_light_on_grows_x(params: PhysicsParams, config: PhysicsConfig):
    """With constant U=1 the copy number X grows from zero."""
    key = jax.random.PRNGKey(0)
    state = PhysicsState.create(time=jnp.array(0.0), X=jnp.array(0.0))
    action = jnp.array(1.0, dtype=jnp.float32)
    for t in range(40):
        key, subkey = jax.random.split(key)
        interval_start = jnp.array(t * config.timestep_minutes)
        state = step_physics(subkey, state, action, params, config, action, interval_start)
    assert state.X > 0.0


def test_step_physics_light_off_decays_x(params: PhysicsParams, config: PhysicsConfig):
    """With U=0 the copy number decays from a high initial value."""
    key = jax.random.PRNGKey(0)
    state = PhysicsState.create(time=jnp.array(0.0), X=jnp.array(200.0))
    action = jnp.array(0.0, dtype=jnp.float32)
    for t in range(80):
        key, subkey = jax.random.split(key)
        interval_start = jnp.array(t * config.timestep_minutes)
        state = step_physics(subkey, state, action, params, config, action, interval_start)
    assert state.X < 200.0


def test_step_physics_finite_values(params: PhysicsParams, config: PhysicsConfig):
    key = jax.random.PRNGKey(0)
    state = PhysicsState.create(time=jnp.array(0.0), X=jnp.array(50.0))
    previous_action = jnp.array(0.0, dtype=jnp.float32)

    for t in range(50):
        key, k_action, k_step = jax.random.split(key, 3)
        action = jax.random.uniform(k_action, minval=0.0, maxval=1.0, dtype=jnp.float32)
        interval_start = jnp.array(t * config.timestep_minutes)
        state = step_physics(k_step, state, action, params, config, previous_action, interval_start)
        previous_action = action

        assert jnp.isfinite(state.time)
        assert jnp.isfinite(state.X) and state.X >= 0


def test_step_physics_stochastic(params: PhysicsParams, config: PhysicsConfig):
    state = PhysicsState.create(time=jnp.array(0.0), X=jnp.array(50.0))
    action = jnp.array(1.0, dtype=jnp.float32)

    k1 = jax.random.PRNGKey(0)
    k2 = jax.random.PRNGKey(1)
    s1, s2 = state, state
    for t in range(10):
        k1, sub1 = jax.random.split(k1)
        k2, sub2 = jax.random.split(k2)
        interval_start = jnp.array(t * config.timestep_minutes)
        s1 = step_physics(sub1, s1, action, params, config, action, interval_start)
        s2 = step_physics(sub2, s2, action, params, config, action, interval_start)

    assert not jnp.allclose(s1.X, s2.X) or not jnp.allclose(s1.time, s2.time), "Different RNG keys should diverge"


def test_step_physics_jit(params: PhysicsParams, config: PhysicsConfig):
    key = jax.random.PRNGKey(0)
    state = PhysicsState.create(time=jnp.array(0.0), X=jnp.array(20.0))
    action = jnp.array(0.7, dtype=jnp.float32)
    jitted = jax.jit(step_physics, static_argnames=["config"])
    next_state = jitted(key, state, action, params, config, action, jnp.array(0.0))
    assert jnp.isfinite(next_state.X)


def test_step_physics_vmap(params: PhysicsParams, config: PhysicsConfig):
    batch_size = 5
    states = PhysicsState(
        time=jnp.zeros(batch_size),
        X=jnp.linspace(10.0, 100.0, batch_size),
        next_reaction_time=jnp.full(batch_size, jnp.inf),
    )
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    actions = jnp.linspace(0.0, 1.0, batch_size, dtype=jnp.float32)
    previous_actions = jnp.zeros(batch_size, dtype=jnp.float32)
    interval_starts = jnp.zeros(batch_size)

    vmap_step = jax.vmap(step_physics, in_axes=(0, 0, 0, None, None, 0, 0))
    next_states = vmap_step(keys, states, actions, params, config, previous_actions, interval_starts)

    assert next_states.X.shape == (batch_size,)
    assert next_states.time.shape == (batch_size,)
    assert next_states.next_reaction_time.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(next_states.X))


def test_state_to_array_roundtrip():
    state = PhysicsState.create(time=jnp.array(10.0), X=jnp.array(42.0))
    arr = state.to_array()
    assert arr.shape == (2,)
    assert jnp.allclose(arr, jnp.array([10.0, 42.0]))

    restored = PhysicsState.from_array(arr)
    assert jnp.allclose(restored.time, state.time)
    assert jnp.allclose(restored.X, state.X)


@pytest.mark.slow
def test_steady_state_mean_tracks_theory(params: PhysicsParams, config: PhysicsConfig):
    """Long-run mean of X under constant U=1 should approach k_prod/k_deg.

    At U=1 with K=0.5 and n=2, hill(1, 0.5, 2) = 1/(1 + 0.25) = 0.8, so the
    steady-state mean is 0.8 · k_prod / k_deg = 0.8 · 5.0 / 0.05 = 80.
    Average over many cells and the tail of the trajectory.
    """
    n_envs = 128
    n_steps = 400  # long enough to equilibrate
    burn_in = 200

    keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
    states = PhysicsState(
        time=jnp.zeros(n_envs),
        X=jnp.zeros(n_envs),
        next_reaction_time=jnp.full(n_envs, jnp.inf),
    )
    action = jnp.ones(n_envs, dtype=jnp.float32)
    prev_action = jnp.zeros(n_envs, dtype=jnp.float32)

    vmap_step = jax.jit(
        jax.vmap(step_physics, in_axes=(0, 0, 0, None, None, 0, 0)),
        static_argnames=["config"],
    )

    x_samples = []
    for t in range(n_steps):
        keys = jax.vmap(lambda k: jax.random.split(k)[0])(keys)
        step_keys = jax.vmap(lambda k: jax.random.split(k)[1])(keys)
        interval_starts = jnp.full(n_envs, t * config.timestep_minutes)
        states = vmap_step(step_keys, states, action, params, config, prev_action, interval_starts)
        prev_action = action
        if t >= burn_in:
            x_samples.append(states.X)

    x_mean = float(jnp.mean(jnp.stack(x_samples)))
    expected = 0.8 * float(params.k_prod) / float(params.k_deg)
    # Generous tolerance — stochastic and finite-sample
    assert abs(x_mean - expected) / expected < 0.2, f"mean X={x_mean:.1f}, expected ≈ {expected:.1f}"
