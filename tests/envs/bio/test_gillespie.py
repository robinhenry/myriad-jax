"""Tests for the shared bio Gillespie interval helper.

The helper assumes state has ``time``, ``next_reaction_time``, and ``_replace``,
and config exposes ``timestep_minutes`` and ``max_gillespie_steps``. Both real
bio envs (ccasr_gfp, opto_hill_1d) already satisfy these contracts; this suite
exercises the helper against a minimal pure-Python state to guarantee the
wrapper logic itself is covered.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import struct

from myriad.envs.bio.gillespie import step_gillespie_interval


class _State(NamedTuple):
    time: jnp.ndarray
    x: jnp.ndarray
    next_reaction_time: jnp.ndarray


@struct.dataclass
class _Config:
    timestep_minutes: float = 5.0
    max_gillespie_steps: int = 1000


@struct.dataclass
class _Params:
    rate: float = 0.5


def _propensities(state, action, params):
    # Single reaction with action-dependent propensity
    return jnp.array([params.rate * (action + 0.1)])


def _apply(state, _idx):
    return state._replace(x=state.x + 1)


def test_interval_advances_time_within_bounds():
    config = _Config()
    params = _Params(rate=2.0)
    initial = _State(
        time=jnp.array(0.0),
        x=jnp.array(0.0),
        next_reaction_time=jnp.array(jnp.inf),
    )
    next_state = step_gillespie_interval(
        jax.random.PRNGKey(0),
        initial,
        jnp.array(1.0),
        params,
        config,
        compute_propensities_fn=_propensities,
        apply_reaction_fn=_apply,
        previous_action=jnp.array(1.0),
        interval_start=jnp.array(0.0),
    )
    assert float(next_state.time) >= 0.0
    assert float(next_state.time) <= config.timestep_minutes
    assert jnp.isfinite(next_state.next_reaction_time) or jnp.isinf(next_state.next_reaction_time)


def test_interval_preserves_next_reaction_time_field():
    """The helper must always write back next_reaction_time on the returned state."""
    config = _Config()
    params = _Params()
    initial = _State(
        time=jnp.array(0.0),
        x=jnp.array(0.0),
        next_reaction_time=jnp.array(jnp.inf),
    )
    next_state = step_gillespie_interval(
        jax.random.PRNGKey(1),
        initial,
        jnp.array(0.0),  # zero action → very slow reactions
        params,
        config,
        compute_propensities_fn=_propensities,
        apply_reaction_fn=_apply,
        previous_action=jnp.array(0.0),
        interval_start=jnp.array(0.0),
    )
    assert hasattr(next_state, "next_reaction_time")
    assert not jnp.isnan(next_state.next_reaction_time)


def test_high_propensity_fires_reactions():
    """With a large propensity, multiple reactions should fire in one interval."""
    config = _Config(timestep_minutes=10.0)
    params = _Params(rate=5.0)
    initial = _State(
        time=jnp.array(0.0),
        x=jnp.array(0.0),
        next_reaction_time=jnp.array(jnp.inf),
    )
    next_state = step_gillespie_interval(
        jax.random.PRNGKey(2),
        initial,
        jnp.array(1.0),
        params,
        config,
        compute_propensities_fn=_propensities,
        apply_reaction_fn=_apply,
        previous_action=jnp.array(1.0),
        interval_start=jnp.array(0.0),
    )
    assert float(next_state.x) > 0
