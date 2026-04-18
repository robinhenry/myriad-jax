"""Regression tests for myriad.physics.gillespie.run_gillespie_loop.

The action-change detection logic must handle both discrete binary actions
(as used by ccasr_gfp) and continuous actions (as used by opto_hill_1d).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from myriad.physics.gillespie import run_gillespie_loop


class _State(NamedTuple):
    time: jnp.ndarray
    x: jnp.ndarray


def _minimal_system():
    """Trivial 1-reaction system with action-dependent propensity.

    Propensity depends on the action so that changing U must invalidate
    any pending reaction scheduled with the previous propensity.
    """

    def propensities(state, action):
        return jnp.array([action + 0.1])  # always positive

    def apply_reaction(state, _idx):
        return state._replace(x=state.x + 1)

    return propensities, apply_reaction


def test_action_change_invalidates_pending_binary():
    """Binary action change still invalidates pending reaction (regression)."""
    propensities, apply_reaction = _minimal_system()
    key = jax.random.PRNGKey(0)
    initial = _State(time=jnp.array(0.0), x=jnp.array(0.0))

    # Pending reaction scheduled far in the future
    pending = jnp.array(10.0)
    _, next_rxn = run_gillespie_loop(
        key=key,
        initial_state=initial,
        action=jnp.array(1.0),
        target_time=jnp.array(0.01),  # short enough to not fire a reaction
        max_steps=1,
        compute_propensities_fn=propensities,
        apply_reaction_fn=apply_reaction,
        get_time_fn=lambda s: s.time,
        update_time_fn=lambda s, t: s._replace(time=t),
        pending_reaction_time=pending,
        previous_action=jnp.array(0.0),  # action changed 0 -> 1
    )
    # Pending time must have been invalidated (resampled)
    assert not jnp.isclose(next_rxn, pending), "Binary action change must invalidate pending"


def test_action_unchanged_preserves_pending_binary():
    """When action is unchanged and pending time is future, it's preserved."""
    propensities, apply_reaction = _minimal_system()
    key = jax.random.PRNGKey(0)
    initial = _State(time=jnp.array(0.0), x=jnp.array(0.0))

    pending = jnp.array(10.0)
    _, next_rxn = run_gillespie_loop(
        key=key,
        initial_state=initial,
        action=jnp.array(1.0),
        target_time=jnp.array(0.01),
        max_steps=1,
        compute_propensities_fn=propensities,
        apply_reaction_fn=apply_reaction,
        get_time_fn=lambda s: s.time,
        update_time_fn=lambda s, t: s._replace(time=t),
        pending_reaction_time=pending,
        previous_action=jnp.array(1.0),  # unchanged
    )
    assert jnp.isclose(next_rxn, pending), "Unchanged action must preserve pending time"


def test_action_change_invalidates_pending_continuous():
    """Continuous action change must invalidate pending reaction.

    This is the case that broke with the old `jnp.logical_xor(prev, curr)`
    check: prev=0.3 and curr=0.7 both cast to True, XOR is False, and the
    stale pending time (sampled under a different propensity) was wrongly kept.
    """
    propensities, apply_reaction = _minimal_system()
    key = jax.random.PRNGKey(0)
    initial = _State(time=jnp.array(0.0), x=jnp.array(0.0))

    pending = jnp.array(10.0)
    _, next_rxn = run_gillespie_loop(
        key=key,
        initial_state=initial,
        action=jnp.array(0.7),
        target_time=jnp.array(0.01),
        max_steps=1,
        compute_propensities_fn=propensities,
        apply_reaction_fn=apply_reaction,
        get_time_fn=lambda s: s.time,
        update_time_fn=lambda s, t: s._replace(time=t),
        pending_reaction_time=pending,
        previous_action=jnp.array(0.3),  # both "truthy" under old XOR check
    )
    assert not jnp.isclose(
        next_rxn, pending
    ), "Continuous action change must invalidate pending (regression for logical_xor bug)"


def test_action_unchanged_preserves_pending_continuous():
    """Unchanged continuous action must preserve the pending reaction time."""
    propensities, apply_reaction = _minimal_system()
    key = jax.random.PRNGKey(0)
    initial = _State(time=jnp.array(0.0), x=jnp.array(0.0))

    pending = jnp.array(10.0)
    _, next_rxn = run_gillespie_loop(
        key=key,
        initial_state=initial,
        action=jnp.array(0.42),
        target_time=jnp.array(0.01),
        max_steps=1,
        compute_propensities_fn=propensities,
        apply_reaction_fn=apply_reaction,
        get_time_fn=lambda s: s.time,
        update_time_fn=lambda s, t: s._replace(time=t),
        pending_reaction_time=pending,
        previous_action=jnp.array(0.42),
    )
    assert jnp.isclose(next_rxn, pending), "Unchanged continuous action must preserve pending"
