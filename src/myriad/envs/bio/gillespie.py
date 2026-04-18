"""Gillespie interval helper shared across bio environments.

Each bio environment wraps the generic ``myriad.physics.gillespie.run_gillespie_loop``
with the same boilerplate: compute ``target_time = interval_start + timestep_minutes``,
close the env's ``compute_propensities`` over ``params``, thread the pending reaction
time through the state, and write the new pending time back on the returned state.

This helper centralises that wrapping so individual envs only need to provide their
propensity and reaction functions.

State assumptions (structural, not a Protocol):
    The state object must be a NamedTuple-like value with
    - ``time`` (scalar Array)
    - ``next_reaction_time`` (scalar Array)
    - ``_replace(**kwargs)`` returning a new instance
    PhysicsConfig must expose ``timestep_minutes`` and ``max_gillespie_steps``.
"""

from typing import Any, Callable

from jax import Array

from myriad.core.types import PRNGKey
from myriad.physics.gillespie import run_gillespie_loop


def step_gillespie_interval(
    key: PRNGKey,
    state: Any,
    action: Array,
    params: Any,
    config: Any,
    *,
    compute_propensities_fn: Callable[[Any, Array, Any], Array],
    apply_reaction_fn: Callable[[Any, Array], Any],
    previous_action: Array,
    interval_start: Array,
) -> Any:
    """Advance a bio state one RL interval via the Gillespie SSA.

    Args:
        key: RNG key for stochastic simulation.
        state: Current state (see module docstring for required fields).
        action: Control input used by ``compute_propensities_fn``.
        params: Dynamic parameters forwarded to ``compute_propensities_fn``.
        config: Physics config exposing ``timestep_minutes`` and
            ``max_gillespie_steps``.
        compute_propensities_fn: ``(state, action, params) -> Array`` of
            reaction propensities.
        apply_reaction_fn: ``(state, reaction_idx) -> state`` applying a
            single reaction.
        previous_action: Action from the previous interval — used to invalidate
            pending reaction times when it differs from ``action``.
        interval_start: Absolute start time of this interval (``t * dt``).

    Returns:
        Updated state with ``time`` and ``next_reaction_time`` set.
    """
    target_time = interval_start + config.timestep_minutes

    def _propensities(s: Any, a: Array) -> Array:
        return compute_propensities_fn(s, a, params)

    final_state, next_reaction_time = run_gillespie_loop(
        key=key,
        initial_state=state,
        action=action,
        target_time=target_time,
        max_steps=config.max_gillespie_steps,
        compute_propensities_fn=_propensities,
        apply_reaction_fn=apply_reaction_fn,
        get_time_fn=lambda s: s.time,
        update_time_fn=lambda s, t: s._replace(time=t),
        pending_reaction_time=state.next_reaction_time,
        previous_action=previous_action,
    )
    return final_state._replace(next_reaction_time=next_reaction_time)
