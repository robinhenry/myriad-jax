"""Gillespie algorithm for exact stochastic simulation of chemical kinetics.

This module implements the Stochastic Simulation Algorithm (SSA) for simulating
chemical reaction networks with discrete molecular counts.

Mathematical Background
-----------------------
For a system with M reaction channels and propensities $a_j(x)$ where $x$ is the
state vector, the Gillespie algorithm samples from the Chemical Master Equation:

1. **Time to next reaction**: $\\tau \\sim \\text{Exp}(a_0)$ where $a_0 = \\sum_{j=1}^{M} a_j$

2. **Which reaction occurs**: $P(\\text{reaction } j) = a_j / a_0$

The exponential distribution has the **memoryless property**:
$P(\\tau > t + s \\mid \\tau > t) = P(\\tau > s)$

This means that if we sample $\\tau$ at time $t$ and the reaction is scheduled for
$t + \\tau$, but we only simulate until some boundary $t_{end} < t + \\tau$, then:
- The reaction is still pending at $t + \\tau$
- We should NOT resample when continuing from $t_{end}$
- Resampling would be statistically equivalent but physically incorrect

Implementation Design
---------------------
This implementation preserves pending reaction times across simulation boundaries
to maintain physical accuracy:

- When $t + \\tau > t_{end}$: return the pending time $t + \\tau$ for use in next call
- When action changes: invalidate pending time (propensities $a_j$ changed)
- When state changes (reaction occurred): sample fresh (propensities changed)

This ensures trajectories are independent of the RL discretization boundaries.

Reference
---------
Gillespie, D. T. (1977). "Exact stochastic simulation of coupled chemical reactions."
The Journal of Physical Chemistry, 81(25), 2340-2361.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from myriad.core.types import PRNGKey


def run_gillespie_loop(
    key: PRNGKey,
    initial_state: Any,
    action: Array,
    config: Any,
    target_time: float | Array,
    max_steps: int,
    compute_propensities_fn: Callable[[Any, Array, Any], Array],
    apply_reaction_fn: Callable[[Any, Array], Any],
    get_time_fn: Callable[[Any], Array],
    update_time_fn: Callable[[Any, float | Array], Any],
    pending_reaction_time: Array | None = None,
    previous_action: Array | None = None,
) -> tuple[Any, Array]:
    """Execute Gillespie simulation with preserved reaction times.

    Simulates the chemical system from current time until ``target_time``,
    executing reactions as they occur according to the SSA.

    Preserving Reaction Times
    -------------------------
    When a sampled reaction time $t_{rxn}$ exceeds ``target_time``, we return
    $t_{rxn}$ so the caller can use it in the next simulation interval. This is
    physically correct because:

    1. The reaction was legitimately scheduled at $t_{rxn}$
    2. The memoryless property does NOT mean we should resample
    3. Resampling at boundaries would make trajectories depend on discretization

    Action Changes
    --------------
    When ``previous_action`` is provided and differs from ``action``, the pending
    reaction time is automatically invalidated. This is necessary because:

    - Propensities $a_j(x, u)$ depend on the action $u$
    - A pending time sampled with old propensities is no longer valid
    - We must sample fresh with the new action's propensities

    Args:
        key: RNG key for stochastic sampling
        initial_state: Starting state (environment-specific structure)
        action: Current control input passed to propensity function
        config: Configuration object with rate constants
        target_time: Simulate until this time (typically interval end)
        max_steps: Safety limit to prevent infinite loops
        compute_propensities_fn: $(x, u, \\theta) \\to [a_1, ..., a_M]$
        apply_reaction_fn: $(x, j) \\to x'$ applies reaction $j$
        get_time_fn: Extracts time from state
        update_time_fn: Sets time in state
        pending_reaction_time: Scheduled time from previous call, or ``inf``/``None``
            to sample fresh.
        previous_action: Action from previous timestep. If provided and different
            from ``action``, the pending reaction time is invalidated.

    Returns:
        final_state: State after simulating until ``target_time``
        next_reaction_time: Pending reaction time for next call (may be > ``target_time``)
    """

    def sample_tau(key: PRNGKey, propensities: Array) -> Array:
        """Sample time until next reaction: $\\tau \\sim \\text{Exp}(a_0)$."""
        a0 = jnp.sum(propensities)
        return jax.random.exponential(key) / jnp.maximum(a0, 1e-10)

    def sample_reaction(key: PRNGKey, propensities: Array) -> Array:
        """Sample which reaction: $P(j) = a_j / a_0$."""
        a0 = jnp.sum(propensities)
        probs = propensities / jnp.maximum(a0, 1e-10)
        return jax.random.choice(key, len(propensities), p=probs)

    # Initialize pending reaction time (inf means "sample fresh")
    if pending_reaction_time is None:
        pending_reaction_time = jnp.array(jnp.inf)

    # Invalidate pending reaction if action changed (propensities are different)
    if previous_action is not None:
        action_changed = jnp.logical_xor(previous_action, action)
        pending_reaction_time = jnp.where(action_changed, jnp.array(jnp.inf), pending_reaction_time)

    # Sample initial reaction time if needed
    initial_time = get_time_fn(initial_state)
    key, key_init = jax.random.split(key)
    needs_sample = jnp.isinf(pending_reaction_time)
    initial_propensities = compute_propensities_fn(initial_state, action, config)
    sampled_tau = sample_tau(key_init, initial_propensities)
    next_reaction_time = jnp.where(needs_sample, initial_time + sampled_tau, pending_reaction_time)

    def cond_fn(carry):
        """Continue while next reaction is within interval and under step limit."""
        state, next_rxn_time, step, key = carry
        return (next_rxn_time < target_time) & (step < max_steps)

    def body_fn(carry):
        """Execute one Gillespie step.

        1. Advance time to scheduled reaction time
        2. Sample which reaction occurs (using pre-reaction propensities)
        3. Apply the reaction to get new state
        4. Sample next reaction time (using post-reaction propensities)
        """
        state, next_rxn_time, step, key = carry
        key, key_reaction, key_time = jax.random.split(key, 3)

        # Advance time to when reaction occurs
        state = update_time_fn(state, next_rxn_time)

        # Compute propensities for current (pre-reaction) state
        propensities = compute_propensities_fn(state, action, config)

        # Sample and apply reaction
        reaction_idx = sample_reaction(key_reaction, propensities)
        state = apply_reaction_fn(state, reaction_idx)

        # Compute propensities for new (post-reaction) state and sample next tau
        # Note: propensities change because state changed
        new_propensities = compute_propensities_fn(state, action, config)
        tau = sample_tau(key_time, new_propensities)
        new_next_rxn_time = get_time_fn(state) + tau

        return state, new_next_rxn_time, step + 1, key

    # Run simulation loop
    final_state, final_next_rxn_time, _, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (initial_state, next_reaction_time, jnp.array(0), key),
    )

    return final_state, final_next_rxn_time
