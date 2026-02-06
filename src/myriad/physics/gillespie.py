"""Shared Gillespie algorithm utilities for stochastic chemical kinetics.

This module provides reusable components for implementing exact stochastic simulation
of chemical reaction networks using the Gillespie algorithm (aka Stochastic Simulation Algorithm).

The core philosophy is to extract the common while_loop logic while keeping
environment-specific parts (propensities, state updates) fast and explicit.

Usage Pattern:
    1. Each environment defines its own PhysicsState (as NamedTuple)
    2. Implements compute_propensities(state, action, config) -> Array
    3. Implements apply_reaction(state, reaction_idx) -> PhysicsState
    4. Calls run_gillespie_loop() with these callbacks

Performance:
    - The while_loop is compiled once and reused
    - Environment-specific functions (propensities, reactions) stay fast
    - No overhead from abstraction (callbacks are inlined by JIT)

Reference:
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
) -> Any:
    """Execute Gillespie algorithm simulation loop.

    Runs exact stochastic simulation from current time until target_time,
    sampling reaction times and events from the master equation.

    This is the SHARED part of all Gillespie-based environments. The algorithm:
    1. Compute reaction propensities (rates) for current state
    2. Sample time until next reaction (exponential distribution)
    3. Sample which reaction occurs (categorical distribution)
    4. Update state by applying reaction
    5. Repeat until target_time or max_steps reached

    Args:
        key: RNG key for stochastic sampling
        initial_state: Initial state (environment-specific structure)
        action: Control input (passed to propensity function)
        config: Configuration object (environment-specific)
        target_time: Simulation target time (typically timestep_minutes)
        max_steps: Safety limit for maximum Gillespie steps
        compute_propensities_fn: Function to compute reaction rates
            Signature: (state, action, config) -> Array[n_reactions]
            Returns: Array of propensities for each reaction
        apply_reaction_fn: Function to apply a reaction to state
            Signature: (state, reaction_idx) -> state
            Returns: Updated state after reaction
        get_time_fn: Function to extract current time from state
            Signature: (state) -> float
        update_time_fn: Function to set time in state
            Signature: (state, time) -> state
            Returns: State with updated time

    Returns:
        final_state: State after simulating until target_time

    Example:
        >>> # In your environment's physics.py:
        >>> from myriad.physics.gillespie import run_gillespie_loop
        >>>
        >>> def step_physics(key, state, action, params, config):
        ...     final_state = run_gillespie_loop(
        ...         key=key,
        ...         initial_state=state,
        ...         action=action,
        ...         config=config,
        ...         target_time=config.timestep_minutes,
        ...         max_steps=config.max_gillespie_steps,
        ...         compute_propensities_fn=compute_propensities,
        ...         apply_reaction_fn=apply_reaction,
        ...         get_time_fn=lambda s: s.time,
        ...         update_time_fn=lambda s, t: s._replace(time=t),
        ...     )
        ...     return final_state

    Notes:
        - This function is JIT-compiled together with the callbacks
        - Callbacks are inlined by XLA, so there's no function call overhead
        - The while_loop is compiled once and reused across environments
        - Time advances stochastically based on reaction rates
        - Simulations are exact (not approximate like tau-leaping)
    """

    def cond_fn(carry):
        """Continue while time < target and steps < max."""
        state, time, step, key = carry
        return (time < target_time) & (step < max_steps)

    def body_fn(carry):
        """Execute one Gillespie step: sample and apply reaction."""
        state, time, step, key = carry

        # Split key and use gillespie_step for core logic
        key, subkey = jax.random.split(key)

        new_state, tau = gillespie_step(subkey, state, action, config, compute_propensities_fn, apply_reaction_fn)

        # Advance time
        new_time = time + tau

        return new_state, new_time, step + 1, key

    # Run the Gillespie loop
    initial_time = get_time_fn(initial_state)
    final_state, final_time, final_step, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (initial_state, initial_time, jnp.array(0), key),
    )

    # Set final time to exactly target_time (may slightly exceed due to discrete events)
    final_state = update_time_fn(final_state, target_time)

    return final_state


def gillespie_step(
    key: PRNGKey,
    state: Any,
    action: Array,
    config: Any,
    compute_propensities_fn: Callable[[Any, Array, Any], Array],
    apply_reaction_fn: Callable[[Any, Array], Any],
) -> tuple[Any, Array]:
    """Execute a single Gillespie step (one reaction).

    This is a lower-level utility for cases where you want manual control
    over the simulation loop. Most environments should use run_gillespie_loop() instead.

    Args:
        key: RNG key for stochastic sampling
        state: Current state
        action: Control input
        config: Configuration object
        compute_propensities_fn: Function to compute reaction rates
        apply_reaction_fn: Function to apply reaction

    Returns:
        new_state: State after one reaction
        tau: Time until reaction occurred

    Example:
        >>> # Manual control over Gillespie loop:
        >>> state = initial_state
        >>> time = 0.0
        >>> while time < target_time:
        ...     key, subkey = jax.random.split(key)
        ...     state, tau = gillespie_step(subkey, state, action, config, ...)
        ...     time += tau
    """
    # Split RNG keys
    key_time, key_reaction = jax.random.split(key)

    # Compute propensities
    propensities = compute_propensities_fn(state, action, config)
    a0 = jnp.sum(propensities)

    # Sample time until next reaction
    tau = jax.random.exponential(key_time) / jnp.maximum(a0, 1e-10)

    # Sample which reaction occurs
    reaction_probs = propensities / jnp.maximum(a0, 1e-10)
    reaction_idx = jax.random.choice(key_reaction, len(propensities), p=reaction_probs)

    # Apply reaction
    new_state = apply_reaction_fn(state, reaction_idx)

    return new_state, tau
