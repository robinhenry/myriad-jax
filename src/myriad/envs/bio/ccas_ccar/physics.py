"""Pure stateless physics for the CcaS-CcaR gene circuit system.

This module contains the ground truth stochastic dynamics for the bi-stable genetic
circuit, using the Gillespie algorithm for exact stochastic simulation.

The physics is completely decoupled from any task-specific logic (rewards, terminations, observations).
It can be reused by different tasks (control, SysID, etc.).

System Description:
    Light Input (U) → CcaSR (H) → GFP (F) with autoactivation feedback

    Five Chemical Reactions:
    1. CcaSR activation: ∅ → H  (rate: eta * U)
    2. CcaSR deactivation: H → ∅  (rate: nu * H)
    3. F creation from H: ∅ → F  (rate: 0.5 * a * H^nh / (Kh^nh + H^nh))
    4. F self-activation: ∅ → F  (rate: 0.5 * a * F^nf / (Kf^nf + F^nf))
    5. F dilution: F → ∅  (rate: nu * F)

Reference:
    Based on the bi-stable genetic circuit model from:
    "Control of a Bi-Stable Genetic System via Parallelized Reinforcement Learning"
    CDC 2025, https://gitlab.com/lugagnelab/pqn-control-cdc2025
"""

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from myriad.core.types import PRNGKey
from myriad.physics import hill_function
from myriad.physics.gillespie import run_gillespie_loop


class PhysicsState(NamedTuple):
    """Pure physical state of the gene circuit system.

    Attributes:
        time: Current simulation time (minutes)
        H: CcaS-CcaR protein concentration (molecules)
        F: GFP reporter protein concentration (molecules)
        next_reaction_time: Scheduled time of next reaction (minutes).
            Set to inf when no reaction is pending (sample fresh).
            Preserved across RL step boundaries for physical accuracy.
    """

    time: Array
    H: Array
    F: Array
    next_reaction_time: Array

    def to_array(self) -> Array:
        """Convert to flat array for NN-based agents.

        Note: next_reaction_time is excluded as it's internal bookkeeping.

        Returns:
            Array of shape (3,) with [time, H, F]
        """
        return jnp.stack([self.time, self.H, self.F])

    @classmethod
    def from_array(cls, arr: Array) -> "PhysicsState":
        """Create state from flat array.

        Args:
            arr: Array of shape (3,) with [time, H, F]

        Returns:
            PhysicsState instance (next_reaction_time defaults to inf)
        """
        chex.assert_shape(arr, (3,))
        return cls(
            time=arr[0],  # type: ignore
            H=arr[1],  # type: ignore
            F=arr[2],  # type: ignore
            next_reaction_time=jnp.array(jnp.inf),
        )

    @classmethod
    def create(
        cls,
        time: Array,
        H: Array,
        F: Array,
        next_reaction_time: Array | None = None,
    ) -> "PhysicsState":
        """Factory method to create PhysicsState with default next_reaction_time.

        Args:
            time: Current simulation time
            H: CcaS-CcaR protein concentration
            F: GFP reporter protein concentration
            next_reaction_time: Optional pending reaction time (defaults to inf)

        Returns:
            PhysicsState instance
        """
        if next_reaction_time is None:
            next_reaction_time = jnp.array(jnp.inf)
        return cls(time=time, H=H, F=F, next_reaction_time=next_reaction_time)


@struct.dataclass
class PhysicsConfig:
    """Static physics constants for the gene circuit system.

    These are compile-time constants passed as static_argnames to jit.
    Changing these values requires recompilation but enables better optimization.

    Default values from the CDC 2025 paper implementation.
    """

    # Production and dilution rates
    eta: float = 1.0  # CcaSR (H) production rate (1/min)
    nu: float = 0.01  # Protein dilution rate (1/min)

    # Promoter dynamics for H-induced F production
    a: float = 1.0  # Maximum promoter activity (1/min)
    Kh: float = 90.0  # Half-maximal H concentration
    nh: float = 3.6  # Hill coefficient for H cooperativity

    # Self-activation dynamics for F-induced F production
    Kf: float = 30.0  # Half-maximal F concentration
    nf: float = 3.6  # Hill coefficient for F cooperativity

    # Time discretization
    timestep_minutes: float = 5.0  # Physical timestep (minutes)

    # Gillespie algorithm parameters
    max_gillespie_steps: int = 10000  # Safety limit for Gillespie loop per step


@struct.dataclass
class PhysicsParams:
    """Dynamic physics parameters for domain randomization.

    These can be randomized per episode to create diverse dynamics.
    Currently empty but maintained for protocol consistency.
    """

    ...


def compute_propensities(
    state: PhysicsState,
    action: Array,
    config: PhysicsConfig,
) -> Array:
    """Compute reaction propensities (rates) for all five reactions.

    Args:
        state: Current physical state (time, H, F)
        action: Discrete action {0, 1} representing light input U
        config: Static physics constants

    Returns:
        Array of 5 propensities for reactions [R1, R2, R3, R4, R5]
    """
    H = state.H
    F = state.F
    U = action  # Light input is directly the action

    # Reaction 1: CcaSR activation (∅ → H)
    r1 = config.eta * U

    # Reaction 2: CcaSR deactivation (H → ∅)
    r2 = config.nu * H

    # Reaction 3: F creation from H (∅ → F)
    # Hill function: 0.5 * a * H^nh / (Kh^nh + H^nh)
    r3 = 0.5 * config.a * hill_function(H, config.Kh, config.nh)

    # Reaction 4: F self-activation (∅ → F)
    # Hill function: 0.5 * a * F^nf / (Kf^nf + F^nf)
    r4 = 0.5 * config.a * hill_function(F, config.Kf, config.nf)

    # Reaction 5: F dilution (F → ∅)
    r5 = config.nu * F

    return jnp.array([r1, r2, r3, r4, r5])


def apply_reaction(state: PhysicsState, reaction_idx: Array) -> PhysicsState:
    """Apply a single reaction to update the state.

    Uses jax.lax.switch for JAX-compatible control flow.

    Args:
        state: Current physical state
        reaction_idx: Index of reaction to apply (0-4)

    Returns:
        Updated physical state after reaction
    """

    def reaction_0(s):
        """Reaction 1: ∅ → H (CcaSR activation)"""
        return s._replace(H=s.H + 1)

    def reaction_1(s):
        """Reaction 2: H → ∅ (CcaSR deactivation)"""
        return s._replace(H=jnp.maximum(s.H - 1, 0))

    def reaction_2(s):
        """Reaction 3: ∅ → F (F creation from H)"""
        return s._replace(F=s.F + 1)

    def reaction_3(s):
        """Reaction 4: ∅ → F (F self-activation)"""
        return s._replace(F=s.F + 1)

    def reaction_4(s):
        """Reaction 5: F → ∅ (F dilution)"""
        return s._replace(F=jnp.maximum(s.F - 1, 0))

    branches = [reaction_0, reaction_1, reaction_2, reaction_3, reaction_4]
    return jax.lax.switch(reaction_idx, branches, state)


def step_physics(
    key: PRNGKey,
    state: PhysicsState,
    action: Array,
    params: PhysicsParams,
    config: PhysicsConfig,
    previous_action: Array,
    interval_start: Array,
) -> PhysicsState:
    """Pure physics step using a discrete Gillespie algorithm.

    Runs Gillespie simulation from current time until the end of the current
    interval (``interval_start + timestep_minutes``). Intervals are at fixed absolute
    times (0, 5, 10, 15...), matching the physical setup where observations and
    actions occur at regular intervals.

    Args:
        key: RNG key for stochastic simulation
        state: Current physical state (``time``, ``H``, ``F``, ``next_reaction_time``)
        action: Discrete action ``{0, 1}`` representing light input
        params: Dynamic parameters
        config: Static physics constants
        previous_action: Action from previous timestep. If different from action,
            the pending reaction time is invalidated (propensities changed).
        interval_start: Start time of current interval (``t * timestep_minutes``).

    Returns:
        Next physical state after simulating until interval end
    """
    target_time = interval_start + config.timestep_minutes

    final_state, next_reaction_time = run_gillespie_loop(
        key=key,
        initial_state=state,
        action=action,
        config=config,
        target_time=target_time,
        max_steps=config.max_gillespie_steps,
        compute_propensities_fn=compute_propensities,
        apply_reaction_fn=apply_reaction,
        get_time_fn=lambda s: s.time,
        update_time_fn=lambda s, t: s._replace(time=t),
        pending_reaction_time=state.next_reaction_time,
        previous_action=previous_action,
    )

    # Store the pending reaction time for the next step
    return final_state._replace(next_reaction_time=next_reaction_time)


def create_physics_params(**kwargs) -> PhysicsParams:
    """Factory function to create PhysicsParams.

    Args:
        **kwargs: Reserved for future domain randomization parameters

    Returns:
        PhysicsParams instance
    """
    return PhysicsParams()
