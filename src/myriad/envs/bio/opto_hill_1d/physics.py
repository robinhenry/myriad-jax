"""Pure stateless physics for the 1D optogenetic circuit (opto_hill_1d).

Minimal single-species stochastic gene circuit driven by a continuous light input.
Intended as a transparent sandbox for system-identification experiments — everything
about the dynamics is controlled by four kinetic parameters and one continuous input.

System Description:
    Light Input (U ∈ [0, 1]) → X (fluorescent protein copy number)

    Two Chemical Reactions:
    1. Light-driven production: ∅ → X   (rate: k_prod · hill(U, K, n))
    2. Protein degradation:     X → ∅   (rate: k_deg · X)

    Steady state at constant U: ⟨X⟩_ss = k_prod · hill(U, K, n) / k_deg
"""

import math
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from myriad.core.types import PRNGKey
from myriad.envs.bio.gillespie import step_gillespie_interval
from myriad.physics import hill_function, sample_lognormal


class PhysicsState(NamedTuple):
    """Pure physical state of the 1D optogenetic circuit.

    Attributes:
        time: Current simulation time (minutes)
        X: Fluorescent protein copy number (molecules, integer-valued)
        next_reaction_time: Scheduled time of next reaction (minutes).
            Set to inf when no reaction is pending (sample fresh).
            Preserved across RL step boundaries for physical accuracy.
    """

    time: Array
    X: Array
    next_reaction_time: Array

    def to_array(self) -> Array:
        """Convert to flat array for NN-based agents.

        Note: next_reaction_time is excluded as it's internal bookkeeping.

        Returns:
            Array of shape (2,) with [time, X]
        """
        return jnp.stack([self.time, self.X])

    @classmethod
    def from_array(cls, arr: Array) -> "PhysicsState":
        """Create state from flat array.

        Args:
            arr: Array of shape (2,) with [time, X]

        Returns:
            PhysicsState instance (next_reaction_time defaults to inf)
        """
        chex.assert_shape(arr, (2,))
        return cls(
            time=arr[0],  # type: ignore
            X=arr[1],  # type: ignore
            next_reaction_time=jnp.array(jnp.inf),
        )

    @classmethod
    def create(
        cls,
        time: Array,
        X: Array,
        next_reaction_time: Array | None = None,
    ) -> "PhysicsState":
        """Factory method to create PhysicsState with default next_reaction_time."""
        if next_reaction_time is None:
            next_reaction_time = jnp.array(jnp.inf)
        return cls(time=time, X=X, next_reaction_time=next_reaction_time)


@struct.dataclass
class PhysicsConfig:
    """Static structural constants for the 1D optogenetic circuit.

    Passed as static_argnames to jit. Kinetic parameters (k_prod, K, n, k_deg)
    belong in PhysicsParams because they vary between cells and are the targets
    of system identification.
    """

    timestep_minutes: float = 5.0
    max_gillespie_steps: int = 10_000


@struct.dataclass
class PhysicsParams:
    """Dynamic kinetic parameters — vmappable over parameter space.

    Defaults yield a steady state near X ≈ k_prod / k_deg = 100 molecules
    at full light (U=1) and half-max response at U=0.5.
    """

    k_prod: float | Array = 5.0  # Max production rate at full light (1/min)
    K: float | Array = 0.5  # Hill half-max for light intensity (dimensionless)
    n: float | Array = 2.0  # Hill cooperativity coefficient
    k_deg: float | Array = 0.05  # First-order degradation rate (1/min)


@struct.dataclass
class PhysicsParamsPrior:
    """Log-normal prior over kinetic parameters.

    Each parameter p is sampled as p ~ exp(Normal(loc, scale)). With scale=0
    the distribution collapses to a point mass at exp(loc), so the default
    (all scales zero) is fully deterministic and backward compatible.
    """

    k_prod_loc: float | Array = math.log(5.0)
    k_prod_scale: float | Array = 0.0
    K_loc: float | Array = math.log(0.5)
    K_scale: float | Array = 0.0
    n_loc: float | Array = math.log(2.0)
    n_scale: float | Array = 0.0
    k_deg_loc: float | Array = math.log(0.05)
    k_deg_scale: float | Array = 0.0

    def sample(self, key: PRNGKey) -> PhysicsParams:
        k_kprod, k_K, k_n, k_kdeg = jax.random.split(key, 4)
        return PhysicsParams(
            k_prod=sample_lognormal(k_kprod, self.k_prod_loc, self.k_prod_scale),
            K=sample_lognormal(k_K, self.K_loc, self.K_scale),
            n=sample_lognormal(k_n, self.n_loc, self.n_scale),
            k_deg=sample_lognormal(k_kdeg, self.k_deg_loc, self.k_deg_scale),
        )


def compute_propensities(
    state: PhysicsState,
    action: Array,
    params: PhysicsParams,
) -> Array:
    """Compute reaction propensities for the two reactions.

    Args:
        state: Current physical state (time, X)
        action: Continuous light intensity U ∈ [0, 1]
        params: Kinetic parameters — vmappable

    Returns:
        Array of shape (2,) with propensities [r_production, r_degradation]
    """
    U = action
    r_prod = params.k_prod * hill_function(U, params.K, params.n)
    r_deg = params.k_deg * state.X
    return jnp.array([r_prod, r_deg])


def apply_reaction(state: PhysicsState, reaction_idx: Array) -> PhysicsState:
    """Apply a single reaction to update the state.

    Uses jax.lax.switch for JAX-compatible control flow.

    Args:
        state: Current physical state
        reaction_idx: Index of reaction to apply (0 = production, 1 = degradation)

    Returns:
        Updated physical state after reaction
    """

    def production(s: PhysicsState) -> PhysicsState:
        return s._replace(X=s.X + 1)

    def degradation(s: PhysicsState) -> PhysicsState:
        return s._replace(X=jnp.maximum(s.X - 1, 0))

    return jax.lax.switch(reaction_idx, [production, degradation], state)


def step_physics(
    key: PRNGKey,
    state: PhysicsState,
    action: Array,
    params: PhysicsParams,
    config: PhysicsConfig,
    previous_action: Array,
    interval_start: Array,
) -> PhysicsState:
    """Pure physics step using the Gillespie SSA.

    Runs Gillespie simulation from current time until the end of the current
    interval (``interval_start + timestep_minutes``).

    Args:
        key: RNG key for stochastic simulation
        state: Current physical state
        action: Continuous light intensity U ∈ [0, 1]
        params: Dynamic parameters
        config: Static physics constants
        previous_action: Action from previous timestep. If different from action,
            the pending reaction time is invalidated (propensities changed).
        interval_start: Start time of current interval (``t * timestep_minutes``).

    Returns:
        Next physical state after simulating until interval end
    """
    return step_gillespie_interval(
        key,
        state,
        action,
        params,
        config,
        compute_propensities_fn=compute_propensities,
        apply_reaction_fn=apply_reaction,
        previous_action=previous_action,
        interval_start=interval_start,
    )
