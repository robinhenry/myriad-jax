"""Base environment definitions for JAX-based RL environments.

This module provides small, focused Protocols for the three environment
components (config, params, state) and a typed container `Environment` which
holds the environment's pure functions. The Protocols are intentionally small
and permissive so concrete environments remain free to use dataclasses,
Flax structs, NamedTuples, etc., while still providing helpful static typing
and documentation.

Design Rationale: Config vs Params
-----------------------------------
Environments separate static configuration (EnvironmentConfig) from dynamic
parameters (EnvironmentParams) to optimize JAX compilation:

- **EnvironmentConfig**: Static, compile-time configuration passed as `static_argnames`
  to `jax.jit`. Changes trigger recompilation but enable better optimization.
  Use for: physics constants, termination thresholds, max_steps, environment
  structure.

- **EnvParams**: Dynamic, runtime parameters that can vary between episodes
  without recompilation. Use for: randomized dynamics, curriculum learning
  parameters, domain randomization values, etc.

If your environment doesn't need dynamic parameters, EnvParams can be empty,
but keep the structure for protocol consistency.
"""

from typing import Any, Generic, NamedTuple, Protocol, TypeVar

from jax import Array

from myriad.core.spaces import Space
from myriad.core.types import Observation, PRNGKey


class EnvironmentConfig(Protocol):
    """Protocol for environment configuration objects.

    **Static, compile-time configuration**: Pass as `static_argnames` to `jax.jit`.
    Changing these values requires recompilation but enables better optimization.

    Required attributes:
        max_steps: Maximum steps per episode (required for training loops)

    Typical fields:
        - Physics constants (gravity, mass, friction coefficients)
        - Environment structure (grid size, number of agents)
        - Termination thresholds
        - Integration timestep (dt)
        - Any parameter that defines "what kind of environment this is"

    Implementation: Use `@struct.dataclass` from Flax for JAX compatibility.
    """

    @property
    def max_steps(self) -> int: ...


class EnvironmentParams(Protocol):
    """Protocol for environment parameter objects.

    **Dynamic, runtime parameters**: NOT passed as static args to `jax.jit`.
    Can vary between episodes/runs without triggering recompilation.

    Use cases:
        - Domain randomization (randomized dynamics, varying targets)
        - Curriculum learning (difficulty parameters that change over training)
        - Multi-task learning (task-specific parameters)
        - Any parameter you want to sweep/randomize frequently

    If your environment doesn't need runtime variation, this can be empty,
    but maintain the structure for protocol consistency.

    Implementation: Use `@struct.dataclass` from Flax for JAX compatibility.

    Note: This is intentionally an empty Protocol — it's a structural marker
    for type consistency in the `Environment` container.
    """

    ...


class EnvironmentState(Protocol):
    """Protocol for environment state objects.

    As with `EnvironmentParams`, this is a marker Protocol. A state should be
    something JAX can transform (e.g., a NamedTuple or a pytree-compatible
    dataclass), but the Protocol leaves that choice to the implementation.
    """

    ...


# Type variables bound to the small Protocols above
S = TypeVar("S", bound=EnvironmentState)
P = TypeVar("P", bound=EnvironmentParams)
C = TypeVar("C", bound=EnvironmentConfig)
Obs = TypeVar("Obs", bound=Observation)

# Variance-specific type variables for Protocol definitions
S_co = TypeVar("S_co", bound=EnvironmentState, covariant=True)
S_inv = TypeVar("S_inv", bound=EnvironmentState)
P_contra = TypeVar("P_contra", bound=EnvironmentParams, contravariant=True)
C_contra = TypeVar("C_contra", bound=EnvironmentConfig, contravariant=True)
Obs_co = TypeVar("Obs_co", bound=Observation, covariant=True)


# ---------------------------------------------------------------------------
# Callback Protocols for Environment functions
# ---------------------------------------------------------------------------


class GetActionSpaceFn(Protocol[C_contra]):
    """Return the environment's action space.

    Parameters
    ----------
    config
        Environment configuration (structural info like action dimensions)

    Returns
    -------
    Space
        The action space specification
    """

    def __call__(self, config: C_contra) -> Space: ...


class GetObsShapeFn(Protocol[C_contra]):
    """Return the shape of observations produced by the environment.

    Parameters
    ----------
    config
        Environment configuration (structural info like state dimensions)

    Returns
    -------
    tuple
        Shape tuple for observations (e.g., (4,) for CartPole)
    """

    def __call__(self, config: C_contra) -> tuple: ...


class ResetFn(Protocol[S_co, P_contra, C_contra, Obs_co]):
    """Reset the environment to an initial state.

    Parameters
    ----------
    key
        JAX PRNG key for stochastic initialization
    params
        Dynamic environment parameters
    config
        Static environment configuration

    Returns
    -------
    tuple[Obs, S]
        Initial observation and initial environment state
    """

    def __call__(self, key: PRNGKey, params: P_contra, config: C_contra) -> tuple[Obs_co, S_co]: ...


class StepFn(Protocol[S_inv, P_contra, C_contra, Obs_co]):
    """Advance the environment by one timestep.

    Parameters
    ----------
    key
        JAX PRNG key for stochastic transitions
    state
        Current environment state
    action
        Action to execute
    params
        Dynamic environment parameters
    config
        Static environment configuration

    Returns
    -------
    tuple[Obs, S, Array, Array, dict]
        - next_obs: Observation after the transition
        - next_state: Updated environment state
        - reward: Scalar reward signal
        - done: Boolean termination flag
        - info: Auxiliary information dictionary
    """

    def __call__(
        self,
        key: PRNGKey,
        state: S_inv,
        action: Array,
        params: P_contra,
        config: C_contra,
    ) -> tuple[Obs_co, S_inv, Array, Array, dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Environment container
# ---------------------------------------------------------------------------


class Environment(NamedTuple, Generic[S, C, P, Obs]):
    """Typed container for a JAX-friendly environment's pure functions.

    Attributes
    ----------
    config
        Static configuration used as `static_argnames` when jitting functions.
        Changes to config require recompilation.
    params
        Dynamic parameters that can vary between runs without recompilation.
        Passed as a regular (non-static) argument to step/reset functions.
    get_action_space
        Pure function returning the action space specification.
    get_obs_shape
        Pure function returning the observation shape tuple.
    reset
        Pure function to reset the environment to an initial state.
    step
        Pure function to advance the environment by one timestep.

    Example
    -------
    When jitting environment functions, use::

        step = jax.jit(_step, static_argnames=["config"])
        reset = jax.jit(_reset, static_argnames=["config"])

    This allows params to vary without recompilation while keeping config static.
    """

    config: C
    params: P

    # Action / observation helpers
    get_action_space: GetActionSpaceFn[C]
    get_obs_shape: GetObsShapeFn[C]

    # Pure, jitted environment functions
    reset: ResetFn[S, P, C, Obs]
    step: StepFn[S, P, C, Obs]
