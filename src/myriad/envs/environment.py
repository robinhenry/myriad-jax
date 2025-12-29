"""Base environment definitions for JAX-based RL environments.

This module provides small, focused Protocols for the three environment
components (config, params, state) and a typed container `Environment` which
holds the environment's pure functions. The Protocols are intentionally small
and permissive so concrete environments remain free to use dataclasses,
Flax structs, NamedTuples, etc., while still providing helpful static typing
and documentation.

Design Rationale: Config vs Params
-----------------------------------
Environments separate static configuration (EnvConfig) from dynamic parameters
(EnvParams) to optimize JAX compilation:

- **EnvConfig**: Static, compile-time configuration passed as `static_argnames`
  to `jax.jit`. Changes trigger recompilation but enable better optimization.
  Use for: physics constants, termination thresholds, max_steps, environment
  structure.

- **EnvParams**: Dynamic, runtime parameters that can vary between episodes
  without recompilation. Use for: randomized dynamics, curriculum learning
  parameters, domain randomization values.

If your environment doesn't need dynamic parameters, EnvParams can be empty
(see CartPole for an example), but keep the structure for protocol consistency.
"""

from typing import Callable, Generic, NamedTuple, Protocol, Tuple, TypeVar

import chex

from myriad.core.spaces import Space


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

    If your environment doesn't need runtime variation, this can be empty
    (e.g., CartPole), but maintain the structure for protocol consistency.

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


class Environment(NamedTuple, Generic[S, P, C]):
    """Typed container for a JAX-friendly environment's pure functions.

    This container bundles the environment's configuration, parameters, and
    core functions. All functions should be pure and JIT-compatible.

    Fields
    ------
    params : P (EnvironmentParams)
        Dynamic parameters that can vary between runs without recompilation.
        Passed as a regular (non-static) argument to step/reset functions.

    config : C (EnvironmentConfig)
        Static configuration used as `static_argnames` when jitting functions.
        Changes to config require recompilation.

    get_action_space : Callable[[C], Space]
        Returns the action space (takes only config since it's structural).

    get_obs_shape : Callable[[C], Tuple]
        Returns the observation shape (takes only config since it's structural).

    reset : Callable[[chex.PRNGKey, P, C], tuple[chex.Array, S]]
        Pure reset function returning (initial_obs, initial_state).
        Should be jitted with config as static_argnames.

    step : Callable[[chex.PRNGKey, S, chex.Array, P, C], tuple[chex.Array, S, chex.Array, chex.Array, dict]]
        Pure step function returning (next_obs, next_state, reward, done, info).
        Should be jitted with config as static_argnames.

    Example
    -------
    When jitting environment functions, use::

        step = jax.jit(_step, static_argnames=["config"])
        reset = jax.jit(_reset, static_argnames=["config"])

    This allows params to vary without recompilation while keeping config static.
    """

    params: P
    config: C

    # Action / observation helpers
    get_action_space: Callable[[C], Space]
    get_obs_shape: Callable[[C], Tuple]

    # Pure, jitted environment functions
    reset: Callable[[chex.PRNGKey, P, C], Tuple[chex.Array, S]]
    step: Callable[
        [chex.PRNGKey, S, chex.Array, P, C],
        Tuple[chex.Array, S, chex.Array, chex.Array, dict],
    ]
