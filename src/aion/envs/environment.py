"""Base environment definitions for JAX-based RL environments.

This module provides small, focused Protocols for the three environment
components (config, params, state) and a typed container `Environment` which
holds the environment's pure functions. The Protocols are intentionally small
and permissive so concrete environments remain free to use dataclasses,
Flax structs, NamedTuples, etc., while still providing helpful static typing
and documentation.
"""

from typing import Callable, Generic, NamedTuple, Protocol, Tuple, TypeVar

import chex

from aion.core.spaces import Space


class EnvironmentConfig(Protocol):
    """Protocol for environment configuration objects.

    Implementations must provide at least a `max_steps: int` attribute so
    training loops can determine episode length. Environments may add other
    fields.
    """

    max_steps: int


class EnvironmentParams(Protocol):
    """Protocol for environment parameter objects.

    Concrete envs can use dataclasses, Flax structs, or simple NamedTuples.
    This Protocol is intentionally empty — it's a structural marker used only
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

    Fields
    ------
    params
        Dynamic parameters for the environment (may change between runs).
    config
        Static configuration for the environment (used as a static arg when
        jitting functions).
    get_action_space
        Returns the action space.
    get_obs_shape
        Returns the observation shape given the config.
    reset: Callable[[chex.PRNGKey, P, C], tuple[chex.Array, S]]
        Pure reset function; should be jittable and accept a PRNGKey.
    step: Callable[[chex.PRNGKey, S, chex.Array, P, C], tuple[chex.Array, S, chex.Array, chex.Array, dict]]
        Pure step function; should be jittable and accept a PRNGKey.
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
