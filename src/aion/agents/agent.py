"""Base agent definitions for JAX-based agents.

This module provides small, focused Protocols for the three environment
components (config, params, state) and a typed container `Environment` which
holds the environment's pure functions. The Protocols are intentionally small
and permissive so concrete environments remain free to use dataclasses,
Flax structs, NamedTuples, etc., while still providing helpful static typing
and documentation.
"""

from typing import Any, Callable, Generic, NamedTuple, Protocol, TypeVar

import chex

from aion.core.spaces import Space


class AgentParams(Protocol):
    """Protocol for agent parameter objects.

    Concrete agents can use dataclasses, Flax structs, or simple NamedTuples.
    This Protocol is intentionally empty — it's a structural marker used only
    for type consistency in the `Agent` container.
    """

    action_space: Space


class AgentState(Protocol):
    """Protocol for agent state objects.

    As with `AgentParams`, this is a marker Protocol. A state should be
    something JAX can transform (e.g., a NamedTuple or a pytree-compatible
    dataclass), but the Protocol leaves that choice to the implementation.
    """

    ...


# Type variables bound to the small Protocols above
S = TypeVar("S", bound=AgentState)
P = TypeVar("P", bound=AgentParams)


class Agent(NamedTuple, Generic[S, P]):
    """Typed container for a JAX-friendly agent's pure functions.

    Fields
    ------
    params
        Agent hyper parameters

    """

    params: P

    init: Callable[[chex.PRNGKey, chex.Array, P], S]
    select_action: Callable[[chex.PRNGKey, chex.Array, S, P], tuple[chex.Array, S]]
    update: Callable[[chex.PRNGKey, S, Any, P], tuple[S, dict]]
