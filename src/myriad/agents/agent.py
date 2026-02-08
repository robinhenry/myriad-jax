"""Base agent definitions for JAX-based agents.

This module provides small, focused Protocols for the two agent
components (params, state) and a typed container `Agent` which
holds the agent's pure functions. The Protocols are intentionally small
and permissive so concrete environments remain free to use dataclasses,
Flax structs, NamedTuples, etc., while still providing helpful static typing
and documentation.
"""

from typing import Any, Generic, NamedTuple, Protocol, TypeVar

from jax import Array

from myriad.core.spaces import Space
from myriad.core.types import Observation, PRNGKey


class AgentParams(Protocol):
    """Protocol for agent parameter objects.

    Concrete agents can use dataclasses, Flax structs, or simple NamedTuples.
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
Obs = TypeVar("Obs", bound=Observation)

# Variance-specific type variables for Protocol definitions
S_co = TypeVar("S_co", bound=AgentState, covariant=True)
S_inv = TypeVar("S_inv", bound=AgentState)
P_contra = TypeVar("P_contra", bound=AgentParams, contravariant=True)
Obs_contra = TypeVar("Obs_contra", bound=Observation, contravariant=True)


# ---------------------------------------------------------------------------
# Callback Protocols for Agent functions
# ---------------------------------------------------------------------------


class InitFn(Protocol[S_co, P_contra, Obs_contra]):
    """Initialize the agent's state.

    Parameters
    ----------
    key
        JAX PRNG key for stochastic initialization (e.g., network weights)
    sample_obs
        Sample observation to infer network architecture and field names
    params
        Agent hyperparameters (learning rate, network architecture, etc.)

    Returns
    -------
    S
        Initialized agent state (e.g., network parameters, optimizer state)
    """

    def __call__(self, key: PRNGKey, sample_obs: Obs_contra, params: P_contra) -> S_co: ...


class SelectActionFn(Protocol[S_inv, P_contra, Obs_contra]):
    """Select an action given the current observation.

    Parameters
    ----------
    key
        JAX PRNG key for stochastic action selection (e.g., epsilon-greedy)
    obs
        Current observation from the environment
    state
        Current agent state (e.g., network parameters)
    params
        Agent hyperparameters
    deterministic
        If True, select the greedy/deterministic action (e.g., for evaluation).
        If False, sample from the policy distribution (e.g., for exploration).

    Returns
    -------
    tuple[Array, S]
        Selected action and (possibly updated) agent state
    """

    def __call__(
        self,
        key: PRNGKey,
        obs: Obs_contra,
        state: S_inv,
        params: P_contra,
        deterministic: bool,
    ) -> tuple[Array, S_inv]: ...


class UpdateFn(Protocol[S_inv, P_contra]):
    """Update the agent's state from a batch of experience.

    Parameters
    ----------
    key
        JAX PRNG key for stochastic updates (e.g., dropout, minibatch sampling)
    state
        Current agent state to update
    batch
        Batch of experience data (structure depends on the agent/algorithm)
    params
        Agent hyperparameters

    Returns
    -------
    tuple[S, dict[str, Any]]
        Updated agent state and a metrics dictionary (e.g., loss values)
    """

    def __call__(
        self,
        key: PRNGKey,
        state: S_inv,
        batch: Any,
        params: P_contra,
    ) -> tuple[S_inv, dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Agent container
# ---------------------------------------------------------------------------


class Agent(NamedTuple, Generic[S, P, Obs]):
    """Typed container for a JAX-friendly agent's pure functions.

    Attributes
    ----------
    params
        Agent hyperparameters (learning rate, network config, action space, etc.).
    init
        Pure function to initialize the agent's state.
    select_action
        Pure function to select an action from the agent's policy.
    update
        Pure function to update the agent's state from experience.
    """

    params: P
    init: InitFn[S, P, Obs]
    select_action: SelectActionFn[S, P, Obs]
    update: UpdateFn[S, P]
