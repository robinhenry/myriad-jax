"""Base inferrer definitions for system identification.

This module provides the Inferrer container — a typed, pure-functional
interface for Bayesian inference algorithms that update a posterior estimate
from observed trajectories.

The design mirrors :mod:`myriad.agents.agent`: small Protocol markers for
state and params, callback Protocols for each function, and a NamedTuple
container that holds them together.

The Inferrer is deliberately agnostic about posterior representation.
Different algorithms produce different outputs (MCMC samples, weighted
particles, neural density estimators, etc.), so the state type is left
fully generic.  Loggable summary statistics are returned via the metrics
dict in ``update``.
"""

from typing import Any, Generic, NamedTuple, Protocol, TypeVar


class InferrerParams(Protocol):
    """Marker protocol for inferrer hyperparameters.

    Concrete implementations may use dataclasses, Flax structs, NamedTuples,
    or any other pytree-compatible type.
    """

    ...


class InferrerState(Protocol):
    """Marker protocol for inferrer state.

    The state carries the current posterior representation plus any
    bookkeeping (e.g., step counters, diagnostics).  Its internal
    structure is entirely up to the implementation.
    """

    ...


# Type variables
S = TypeVar("S", bound=InferrerState)
P = TypeVar("P", bound=InferrerParams)

S_co = TypeVar("S_co", bound=InferrerState, covariant=True)
S_inv = TypeVar("S_inv", bound=InferrerState)
P_contra = TypeVar("P_contra", bound=InferrerParams, contravariant=True)


# ---------------------------------------------------------------------------
# Callback Protocols
# ---------------------------------------------------------------------------


class InitFn(Protocol[S_co, P_contra]):
    """Initialize the inferrer state from a prior.

    Parameters
    ----------
    key
        JAX PRNG key for stochastic initialization.
    params
        Inferrer hyperparameters (prior specification, algorithm settings).

    Returns
    -------
    S
        Initial inferrer state (e.g., prior samples, empty observation log).
    """

    def __call__(self, key: Any, params: P_contra) -> S_co: ...


class UpdateFn(Protocol[S_inv, P_contra]):
    """Update the inferrer state given newly observed episode data.

    The ``episodes`` argument is a dictionary of arrays collected from
    evaluation rollouts by the training loop.  It contains full episode
    trajectories with shape ``(num_rollouts, max_steps, ...)``.
    The dict typically contains keys like ``observations``, ``actions``,
    ``rewards``, ``dones``, and ``episode_lengths``.

    The inferrer is responsible for interpreting episode structure
    (boundaries, padding, time ordering, etc.).

    Parameters
    ----------
    key
        JAX PRNG key for stochastic inference (e.g., MCMC sampling).
    state
        Current inferrer state.
    episodes
        Episode data collected by the training loop.  Keys and shapes
        depend on the environment but typically include observations,
        actions, and done flags.
    params
        Inferrer hyperparameters.

    Returns
    -------
    tuple[S, dict[str, Any]]
        Updated state and a metrics dictionary (e.g., posterior entropy,
        effective sample size, acceptance rate).
    """

    def __call__(
        self,
        key: Any,
        state: S_inv,
        episodes: dict[str, Any],
        params: P_contra,
    ) -> tuple[S_inv, dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Inferrer container
# ---------------------------------------------------------------------------


class Inferrer(NamedTuple, Generic[S, P]):
    """Typed container for an inferrer's pure functions.

    Attributes
    ----------
    params
        Inferrer hyperparameters (prior, algorithm config, etc.).
    init
        Pure function to create the initial inferrer state.
    update
        Pure function to update the state from trajectory data.
    """

    params: P
    init: InitFn[S, P]
    update: UpdateFn[S, P]
