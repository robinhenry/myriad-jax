from typing import NamedTuple, Protocol

import jax
from pydantic import BaseModel as _BaseModel, ConfigDict

# Type aliases
PRNGKey = jax.Array


class Observation(Protocol):
    """Protocol for structured observation pytrees.

    Observations are typically NamedTuples with named fields that can be
    converted to flat arrays for neural network input. The `to_array()` method
    enables agents to work with observations in either structured (for field
    introspection) or flattened (for network input) form.
    """

    def to_array(self) -> jax.Array:
        """Convert observation to a flat JAX array."""
        ...


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    done: jax.Array


class BaseModel(_BaseModel):
    """Pydantic BaseModel subclass to be used throughout the codebase."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ...
