"""Utilities for handling different observation types.

This module provides utilities for working with observations in different formats:
- to_array(): Convert observations to JAX arrays
- get_field_index(): Map field names to array indices (for NamedTuple observations)

See src/myriad/agents/bangbang.py for an example of how to use them in practice.
"""

from typing import Any

import jax.numpy as jnp
from jax import Array

from myriad.core.types import Observation


def to_array(obs: Observation | Array) -> Array:
    """Convert observation to array format.

    Handles different observation types:
    - Arrays (JAX/numpy): Returned as-is
    - Observations with .to_array(): Converted via that method
    - Other types: Attempted conversion via jnp.asarray

    Args:
        obs: Observation (satisfying the Observation Protocol) or array

    Returns:
        Array representation of the observation

    Raises:
        ValueError: If observation cannot be converted to array
    """
    # Already an array - return as-is
    if isinstance(obs, (jnp.ndarray, type(jnp.array(0)))):
        return obs  # type: ignore[return-value]

    # Has .to_array() method (satisfies Observation Protocol)
    if hasattr(obs, "to_array") and callable(obs.to_array):  # type: ignore[union-attr]
        return obs.to_array()  # type: ignore[union-attr]

    # Try generic conversion
    try:
        return jnp.asarray(obs)
    except Exception as e:
        raise ValueError(
            f"Cannot convert observation of type {type(obs)} to array. "
            f"Observation should be an array or have a .to_array() method."
        ) from e


def get_field_index(sample_obs: Any, field_name: str) -> int:
    """Get the array index for a named field in a NamedTuple observation.

    This function introspects a NamedTuple observation to find the array index
    corresponding to a named field. It should be called once during agent
    initialization and the result cached for use during action selection.

    Args:
        sample_obs: Sample observation (must be a NamedTuple or similar with ._fields)
        field_name: Name of the field to look up (e.g., "theta", "x")

    Returns:
        Index of the field in the flattened observation array

    Raises:
        ValueError: If sample_obs is not a NamedTuple with ._fields attribute
        ValueError: If field_name is not found in the observation

    Example:
        >>> from myriad.envs.cartpole.physics import PhysicsState
        >>> sample_obs = PhysicsState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)
        >>> theta_idx = get_field_index(sample_obs, "theta")
        >>> theta_idx
        2
    """
    if not hasattr(sample_obs, "_fields"):
        raise ValueError(
            f"get_field_index requires NamedTuple observations with ._fields attribute, "
            f"but got {type(sample_obs)}. "
            f"Ensure your environment returns NamedTuple observations."
        )

    field_to_index = {field: idx for idx, field in enumerate(sample_obs._fields)}

    if field_name not in field_to_index:
        available_fields = list(field_to_index.keys())
        raise ValueError(
            f"Observation field '{field_name}' not found in observation. " f"Available fields: {available_fields}"
        )

    return field_to_index[field_name]
