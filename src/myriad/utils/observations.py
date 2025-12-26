"""Utilities for handling different observation types."""

import jax.numpy as jnp


def to_array(obs):
    """Convert observation to array format.

    Handles different observation types:
    - Arrays (JAX/numpy): Returned as-is
    - NamedTuples with .to_array(): Converted to array
    - Other types: Attempted conversion via jnp.asarray

    Args:
        obs: Observation (array, NamedTuple, or other)

    Returns:
        Array representation of the observation

    Raises:
        ValueError: If observation cannot be converted to array
    """
    # Already an array - return as-is
    if isinstance(obs, (jnp.ndarray, type(jnp.array(0)))):
        return obs

    # Has .to_array() method (e.g., PhysicsState NamedTuple)
    if hasattr(obs, "to_array") and callable(obs.to_array):
        return obs.to_array()

    # Try generic conversion
    try:
        return jnp.asarray(obs)
    except Exception as e:
        raise ValueError(
            f"Cannot convert observation of type {type(obs)} to array. "
            f"Observation should be an array or have a .to_array() method."
        ) from e
