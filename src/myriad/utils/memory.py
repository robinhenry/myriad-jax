"""Memory estimation utilities for JAX PyTrees and arrays."""

from typing import Any

import jax
import jax.numpy as jnp


def estimate_pytree_memory_mb(pytree: Any) -> float:
    """Estimate memory usage of a PyTree in MB.

    Args:
        pytree: A JAX PyTree (nested structure of arrays)

    Returns:
        Estimated memory usage in megabytes
    """
    leaves = jax.tree_util.tree_leaves(pytree)
    total_bytes = sum(leaf.nbytes for leaf in leaves if isinstance(leaf, jnp.ndarray))
    return total_bytes / (1024 * 1024)


def get_array_memory_mb(arr: jnp.ndarray) -> float:
    """Calculate memory usage of a JAX array in megabytes.

    Args:
        arr: JAX array

    Returns:
        Memory usage in MB
    """
    return arr.nbytes / (1024 * 1024)
