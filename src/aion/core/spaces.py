"""JAX-friendly space definitions for RL environments."""

from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp


class Space:
    """Base class for all spaces."""

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        """Sample a random value from the space."""
        raise NotImplementedError

    def contains(self, x: chex.Array) -> bool:
        """Check if x is a valid value in this space."""
        raise NotImplementedError


class Box(Space):
    """A box in R^n with a bounds per dimension."""

    def __init__(
        self,
        low: float | chex.Array,
        high: float | chex.Array,
        shape: Tuple[int, ...] = (),
        dtype: Any = jnp.float32,
    ):
        self.low = jnp.asarray(low, dtype=dtype)
        self.high = jnp.asarray(high, dtype=dtype)
        self.shape = shape if shape else self.low.shape
        self.dtype = dtype

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        """Sample uniformly from the box."""
        return jnp.asarray(
            jax.random.uniform(key, shape=self.shape, minval=self.low, maxval=self.high, dtype=self.dtype)
        )

    def contains(self, x: chex.Array) -> bool:
        """Check if x is within bounds."""
        return bool(jnp.all(x >= self.low) and jnp.all(x <= self.high))


class Discrete(Space):
    """A finite set of integer actions {0, 1, ..., n-1}."""

    def __init__(self, n: int, dtype: Any = jnp.int32):
        if n <= 0:
            raise ValueError("Discrete space size must be positive")
        self.n = int(n)
        self.dtype = dtype
        self.shape: Tuple[int, ...] = ()

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        """Sample uniformly from the discrete set."""
        return jax.random.randint(key, shape=self.shape, minval=0, maxval=self.n, dtype=self.dtype)

    def contains(self, x: chex.Array) -> bool:
        """Check if x is a valid discrete value."""
        value = jnp.asarray(x)
        if value.ndim != 0:
            return False
        return bool((value >= 0) & (value < self.n))
