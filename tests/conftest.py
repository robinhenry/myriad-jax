from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest


class MockObs(NamedTuple):
    """Minimal observation for agent tests.

    Fields typed as Any to accept both scalars and arrays.
    """

    x: jax.Array | float = 0.0
    x_dot: jax.Array | float = 0.0
    theta: jax.Array | float = 0.0
    theta_dot: jax.Array | float = 0.0


@pytest.fixture
def make_obs():
    """Factory fixture for creating mock observations with jnp.asarray conversion."""

    def _make(**kwargs) -> MockObs:
        defaults = {f: 0.0 for f in MockObs._fields}
        defaults.update(kwargs)
        return MockObs(**{k: jnp.asarray(v) for k, v in defaults.items()})

    return _make


@pytest.fixture
def key() -> jax.Array:
    return jax.random.key(0)
