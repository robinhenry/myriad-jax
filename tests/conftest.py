import chex
import jax.random
import pytest


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.key(0)
