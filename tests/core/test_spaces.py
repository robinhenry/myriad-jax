import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aion.core.spaces import Box, Discrete


def test_box_sample_and_contains():
    key = jax.random.PRNGKey(0)
    space = Box(low=-1.0, high=1.0, shape=(3,), dtype=jnp.float32)
    sample = space.sample(key)
    assert sample.shape == (3,)
    assert sample.dtype == jnp.float32
    assert space.contains(sample) is True


def test_discrete_sample_within_bounds():
    key = jax.random.PRNGKey(0)
    space = Discrete(5)
    sample = space.sample(key)
    assert sample.shape == ()
    assert sample.dtype == jnp.int32
    assert space.contains(sample) is True
    assert 0 <= int(np.asarray(sample)) < 5


def test_discrete_contains_rejects_invalid_values():
    space = Discrete(3)
    assert space.contains(jnp.array(2, dtype=jnp.int32)) is True
    assert space.contains(jnp.array(3, dtype=jnp.int32)) is False
    assert space.contains(jnp.array(-1, dtype=jnp.int32)) is False
    assert space.contains(jnp.array([0, 1])) is False


def test_discrete_requires_positive_n():
    with pytest.raises(ValueError):
        Discrete(0)
