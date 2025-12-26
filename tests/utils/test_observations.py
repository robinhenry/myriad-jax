"""Tests for observation utility functions."""

import jax.numpy as jnp
import pytest

from myriad.envs.cartpole.physics import PhysicsState
from myriad.utils.observations import get_field_index


# Test get_field_index
def test_get_field_index_valid_field():
    """Test getting index for a valid field name."""
    sample_obs = PhysicsState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)

    assert get_field_index(sample_obs, "x") == 0
    assert get_field_index(sample_obs, "x_dot") == 1
    assert get_field_index(sample_obs, "theta") == 2
    assert get_field_index(sample_obs, "theta_dot") == 3


def test_get_field_index_invalid_field():
    """Test error when field name doesn't exist."""
    sample_obs = PhysicsState(x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0)

    with pytest.raises(ValueError, match="Observation field 'invalid_field' not found"):
        get_field_index(sample_obs, "invalid_field")

    # Check that error message includes available fields
    with pytest.raises(ValueError, match="Available fields"):
        get_field_index(sample_obs, "nonexistent")


def test_get_field_index_not_namedtuple():
    """Test error when observation is not a NamedTuple."""
    obs_array = jnp.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="requires NamedTuple observations with ._fields"):
        get_field_index(obs_array, "x")


def test_get_field_index_dict_not_supported():
    """Test error when observation is a dict (not a NamedTuple)."""
    obs_dict = {"x": 0.0, "y": 1.0}

    with pytest.raises(ValueError, match="requires NamedTuple observations"):
        get_field_index(obs_dict, "x")
