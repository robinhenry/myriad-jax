"""Tests for Flax-based agent state serialization."""

import jax.numpy as jnp
import pytest

from myriad.platform.serialization import (
    deserialize_agent_state,
    load_agent_state,
    save_agent_state,
    serialize_agent_state,
)


def test_serialize_deserialize_round_trip():
    """Test serialization and deserialization of agent state."""
    # Create a pytree (dict with JAX arrays)
    state = {
        "params": {"weights": jnp.array([1.0, 2.0, 3.0]), "bias": jnp.array([0.1])},
        "step": 42,
    }

    # Serialize
    data = serialize_agent_state(state)
    assert isinstance(data, bytes)

    # Deserialize
    loaded = deserialize_agent_state(data)
    assert loaded["step"] == state["step"]
    assert jnp.allclose(loaded["params"]["weights"], state["params"]["weights"])
    assert jnp.allclose(loaded["params"]["bias"], state["params"]["bias"])


def test_save_load_round_trip(tmp_path):
    """Test saving and loading agent state from file."""
    state = {
        "params": {"weights": jnp.array([1.0, 2.0, 3.0])},
        "step": 100,
    }

    path = tmp_path / "agent.msgpack"

    # Save
    save_agent_state(state, path)
    assert path.exists()

    # Load
    loaded = load_agent_state(path)
    assert loaded["step"] == state["step"]
    assert jnp.allclose(loaded["params"]["weights"], state["params"]["weights"])


def test_load_agent_state_missing_file():
    """Test that loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Agent checkpoint not found"):
        load_agent_state("/nonexistent/path.msgpack")


def test_serialize_invalid_type():
    """Test that serializing non-pytree raises RuntimeError with helpful message."""
    # Non-serializable object
    invalid_state = {"func": lambda x: x}  # Functions can't be serialized

    with pytest.raises(RuntimeError, match="Failed to serialize agent state"):
        serialize_agent_state(invalid_state)


def test_deserialize_corrupted_data():
    """Test that deserializing corrupted data raises RuntimeError."""
    corrupted_data = b"not valid msgpack data"

    with pytest.raises(RuntimeError, match="Failed to deserialize agent state"):
        deserialize_agent_state(corrupted_data)


def test_save_creates_parent_directory(tmp_path):
    """Test that save_agent_state creates parent directories."""
    path = tmp_path / "nested" / "dir" / "agent.msgpack"
    state = {"params": {}, "step": 0}

    save_agent_state(state, path)
    assert path.exists()
    assert path.parent.exists()
