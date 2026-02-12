"""Agent state serialization using Flax msgpack format.

Provides utilities for serializing and deserializing agent states (typically
Flax optimizer states and neural network parameters) using Flax's msgpack
serialization. This is more reliable than pickle for JAX/Flax objects.

All functions raise RuntimeError with clear messages on failure.
"""

from pathlib import Path
from typing import Any

from flax import serialization


def serialize_agent_state(agent_state: Any) -> bytes:
    """Serialize agent state to msgpack bytes.

    Args:
        agent_state: Agent state to serialize (typically Flax TrainState or similar)

    Returns:
        Serialized bytes

    Raises:
        RuntimeError: If serialization fails
    """
    try:
        return serialization.msgpack_serialize(agent_state)
    except Exception as e:
        raise RuntimeError(
            f"Failed to serialize agent state. Ensure the agent state contains only "
            f"JAX/Flax types (pytrees, arrays, etc.). Original error: {e}"
        ) from e


def deserialize_agent_state(data: bytes) -> Any:
    """Deserialize agent state from msgpack bytes.

    Args:
        data: Msgpack-serialized bytes

    Returns:
        Deserialized agent state

    Raises:
        RuntimeError: If deserialization fails
    """
    try:
        return serialization.msgpack_restore(data)
    except Exception as e:
        raise RuntimeError(
            f"Failed to deserialize agent state. The data may be corrupted or " f"incompatible. Original error: {e}"
        ) from e


def save_agent_state(agent_state: Any, path: str | Path) -> None:
    """Serialize and save agent state to file.

    Args:
        agent_state: Agent state to save
        path: File path (typically with .msgpack extension)

    Raises:
        RuntimeError: If serialization or file writing fails
    """
    path = Path(path)
    try:
        data = serialize_agent_state(agent_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except RuntimeError:
        # Re-raise serialization errors as-is
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to write agent state to {path}. Check file permissions and " f"disk space. Original error: {e}"
        ) from e


def load_agent_state(path: str | Path) -> Any:
    """Load and deserialize agent state from file.

    Args:
        path: File path to load from

    Returns:
        Deserialized agent state

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If deserialization fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Agent checkpoint not found: {path}")

    try:
        data = path.read_bytes()
        return deserialize_agent_state(data)
    except RuntimeError:
        # Re-raise deserialization errors as-is
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to read agent state from {path}. The file may be corrupted. " f"Original error: {e}"
        ) from e
