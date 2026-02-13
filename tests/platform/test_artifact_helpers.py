"""Tests for artifact saving helpers."""

import pickle

import jax.numpy as jnp
import pytest

from myriad.configs.builder import create_eval_config
from myriad.platform.artifact_helpers import save_results_to_disk
from myriad.platform.constants import (
    CHECKPOINT_EXTENSION,
    CHECKPOINTS_DIR,
    FINAL_CHECKPOINT_NAME,
    RESULTS_FILENAME,
)
from myriad.platform.serialization import load_agent_state
from myriad.platform.types import EvaluationResults


def test_save_results_to_disk_without_checkpoint(tmp_path):
    """Test saving results without checkpoint."""
    # Create mock config and results object
    config = create_eval_config(env="cartpole-control", agent="random")
    results = EvaluationResults(
        mean_return=100.0,
        std_return=10.0,
        min_return=80.0,
        max_return=120.0,
        mean_length=50.0,
        std_length=5.0,
        min_length=40,
        max_length=60,
        episode_returns=jnp.array([100.0]),
        episode_lengths=jnp.array([50]),
        num_episodes=1,
        seed=42,
        config=config,
        run_dir=tmp_path,
    )

    # Save without checkpoint
    save_results_to_disk(results, tmp_path, agent_state=None, save_checkpoint=False)

    # Verify results.pkl exists
    results_path = tmp_path / RESULTS_FILENAME
    assert results_path.exists()

    # Verify no checkpoint was created
    checkpoint_dir = tmp_path / CHECKPOINTS_DIR
    assert not checkpoint_dir.exists()

    # Verify results can be loaded
    with open(results_path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded.mean_return == 100.0


def test_save_results_to_disk_with_checkpoint(tmp_path):
    """Test saving results with checkpoint."""
    # Create mock config
    config = create_eval_config(env="cartpole-control", agent="random")

    # Create mock results and agent state
    results = EvaluationResults(
        mean_return=100.0,
        std_return=10.0,
        min_return=80.0,
        max_return=120.0,
        mean_length=50.0,
        std_length=5.0,
        min_length=40,
        max_length=60,
        episode_returns=jnp.array([100.0]),
        episode_lengths=jnp.array([50]),
        num_episodes=1,
        seed=42,
        config=config,
        run_dir=tmp_path,
    )
    agent_state = {"params": {"weights": jnp.array([1.0, 2.0])}, "step": 100}

    # Save with checkpoint
    save_results_to_disk(results, tmp_path, agent_state, save_checkpoint=True)

    # Verify results.pkl exists
    assert (tmp_path / RESULTS_FILENAME).exists()

    # Verify checkpoint exists
    checkpoint_path = tmp_path / CHECKPOINTS_DIR / f"{FINAL_CHECKPOINT_NAME}{CHECKPOINT_EXTENSION}"
    assert checkpoint_path.exists()

    # Verify checkpoint can be loaded
    loaded_agent = load_agent_state(checkpoint_path)
    assert loaded_agent["step"] == 100
    assert jnp.allclose(loaded_agent["params"]["weights"], jnp.array([1.0, 2.0]))


def test_save_results_to_disk_checkpoint_flag_false_with_state(tmp_path):
    """Test that agent_state is ignored when save_checkpoint=False."""
    # Create mock config
    config = create_eval_config(env="cartpole-control", agent="random")

    results = EvaluationResults(
        mean_return=100.0,
        std_return=10.0,
        min_return=80.0,
        max_return=120.0,
        mean_length=50.0,
        std_length=5.0,
        min_length=40,
        max_length=60,
        episode_returns=jnp.array([100.0]),
        episode_lengths=jnp.array([50]),
        num_episodes=1,
        seed=42,
        config=config,
        run_dir=tmp_path,
    )
    agent_state = {"params": {}, "step": 100}

    # Save with agent_state but save_checkpoint=False
    save_results_to_disk(results, tmp_path, agent_state, save_checkpoint=False)

    # Checkpoint should NOT be created
    checkpoint_dir = tmp_path / CHECKPOINTS_DIR
    assert not checkpoint_dir.exists()


def test_save_results_to_disk_creates_directory(tmp_path):
    """Test that save_results_to_disk creates directory if it doesn't exist."""
    directory = tmp_path / "nested" / "output"

    # Create mock config
    config = create_eval_config(env="cartpole-control", agent="random")

    results = EvaluationResults(
        mean_return=100.0,
        std_return=10.0,
        min_return=80.0,
        max_return=120.0,
        mean_length=50.0,
        std_length=5.0,
        min_length=40,
        max_length=60,
        episode_returns=jnp.array([100.0]),
        episode_lengths=jnp.array([50]),
        num_episodes=1,
        seed=42,
        config=config,
        run_dir=directory,
    )

    save_results_to_disk(results, directory, agent_state=None, save_checkpoint=False)

    assert directory.exists()
    assert (directory / RESULTS_FILENAME).exists()


def test_save_results_to_disk_invalid_agent_state(tmp_path):
    """Test that invalid agent state raises RuntimeError during serialization."""
    # Create mock config
    config = create_eval_config(env="cartpole-control", agent="random")

    results = EvaluationResults(
        mean_return=100.0,
        std_return=10.0,
        min_return=80.0,
        max_return=120.0,
        mean_length=50.0,
        std_length=5.0,
        min_length=40,
        max_length=60,
        episode_returns=jnp.array([100.0]),
        episode_lengths=jnp.array([50]),
        num_episodes=1,
        seed=42,
        config=config,
        run_dir=tmp_path,
    )

    # Non-serializable agent state
    invalid_state = {"func": lambda x: x}

    with pytest.raises(RuntimeError, match="Failed to serialize agent state"):
        save_results_to_disk(results, tmp_path, invalid_state, save_checkpoint=True)
