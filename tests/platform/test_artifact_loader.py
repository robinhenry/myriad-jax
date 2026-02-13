"""Tests for artifact loading functionality."""

import pickle
from pathlib import Path

import jax.numpy as jnp
import pytest
import yaml

from myriad.configs.builder import create_eval_config
from myriad.platform.artifact_loader import (
    RunArtifacts,
    load_run,
    load_run_checkpoint,
    load_run_config,
    load_run_metadata,
    load_run_results,
)
from myriad.platform.constants import (
    CHECKPOINT_EXTENSION,
    CHECKPOINTS_DIR,
    FINAL_CHECKPOINT_NAME,
    METADATA_FILENAME,
    RESULTS_FILENAME,
)
from myriad.platform.serialization import save_agent_state
from myriad.platform.types import EvaluationResults


def create_mock_run_directory(tmpdir: Path, run_type: str = "training") -> Path:
    """Helper to create a mock run directory with all artifacts."""
    run_dir = tmpdir / "mock_run"
    run_dir.mkdir()

    # Create .hydra/config.yaml
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir()
    config_data = {
        "env": {"name": "cartpole-control"},
        "agent": {"name": "random"},
        "run": {"seed": 42, "eval_rollouts": 10, "eval_max_steps": 200},
    }
    with open(hydra_dir / "config.yaml", "w") as f:
        yaml.dump(config_data, f)

    # Create run_metadata.yaml
    metadata = {
        "run_type": run_type,
        "timestamp": "2026-02-12T10:00:00",
        "git_hash": "abc123",
    }
    with open(run_dir / METADATA_FILENAME, "w") as f:
        yaml.dump(metadata, f)

    # Create results.pkl
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
        episode_returns=jnp.array([100.0, 110.0]),
        episode_lengths=jnp.array([50, 55]),
        num_episodes=2,
        seed=42,
        config=config,
        run_dir=run_dir,
    )
    with open(run_dir / RESULTS_FILENAME, "wb") as f:
        pickle.dump(results, f)

    # Create checkpoint
    checkpoint_dir = run_dir / CHECKPOINTS_DIR
    checkpoint_dir.mkdir()
    agent_state = {"params": {"weights": jnp.array([1.0, 2.0])}, "step": 100}
    checkpoint_path = checkpoint_dir / f"{FINAL_CHECKPOINT_NAME}{CHECKPOINT_EXTENSION}"
    save_agent_state(agent_state, checkpoint_path)

    return run_dir


def test_load_run_metadata(tmp_path):
    """Test loading run metadata."""
    run_dir = create_mock_run_directory(tmp_path)
    metadata = load_run_metadata(run_dir)

    assert metadata["run_type"] == "training"
    assert metadata["git_hash"] == "abc123"


def test_load_run_metadata_missing_file(tmp_path):
    """Test that missing metadata file raises FileNotFoundError with helpful message."""
    with pytest.raises(FileNotFoundError, match="Run metadata is required"):
        load_run_metadata(tmp_path)


def test_load_run_config_requires_metadata(tmp_path):
    """Test that load_run_config requires metadata to exist."""
    # Create config but no metadata
    hydra_dir = tmp_path / ".hydra"
    hydra_dir.mkdir()
    with open(hydra_dir / "config.yaml", "w") as f:
        yaml.dump({"env": {"name": "test"}, "agent": {"name": "random"}}, f)

    # Should raise because metadata is missing
    with pytest.raises(FileNotFoundError, match="Run metadata is required"):
        load_run_config(tmp_path)


def test_load_run_config_missing_run_type(tmp_path):
    """Test that load_run_config raises if run_type missing from metadata."""
    # Create config
    hydra_dir = tmp_path / ".hydra"
    hydra_dir.mkdir()
    config_data = {
        "env": {"name": "cartpole-control"},
        "agent": {"name": "random"},
        "run": {"seed": 42, "eval_rollouts": 10, "eval_max_steps": 200},
    }
    with open(hydra_dir / "config.yaml", "w") as f:
        yaml.dump(config_data, f)

    # Create metadata WITHOUT run_type
    with open(tmp_path / METADATA_FILENAME, "w") as f:
        yaml.dump({"timestamp": "2026-02-12"}, f)

    # Should raise because run_type is missing
    with pytest.raises(RuntimeError, match="Missing 'run_type' field"):
        load_run_config(tmp_path)


def test_load_run_checkpoint(tmp_path):
    """Test loading agent checkpoint."""
    run_dir = create_mock_run_directory(tmp_path)
    checkpoint = load_run_checkpoint(run_dir)

    assert checkpoint["step"] == 100
    assert jnp.allclose(checkpoint["params"]["weights"], jnp.array([1.0, 2.0]))


def test_load_run_checkpoint_missing_file(tmp_path):
    """Test that missing checkpoint raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        load_run_checkpoint(tmp_path)


def test_load_run_results(tmp_path):
    """Test loading results."""
    run_dir = create_mock_run_directory(tmp_path)
    results = load_run_results(run_dir)

    assert isinstance(results, EvaluationResults)
    assert results.mean_return == 100.0
    assert results.num_episodes == 2


def test_load_run_complete(tmp_path):
    """Test load_run loads all artifacts."""
    run_dir = create_mock_run_directory(tmp_path, run_type="evaluation")
    artifacts = load_run(run_dir)

    assert isinstance(artifacts, RunArtifacts)
    assert artifacts.config.run.seed == 42
    assert artifacts.results.mean_return == 100.0
    assert artifacts.metadata["run_type"] == "evaluation"
    assert artifacts.run_path == run_dir


def test_run_artifacts_load_checkpoint_no_caching(tmp_path):
    """Test that load_checkpoint always loads fresh (no caching)."""
    run_dir = create_mock_run_directory(tmp_path, run_type="evaluation")
    artifacts = load_run(run_dir)

    # Load checkpoint twice
    checkpoint1 = artifacts.load_checkpoint()
    checkpoint2 = artifacts.load_checkpoint()

    # Both should have the same values
    assert checkpoint1["step"] == checkpoint2["step"]
    assert jnp.allclose(checkpoint1["params"]["weights"], checkpoint2["params"]["weights"])

    # Importantly, they should be separate objects (not cached)
    # Modify checkpoint1 and verify checkpoint2 is unaffected
    checkpoint1["step"] = 999
    checkpoint3 = artifacts.load_checkpoint()
    assert checkpoint3["step"] == 100  # Original value, not 999


def test_run_artifacts_generics(tmp_path):
    """Test that RunArtifacts maintains type information through generics."""
    run_dir = create_mock_run_directory(tmp_path, run_type="evaluation")
    artifacts = load_run(run_dir)

    # Should have correct types (this is more of a type checker test,
    # but we can at least verify runtime behavior)
    assert hasattr(artifacts.config, "run")
    assert hasattr(artifacts.results, "mean_return")
    assert isinstance(artifacts.metadata, dict)
