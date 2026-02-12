"""Loading API for run artifacts.

Provides utilities to load configs, results, checkpoints, and metadata from
completed training/evaluation runs.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml
from omegaconf import OmegaConf

from myriad.configs.default import Config, EvalConfig
from myriad.platform.types import EvaluationResults, TrainingResults

from .constants import (
    CHECKPOINT_EXTENSION,
    CHECKPOINTS_DIR,
    FINAL_CHECKPOINT_NAME,
    METADATA_FILENAME,
    RESULTS_FILENAME,
)
from .serialization import load_agent_state

# Generic type variables for RunArtifacts
ConfigT = TypeVar("ConfigT", bound=Config | EvalConfig)
ResultsT = TypeVar("ResultsT", bound=TrainingResults | EvaluationResults)


def load_run_config(run_path: str | Path) -> Config | EvalConfig:
    """Load config from run directory.

    Loads from .hydra/config.yaml and validates with Pydantic. Requires
    run_metadata.yaml to determine config type.

    Args:
        run_path: Path to run directory

    Returns:
        Config or EvalConfig depending on run type

    Raises:
        FileNotFoundError: If config.yaml or run_metadata.yaml not found
        RuntimeError: If run_type field missing from metadata

    Example:
        >>> config = load_run_config("outputs/2026-02-12/14-30-52")
        >>> print(config.run.seed)
    """
    run_path = Path(run_path)
    config_path = run_path / ".hydra" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml found in {run_path}/.hydra/")

    # Load config using OmegaConf
    cfg = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_object(cfg)

    # Load metadata to determine run type (mandatory)
    metadata = load_run_metadata(run_path)
    if "run_type" not in metadata:
        raise RuntimeError(
            f"Missing 'run_type' field in {run_path}/{METADATA_FILENAME}. "
            f"Cannot determine whether to load Config or EvalConfig."
        )

    run_type = metadata["run_type"]

    # Validate with appropriate Pydantic model
    if run_type == "training":
        return Config.model_validate(config_dict)
    else:
        return EvalConfig.model_validate(config_dict)


def load_run_results(run_path: str | Path) -> TrainingResults | EvaluationResults:
    """Load results from run directory.

    Args:
        run_path: Path to run directory

    Returns:
        TrainingResults or EvaluationResults

    Example:
        >>> results = load_run_results("outputs/2026-02-12/14-30-52")
        >>> print(results.summary())
    """
    run_path = Path(run_path)
    results_path = run_path / RESULTS_FILENAME

    if not results_path.exists():
        raise FileNotFoundError(f"No {RESULTS_FILENAME} found in {run_path}")

    with open(results_path, "rb") as f:
        return pickle.load(f)


def load_run_checkpoint(
    run_path: str | Path,
    checkpoint: str = FINAL_CHECKPOINT_NAME,
) -> Any:
    """Load agent checkpoint from run directory.

    Args:
        run_path: Path to run directory
        checkpoint: Checkpoint name (default: "final")

    Returns:
        Agent state from checkpoint

    Raises:
        FileNotFoundError: If checkpoint file not found
        RuntimeError: If deserialization fails

    Example:
        >>> agent_state = load_run_checkpoint("outputs/2026-02-12/14-30-52")
        >>> # Use with evaluate()
        >>> results = evaluate(config, agent_state=agent_state)
    """
    run_path = Path(run_path)
    checkpoint_path = run_path / CHECKPOINTS_DIR / f"{checkpoint}{CHECKPOINT_EXTENSION}"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint '{checkpoint}' in {run_path}/{CHECKPOINTS_DIR}/")

    return load_agent_state(checkpoint_path)


def load_run_metadata(run_path: str | Path) -> dict:
    """Load run metadata from run directory.

    Args:
        run_path: Path to run directory

    Returns:
        Dictionary with metadata (run_type, timestamp, git_hash, versions)

    Raises:
        FileNotFoundError: If metadata file not found

    Example:
        >>> metadata = load_run_metadata("outputs/2026-02-12/14-30-52")
        >>> print(metadata["git_hash"])
    """
    run_path = Path(run_path)
    metadata_path = run_path / METADATA_FILENAME

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No {METADATA_FILENAME} in {run_path}. Run metadata is required "
            f"to determine run type and configuration."
        )

    with open(metadata_path) as f:
        return yaml.safe_load(f)


@dataclass
class RunArtifacts(Generic[ConfigT, ResultsT]):
    """Container for all artifacts from a run.

    Provides a unified interface to access configs, results, metadata,
    and optionally load checkpoints.

    Type parameters:
        ConfigT: Config or EvalConfig
        ResultsT: TrainingResults or EvaluationResults
    """

    config: ConfigT
    """Configuration used for this run."""

    results: ResultsT
    """Results from the run."""

    metadata: dict
    """Run metadata (timestamp, git hash, versions)."""

    run_path: Path
    """Path to the run directory."""

    def load_checkpoint(self, checkpoint: str = FINAL_CHECKPOINT_NAME) -> Any:
        """Load agent checkpoint from disk.

        Always loads fresh from disk (no caching).

        Args:
            checkpoint: Checkpoint name (default: "final")

        Returns:
            Agent state from checkpoint

        Raises:
            FileNotFoundError: If checkpoint file not found
            RuntimeError: If deserialization fails
        """
        return load_run_checkpoint(self.run_path, checkpoint)


def load_run(run_path: str | Path) -> RunArtifacts:
    """Load all artifacts from a run directory.

    This is the main entry point for loading runs. It loads config, results,
    and metadata in one call. Agent checkpoints can be loaded on demand.

    Args:
        run_path: Path to run directory

    Returns:
        RunArtifacts container with all run data

    Example:
        >>> run = load_run("outputs/2026-02-12/14-30-52")
        >>> print(f"Final return: {run.results.summary()['mean_return']}")
        >>> agent = run.load_checkpoint()  # Lazy load if needed
    """
    run_path = Path(run_path)

    return RunArtifacts(
        config=load_run_config(run_path),
        results=load_run_results(run_path),
        metadata=load_run_metadata(run_path),
        run_path=run_path,
    )
