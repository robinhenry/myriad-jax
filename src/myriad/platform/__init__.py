"""Platform module for training and evaluation infrastructure."""

from .artifact_loader import (
    RunArtifacts,
    load_run,
    load_run_checkpoint,
    load_run_config,
    load_run_metadata,
    load_run_results,
)
from .constants import (
    CHECKPOINT_EXTENSION,
    CHECKPOINTS_DIR,
    FINAL_CHECKPOINT_NAME,
    METADATA_FILENAME,
    RESULTS_EXTENSION,
    RESULTS_FILENAME,
)
from .evaluation import evaluate
from .logging import SessionLogger
from .serialization import (
    deserialize_agent_state,
    load_agent_state,
    save_agent_state,
    serialize_agent_state,
)
from .training import train_and_evaluate
from .types import EvaluationMetrics, EvaluationResults, TrainingMetrics, TrainingResults

__all__ = [
    # Training and evaluation
    "train_and_evaluate",
    "evaluate",
    # Result types
    "TrainingResults",
    "TrainingMetrics",
    "EvaluationMetrics",
    "EvaluationResults",
    # Logging
    "SessionLogger",
    # Artifact loading
    "load_run",
    "load_run_config",
    "load_run_results",
    "load_run_checkpoint",
    "load_run_metadata",
    "RunArtifacts",
    # Constants
    "FINAL_CHECKPOINT_NAME",
    "CHECKPOINTS_DIR",
    "CHECKPOINT_EXTENSION",
    "RESULTS_FILENAME",
    "RESULTS_EXTENSION",
    "METADATA_FILENAME",
    # Serialization
    "save_agent_state",
    "load_agent_state",
    "serialize_agent_state",
    "deserialize_agent_state",
]
