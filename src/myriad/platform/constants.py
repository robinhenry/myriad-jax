"""Platform constants for artifact management.

Centralizes magic strings used across the platform to prevent duplication
and ensure consistency.
"""

# Checkpoint and artifact filenames
FINAL_CHECKPOINT_NAME = "final"
"""Default checkpoint name for final trained agent."""

RESULTS_FILENAME = "results.pkl"
"""Filename for serialized TrainingResults or EvaluationResults."""

METADATA_FILENAME = "run_metadata.yaml"
"""Filename for run metadata (git info, timestamps, etc.)."""

# Directory names
CHECKPOINTS_DIR = "checkpoints"
"""Directory name for storing agent checkpoints."""

# File extensions
CHECKPOINT_EXTENSION = ".msgpack"
"""File extension for Flax-serialized agent checkpoints."""

RESULTS_EXTENSION = ".pkl"
"""File extension for pickled results objects."""
