"""Shared artifact saving utilities.

Provides common logic for saving results and checkpoints to disk,
used by both TrainingResults and EvaluationResults.
"""

import pickle
from pathlib import Path
from typing import Any

from .constants import CHECKPOINTS_DIR, FINAL_CHECKPOINT_NAME, RESULTS_FILENAME
from .serialization import save_agent_state


def save_results_to_disk(
    results_object: Any,
    directory: Path,
    agent_state: Any | None,
    save_checkpoint: bool,
) -> None:
    """Save results and optionally agent checkpoint to directory.

    This is shared logic extracted from TrainingResults.save() and EvaluationResults.save().

    Args:
        results_object: Results object to pickle (should have agent_state=None already)
        directory: Directory to save to (will be created if needed)
        agent_state: Agent state to save as checkpoint (if save_checkpoint=True)
        save_checkpoint: Whether to save agent checkpoint

    Raises:
        RuntimeError: If agent checkpoint serialization fails
    """
    directory.mkdir(parents=True, exist_ok=True)

    # Save results object (already has agent_state=None)
    with open(directory / RESULTS_FILENAME, "wb") as f:
        pickle.dump(results_object, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save agent checkpoint using Flax serialization
    if save_checkpoint and agent_state is not None:
        checkpoint_dir = directory / CHECKPOINTS_DIR
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{FINAL_CHECKPOINT_NAME}.msgpack"
        save_agent_state(agent_state, checkpoint_path)
