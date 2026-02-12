"""Run metadata tracking for reproducibility.

Captures minimal information needed to reproduce a run:
- Timestamp
- Run type (training vs evaluation)
- Git hash (if in git repo)
- Python/JAX versions
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax
import yaml


def create_and_save_run_metadata(
    run_dir: Path,
    run_type: str,  # "training" or "evaluation"
) -> None:
    """Create and save run metadata to run_dir/run_metadata.yaml.

    Captures minimal reproducibility info:
    - Timestamp
    - Run type
    - Git hash (if in git repo)
    - Python/JAX versions

    Args:
        run_dir: Directory to save metadata to (typically Hydra output directory)
        run_type: Type of run ("training" or "evaluation")
    """
    metadata = {
        "run_type": run_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "jax_version": jax.__version__,
    }

    # Try to get git info (optional - don't fail if not in repo)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        metadata["git_hash"] = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # Not in git repo or git not available

    # Save to standard location
    metadata_path = run_dir / "run_metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
