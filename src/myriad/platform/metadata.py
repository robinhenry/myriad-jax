"""Run metadata tracking for reproducibility.

Captures minimal information needed to reproduce a run:
- Timestamp
- Run type (training vs evaluation)
- Git hash (if in git repo)
- Python/JAX versions
- Duration (seconds)
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from types import TracebackType

import jax
import yaml


class RunMetadata:
    """Context manager that writes run_metadata.yaml on entry and appends duration on exit.

    Usage::

        with RunMetadata(run_dir, run_type="evaluation"):
            ...  # duration written automatically on exit, even on error
    """

    def __init__(self, run_dir: Path, run_type: str) -> None:
        self._run_dir = run_dir
        self._run_type = run_type
        self._start_time: float | None = None

    def __enter__(self) -> "RunMetadata":
        metadata = {
            "run_type": self._run_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "jax_version": jax.__version__,
        }

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            metadata["git_hash"] = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        metadata_path = self._run_dir / "run_metadata.yaml"
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

        self._start_time = monotonic()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        assert self._start_time is not None
        duration = monotonic() - self._start_time
        metadata_path = self._run_dir / "run_metadata.yaml"
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        metadata["duration_seconds"] = round(duration, 3)
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
