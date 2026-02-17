"""Run metadata tracking for reproducibility.

Captures minimal information needed to reproduce a run:
- Timestamp
- Run type (training vs evaluation)
- Git hash (if in git repo)
- Python/JAX versions
- Device backend (cpu/gpu/tpu), architecture, model, and count
- Duration (seconds)
"""

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from types import TracebackType

import jax
import yaml


def _get_detailed_device_info() -> str:
    """Get detailed CPU/machine information on a best-effort basis.

    Tries platform-specific commands to get detailed hardware info,
    falls back to generic platform information.
    """
    # Try macOS sysctl for CPU brand string
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        )
        brand = result.stdout.strip()
        if brand:
            return brand
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try Linux /proc/cpuinfo
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (FileNotFoundError, PermissionError):
        pass

    # Fallback to platform.processor()
    proc = platform.processor()
    if proc:
        return proc

    # Last resort: just return architecture
    return platform.machine()


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

        # Capture device information
        devices = jax.devices()
        backend = jax.default_backend()
        metadata["device_backend"] = backend
        metadata["device_count"] = len(devices)
        metadata["device_architecture"] = platform.machine()

        # For CPU backend, get detailed CPU info
        # For GPU/TPU, JAX's device_kind usually includes the model
        if backend == "cpu":
            metadata["device_model"] = _get_detailed_device_info()
        elif devices:
            metadata["device_model"] = devices[0].device_kind

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
