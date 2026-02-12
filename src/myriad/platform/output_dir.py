"""Output directory management for runs.

Provides utilities to create timestamped output directories and detect
whether running under Hydra's directory management.
"""

from datetime import datetime
from pathlib import Path


def is_hydra_run() -> bool:
    """Detect if we're running under Hydra's directory management.

    Hydra creates a .hydra subdirectory in the output directory, which we can
    use to detect its presence.

    Returns:
        True if running under Hydra, False otherwise
    """
    return (Path.cwd() / ".hydra").exists()


def create_timestamped_output_dir(base_dir: Path | str = "outputs") -> Path:
    """Create a timestamped output directory for a run.

    Creates a directory structure like: outputs/YYYY-MM-DD/HH-MM-SS/

    Args:
        base_dir: Base directory for outputs (default: "outputs")

    Returns:
        Path to the created timestamped directory

    Example:
        >>> output_dir = create_timestamped_output_dir()
        >>> print(output_dir)
        outputs/2026-02-12/14-30-52
    """
    base_dir = Path(base_dir)

    # Create timestamp-based path
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")

    output_dir = base_dir / date_str / time_str

    # Create directory (including parents)
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def get_or_create_output_dir(output_dir: Path | str | None = None) -> Path:
    """Get the output directory for a run, creating if necessary.

    Logic:
    1. If output_dir is provided, use it
    2. If running under Hydra, use current directory
    3. Otherwise, create a timestamped directory in outputs/

    Args:
        output_dir: Optional explicit output directory

    Returns:
        Path to the output directory to use

    Example:
        >>> # Under Hydra: returns Path.cwd()
        >>> # In notebook: returns outputs/2026-02-12/14-30-52/
        >>> output_dir = get_or_create_output_dir()
    """
    if output_dir is not None:
        # Explicit output directory provided
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    if is_hydra_run():
        # Under Hydra: use current directory (Hydra sets this to output dir)
        return Path.cwd()

    # Not under Hydra: create timestamped directory
    return create_timestamped_output_dir()
