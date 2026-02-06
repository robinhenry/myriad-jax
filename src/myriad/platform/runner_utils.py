"""Utilities for running Hydra-based entry points.

This module provides helpers to bridge Click/scripts and Hydra, ensuring
consistent setup and configuration across all entry points.
"""

import sys
from typing import Callable

from myriad.platform.hydra_setup import setup_hydra


def run_with_hydra(
    runner_fn: Callable[[], None],
    script_name: str | None = None,
    args: list[str] | None = None,
) -> None:
    """Run a Hydra-decorated function with proper setup.

    Args:
        runner_fn: The @hydra.main decorated function to run.
        script_name: Optional name for the script (used for sys.argv[0]).
        args: Optional list of command line arguments (overrides sys.argv).
    """
    # If args are provided (from Click), reconstruct sys.argv for Hydra
    if args is not None:
        sys.argv = [script_name or "myriad"] + list(args)

    # Apply global setup (logging, environment variables, etc.)
    setup_hydra()

    # Execute the Hydra runner
    runner_fn()
