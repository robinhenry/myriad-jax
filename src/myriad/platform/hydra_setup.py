"""Global Hydra configuration setup.

This module provides common Hydra setup that should be applied to all scripts.
Import and call setup_hydra() before calling @hydra.main decorated functions.
"""

import sys


def setup_hydra() -> None:
    """Apply global Hydra configuration overrides.

    This ensures consistent behavior across all scripts:
    - Always change to output directory (chdir=true) for clean artifact organization
    - Can still be overridden from command line if needed
    """
    # Ensure Hydra always changes to output directory
    if "hydra.job.chdir" not in " ".join(sys.argv):
        sys.argv.append("hydra.job.chdir=true")
