"""Development entry point for W&B sweeps.

This script is a thin wrapper around the core sweep runner in the myriad package.
For production use, prefer the `myriad sweep` CLI command with W&B agents.

This script is useful during development for:
- Debugging with IDE breakpoints
- Quick iteration without reinstalling the package
- Running from the project root directory
"""

from myriad.platform.hydra_runners import sweep_main
from myriad.platform.runner_utils import run_with_hydra

if __name__ == "__main__":
    run_with_hydra(sweep_main)
