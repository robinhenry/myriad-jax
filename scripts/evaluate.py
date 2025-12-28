"""Development entry point for evaluation.

This script is a thin wrapper around the core evaluation runner in the myriad package.
For production use, prefer the `myriad evaluate` CLI command.

This script is useful during development for:
- Debugging with IDE breakpoints
- Quick iteration without reinstalling the package
- Running from the project root directory
"""

from myriad.platform.hydra_runners import evaluate_main
from myriad.platform.hydra_setup import setup_hydra

if __name__ == "__main__":
    setup_hydra()
    evaluate_main()
