"""Development entry point for training.

This script is a thin wrapper around the core training runner in the myriad package.
For production use, prefer the `myriad train` CLI command.

This script is useful during development for:
- Debugging with IDE breakpoints
- Quick iteration without reinstalling the package
- Running from the project root directory
"""

from myriad.platform.hydra_runners import train_main
from myriad.platform.hydra_setup import setup_hydra

if __name__ == "__main__":
    setup_hydra()
    train_main()
