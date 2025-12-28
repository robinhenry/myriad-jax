#!/usr/bin/env python
"""Development entry point for rendering episodes.

This script is a thin wrapper around the core render CLI in the myriad package.
For production use, prefer the `myriad render` CLI command.

This script is useful during development for:
- Debugging with IDE breakpoints
- Quick iteration without reinstalling the package
- Running from the project root directory

Usage:
    # Render a single episode
    python scripts/render_episodes.py episodes/step_1000000/episode_0.npz

    # Render all episodes in a directory
    python scripts/render_episodes.py episodes/step_1000000/

    # Render with custom output directory and FPS
    python scripts/render_episodes.py episodes/ --output-dir videos/ --fps 60

    # Upload rendered videos to W&B
    python scripts/render_episodes.py episodes/ --wandb-project my-project --wandb-run-id abc123
"""

from myriad.cli.render import render

if __name__ == "__main__":
    render()
