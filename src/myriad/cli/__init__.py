"""Myriad CLI - Command-line interface for the Myriad JAX framework.

This module provides a unified CLI with subcommands for training, evaluation,
hyperparameter sweeps, and episode rendering.

Usage:
    myriad --help              Show available commands
    myriad train --help        Show training options
    myriad evaluate --help     Show evaluation options
    myriad sweep --help        Show sweep options
    myriad render --help       Show rendering options

Examples:
    # Train an agent
    myriad train run.total_timesteps=1e6

    # Evaluate a classical controller
    myriad evaluate --config-name=experiments/eval_bangbang_cartpole

    # Run a W&B hyperparameter sweep
    myriad sweep

    # Render saved episodes to video
    myriad render episodes/ --env cartpole-control --fps 60
"""

import click

from .autotune import autotune
from .evaluate import evaluate
from .render import render
from .seed_eval import seed_eval
from .sweep import sweep
from .sweep_create import sweep_create
from .train import train


@click.group()
@click.version_option(package_name="myriad-jax")
def main() -> None:
    """Myriad: JAX-based Digital Twin Engine for Control & Decision-Making.

    Massively parallel experiments on GPU/TPU for biological/physics systems,
    active system identification, model-based, and model-free control.
    """
    pass


# Register subcommands
main.add_command(train)
main.add_command(evaluate)
main.add_command(sweep)
main.add_command(sweep_create)
main.add_command(seed_eval)
main.add_command(render)
main.add_command(autotune)


__all__ = ["main", "train", "evaluate", "sweep", "sweep_create", "seed_eval", "render", "autotune"]
