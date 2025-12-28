"""Evaluation CLI command.

This module provides a Click wrapper around the Hydra-based evaluation runner.
All arguments are passed through to Hydra for configuration management.
"""

import sys

import click

from myriad.platform.hydra_setup import setup_hydra


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "allow_interspersed_args": False,
    }
)
@click.pass_context
def evaluate(ctx: click.Context) -> None:
    """Run evaluation (no training) using Hydra configuration.

    Use this for:
    - Classical controllers (random, bang-bang, PID)
    - Pre-trained models
    - Baseline comparisons
    - Debugging and visualization

    All arguments are passed through to Hydra. Examples:

    \b
    # Run an evaluation config
    myriad evaluate --config-name=experiments/eval_bangbang_cartpole

    \b
    # Override parameters
    myriad evaluate --config-name=experiments/eval_bangbang_cartpole eval_rollouts=100

    \b
    # Evaluate with video rendering
    myriad evaluate run.eval_render_videos=true run.eval_episode_save_frequency=10

    For full Hydra documentation, see: https://hydra.cc/
    """
    # Reconstruct sys.argv for Hydra
    sys.argv = ["myriad evaluate"] + ctx.args

    # Setup Hydra global configuration
    setup_hydra()

    # Import and run the evaluation main function
    from myriad.platform.hydra_runners import evaluate_main

    evaluate_main()
