"""Training CLI command.

This module provides a Click wrapper around the Hydra-based training runner.
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
def train(ctx: click.Context) -> None:
    """Train an agent using Hydra configuration.

    All arguments are passed through to Hydra. Examples:

    \b
    # Basic training run
    myriad train

    \b
    # Override configuration parameters
    myriad train run.total_timesteps=1e6

    \b
    # Use a specific config file
    myriad train --config-name=experiments/my_experiment

    \b
    # Override multiple parameters
    myriad train env.name=cartpole-control agent.learning_rate=3e-4

    For full Hydra documentation, see: https://hydra.cc/
    """
    # Reconstruct sys.argv for Hydra
    # Hydra expects sys.argv[0] to be the script name
    sys.argv = ["myriad train"] + ctx.args

    # Setup Hydra global configuration
    setup_hydra()

    # Import and run the training main function
    # Import here to avoid issues with Hydra decorator at module level
    from myriad.platform.hydra_runners import train_main

    train_main()
