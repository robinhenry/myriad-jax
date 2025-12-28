"""Sweep CLI command for W&B hyperparameter optimization.

This module provides a Click wrapper around the Hydra-based sweep training runner.
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
def sweep(ctx: click.Context) -> None:
    """Run training with W&B sweep for hyperparameter optimization.

    This command is designed to work with W&B sweeps. It loads the base
    Hydra configuration and then overrides parameters from W&B.

    Typical workflow:

    \b
    1. Create a sweep configuration (YAML file with sweep parameters)
    2. Initialize the sweep: wandb sweep sweep_config.yaml
    3. Start sweep agents: wandb agent <sweep-id>
       (The agent will call this command automatically)

    All arguments are passed through to Hydra. Examples:

    \b
    # Run sweep with specific config
    myriad sweep --config-name=config

    \b
    # Override base parameters
    myriad sweep env.name=cartpole-control

    For W&B sweep documentation, see: https://docs.wandb.ai/guides/sweeps
    For Hydra documentation, see: https://hydra.cc/
    """
    # Reconstruct sys.argv for Hydra
    sys.argv = ["myriad sweep"] + ctx.args

    # Setup Hydra global configuration
    setup_hydra()

    # Import and run the sweep main function
    from myriad.platform.hydra_runners import sweep_main

    sweep_main()
