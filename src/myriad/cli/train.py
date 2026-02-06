"""Training CLI command.

This module provides a Click wrapper around the Hydra-based training runner.
All arguments are passed through to Hydra for configuration management.
"""

import click


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
    # Execute the training runner with Hydra setup
    from myriad.platform.hydra_runners import train_main
    from myriad.platform.runner_utils import run_with_hydra

    run_with_hydra(train_main, script_name="myriad train", args=ctx.args)
