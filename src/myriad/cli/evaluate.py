"""Evaluation CLI command.

This module provides a Click wrapper around the Hydra-based evaluation runner.
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
    # Execute the evaluation runner with Hydra setup
    from myriad.platform.hydra_runners import evaluate_main
    from myriad.platform.runner_utils import run_with_hydra

    run_with_hydra(evaluate_main, script_name="myriad evaluate", args=ctx.args)
