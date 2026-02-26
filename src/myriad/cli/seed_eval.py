"""seed-eval CLI command for statistical validation of top sweep configs."""

import click


@click.command("seed-eval")
@click.argument("sweep_id")
@click.option("--top-k", default=5, show_default=True, help="Number of top Phase 1 configs to validate.")
@click.option(
    "--seeds",
    default="0,1,2,3,4",
    show_default=True,
    help="Comma-separated list of seeds to run for each config.",
)
@click.option(
    "--metric",
    default="eval/episode_return/mean",
    show_default=True,
    help="W&B summary metric used to rank Phase 1 runs.",
)
@click.option("--group", required=True, metavar="NAME", help="Base W&B group name. Rank suffix appended automatically.")
@click.option(
    "--minimize",
    is_flag=True,
    default=False,
    help="Treat metric as a cost to minimise (default: maximise).",
)
@click.option(
    "--mode",
    default="online",
    show_default=True,
    type=click.Choice(["online", "offline", "disabled"]),
    help="W&B logging mode.",
)
@click.option("--tag", "tags", multiple=True, metavar="TAG", help="Tag to attach to every seed-eval run. Repeatable.")
def seed_eval(
    sweep_id: str,
    top_k: int,
    seeds: str,
    metric: str,
    group: str,
    minimize: bool,
    mode: str,
    tags: tuple[str, ...],
) -> None:
    """Re-run top-K sweep configs with multiple seeds for statistical validation.

    SWEEP_ID is the fully-qualified W&B sweep ID: entity/project/sweep_id.

    \b
    Example (smoke test):
      myriad seed-eval entity/project/abc123 \\
          --top-k 1 --seeds 0 --group test_seed_eval --mode disabled

    \b
    Example (full Phase 2):
      myriad seed-eval entity/myriad-ccasr/abc123 \\
          --top-k 5 --seeds 0,1,2,3,4 \\
          --metric eval/episode_return/mean \\
          --group pqn_ccasr_validated
    """
    seed_list = [int(s.strip()) for s in seeds.split(",")]

    from myriad.platform.seed_eval import run_seed_eval

    run_seed_eval(
        sweep_id,
        top_k,
        seed_list,
        metric,
        group,
        maximize=not minimize,
        mode=mode,
        tags=tags,
    )
