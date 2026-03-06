"""sweep-create CLI command for registering W&B sweeps."""

import sys

import click


@click.command("sweep-create")
@click.argument("yaml_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--project", default=None, help="W&B project name (defaults to value in YAML).")
@click.option(
    "--level-param",
    default="run.num_envs",
    show_default=True,
    metavar="PARAM",
    help="Dotted parameter key to patch for each level (e.g. run.num_envs).",
)
@click.option(
    "--levels",
    multiple=True,
    type=str,
    metavar="VALUE",
    help=(
        "Parameter values to create separate sweeps for. Repeat for multiple: --levels 512 --levels 1024."
        "Integers and floats are parsed automatically."
    ),
)
@click.option(
    "--base-group",
    default=None,
    metavar="NAME",
    help=(
        "Base name for wandb.group when creating per-level sweeps. "
        "Level is appended automatically: '{base_group}_{level}'. "
        "Defaults to the wandb.group value in the YAML, then the project name."
    ),
)
def sweep_create(
    yaml_path: str,
    project: str | None,
    level_param: str,
    levels: tuple[str, ...],
    base_group: str | None,
) -> None:
    """Register W&B sweep(s) from YAML_PATH.

    Outputs one fully-qualified sweep ID per line on stdout for easy shell
    scripting. Human-readable status is printed to stderr.

    \b
    Examples:
      # Single sweep
      myriad sweep-create sweep.yaml --project my-project

      # One sweep per num_envs level
      myriad sweep-create sweep.yaml --project my-project --level-param run.num_envs\\
          --levels 512 --levels 1024 --levels 16384

    \b
    Typical run_sweep.sh usage:
      mapfile -t IDS < <(myriad sweep-create sweep.yaml --project $PROJECT \\
                             --levels 512 --levels 1024 --levels 16384)
      for id in "${IDS[@]}"; do wandb agent "$id" & done
      wait
    """
    from myriad.platform.sweep import create_wandb_sweeps

    def _parse(v: str) -> int | float | str:
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            return v

    level_list = [_parse(v) for v in levels] if levels else None
    n = len(level_list) if level_list else 1

    click.echo(
        f"Creating {n} sweep(s) from '{yaml_path}'"
        + (f" in project '{project}'" if project else " (project from YAML)"),
        err=True,
    )

    try:
        sweep_ids = create_wandb_sweeps(
            yaml_path,
            project,
            levels=level_list,
            level_param=level_param,
            base_group=base_group,
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    for i, sweep_id in enumerate(sweep_ids):
        level = level_list[i] if level_list else None
        label = f"  {level_param}={level:<8}" if level is not None else "  "
        click.echo(f"{label}  wandb agent {sweep_id}", err=True)
        # stdout: one sweep ID per line, for shell scripting
        print(sweep_id)
