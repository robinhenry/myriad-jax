"""W&B sweep creation utilities.

Provides reusable functions for registering hyperparameter sweeps with W&B,
including support for multi-level sweeps (e.g. sweeping over `num_envs` values).
"""

import copy
import subprocess
import tempfile
from pathlib import Path

import yaml


def create_wandb_sweeps(
    yaml_path: str | Path,
    project: str | None = None,
    *,
    levels: list[int | float | str] | None = None,
    level_param: str = "run.num_envs",
    base_group: str | None = None,
) -> list[str]:
    """Create W&B sweeps from a YAML config, optionally at multiple parameter levels.

    Args:
        yaml_path: Path to the W&B sweep YAML file.
        project: W&B project name. If None, uses the 'project' field from the YAML.
        levels: List of integer values to sweep over for ``level_param``. If None,
            creates a single sweep without patching any parameter.
        level_param: Dotted parameter key to patch for each level, e.g. ``"run.num_envs"``.
        base_group: Base name for ``wandb.group`` when creating per-level sweeps.
            Level suffix is appended automatically: ``"{base_group}_{level}"``.
            If None, falls back to the ``wandb.group`` value already in the YAML,
            and then to the project name.

    Returns:
        List of fully-qualified sweep IDs (``entity/project/sweep_id``), one per level.
        If levels is None, returns a single-element list.

    Raises:
        ValueError: If project is not given and not found in the YAML.
        RuntimeError: If the ``wandb sweep`` subprocess fails or its output can't be parsed.

    Example::

        from myriad.platform.sweep import create_wandb_sweeps

        sweep_ids = create_wandb_sweeps(
            "sweep.yaml",
            project="my-project",
            levels=[512, 1024, 16384],
            level_param="run.num_envs",
            base_group="myexp",
        )
        # produces groups: myexp_512, myexp_1024, myexp_16384
        for sweep_id in sweep_ids:
            print(f"wandb agent {sweep_id}")
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        base_cfg = yaml.safe_load(f)

    if project is None:
        project = base_cfg.get("project")
        if project is None:
            raise ValueError("project must be specified either as an argument or in the YAML 'project' field")

    sweep_levels: list[int | None] = list(levels) if levels is not None else [None]
    sweep_ids: list[str] = []

    for level in sweep_levels:
        cfg = copy.deepcopy(base_cfg)

        if level is not None:
            params = cfg.setdefault("parameters", {})
            params[level_param] = {"value": level}

            # Determine wandb.group for this level
            if base_group is not None:
                group = f"{base_group}_{level}"
            elif "wandb.group" in params:
                group = f"{params['wandb.group']['value']}_{level}"
            else:
                group = f"{project}_{level}"

            params["wandb.group"] = {"value": group}

        sweep_id = _register_sweep(cfg, project)
        sweep_ids.append(sweep_id)

    return sweep_ids


def _register_sweep(cfg: dict, project: str) -> str:
    """Register a single sweep config with W&B and return the fully-qualified sweep ID."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(cfg, tmp)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["wandb", "sweep", "--project", project, tmp_path],
            capture_output=True,
            text=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"wandb sweep failed (exit {result.returncode}):\n{result.stderr.strip()}")

    # W&B prints "wandb agent <entity>/<project>/<sweep_id>" to stderr
    agent_lines = [line for line in result.stderr.splitlines() if "wandb agent" in line]
    if not agent_lines:
        raise RuntimeError(f"Could not parse sweep ID from wandb output:\n{result.stderr}")
    return agent_lines[-1].split()[-1]
