"""Seed-evaluation: re-run top-K sweep configs with multiple seeds.

Workflow:
1. Query W&B for the top-K finished runs from a sweep.
2. Reconstruct a Config from each W&B run.
3. Re-run each config with a set of seeds, grouped in W&B for statistical reporting.

W&B dashboard result: runs appear in Groups view as
  ``{group}_rank0`` → N runs (best config × N seeds)
  ``{group}_rank1`` → N runs (2nd-best × N seeds)
  …
Filter by ``job_type = "seed-eval"`` to isolate Phase 2 runs.
"""

import warnings
from typing import Any

import wandb  # type: ignore[import]

from myriad.configs.default import Config

from .wandb_helpers import config_from_wandb_run, fetch_top_k_runs


def _make_seed_eval_config(
    base_config: Config,
    seed: int,
    rank_group: str,
    mode: str,
    tags: tuple[str, ...],
) -> Config:
    """Build a Config for one seed-eval run from a base config.

    Overrides seed, W&B group, job_type, mode, and tags. Preserves all other
    fields (project, entity, hyperparameters, etc.) from the base config.

    Args:
        base_config: Config reconstructed from a Phase 1 sweep run.
        seed: Seed for this run.
        rank_group: W&B group name (e.g. ``pqn_ccasr_validated_rank0``).
        mode: W&B mode (``"online"``, ``"offline"``, or ``"disabled"``).
        tags: Additional tags to attach to the W&B run.

    Returns:
        A new validated ``Config`` with overridden fields.
    """
    config_dict: dict[str, Any] = base_config.model_dump()

    config_dict["run"]["seed"] = seed

    wandb_dict: dict[str, Any] = config_dict.get("wandb") or {}
    wandb_dict.update(
        {
            "enabled": True,
            "group": rank_group,
            "run_name": None,  # let W&B auto-generate to avoid name collisions
            "job_type": "seed-eval",
            "mode": mode,
            "tags": list(tags),
        }
    )
    config_dict["wandb"] = wandb_dict

    return Config.model_validate(config_dict)


def run_seed_eval(
    sweep_id: str,
    top_k: int,
    seeds: list[int],
    metric: str,
    group: str,
    *,
    maximize: bool,
    mode: str,
    tags: tuple[str, ...],
) -> None:
    """Re-run top-K sweep configs with multiple seeds for statistical validation.

    Runs are grouped in W&B as ``{group}_rank{rank}`` with ``job_type="seed-eval"``.

    Args:
        sweep_id: Fully-qualified W&B sweep ID (``entity/project/sweep_id``).
        top_k: Number of top runs to validate.
        seeds: List of seeds to run for each config.
        metric: W&B summary metric used to rank Phase 1 runs.
        group: Base W&B group name. Rank suffix is appended automatically.
        maximize: If True, higher metric values are better (default: True).
        mode: W&B logging mode (``"online"``, ``"offline"``, ``"disabled"``).
        tags: Tags to attach to every seed-eval run.
    """
    from myriad.platform.training import train_and_evaluate

    top_runs = fetch_top_k_runs(sweep_id, metric, top_k, maximize=maximize)

    for rank, wandb_run in enumerate(top_runs):
        base_config = config_from_wandb_run(wandb_run)
        rank_group = f"{group}_rank{rank}"

        for seed in seeds:
            # Guard against stale runs (e.g. interrupted previous iteration)
            if wandb.run is not None:
                warnings.warn(
                    "Stale W&B run detected before starting seed-eval iteration; closing it.",
                    UserWarning,
                    stacklevel=1,
                )
                wandb.finish()

            config = _make_seed_eval_config(base_config, seed, rank_group, mode, tags)
            train_and_evaluate(config)
            # session_logger.finalize() calls wandb.finish() → wandb.run becomes None
