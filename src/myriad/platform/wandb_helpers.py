"""W&B helper utilities for fetching and inspecting runs and sweeps.

Useful both internally (seed-eval pipeline) and interactively in notebooks.
"""

import warnings
from typing import Any

import polars as pl
import wandb  # type: ignore[import]

from myriad.configs.default import Config, WandbConfig

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _unwrap_wandb_value(obj: Any) -> Any:
    """Recursively unwrap W&B ``{"value": x}`` wrappers and drop ``_``-prefixed keys."""
    if isinstance(obj, dict):
        if list(obj.keys()) == ["value"]:
            return _unwrap_wandb_value(obj["value"])
        return {k: _unwrap_wandb_value(v) for k, v in obj.items() if not k.startswith("_")}
    return obj


def _unflatten_dotted_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Convert flat dot-separated keys into a nested dict.

    W&B sweep agents store hyperparameters with dotted keys (e.g. ``agent.lr``)
    rather than nested dicts.  This undoes that flattening so Pydantic can
    validate the result as a ``Config``.

    Already-nested values are merged in place, so a mix of flat and nested keys
    is handled correctly.
    """
    out: dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(".")
        node = out
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        leaf = parts[-1]
        # If value is itself a nested dict, recurse so inner dotted keys are also handled.
        node[leaf] = _unflatten_dotted_keys(value) if isinstance(value, dict) else value
    return out


# ---------------------------------------------------------------------------
# ID resolution
# ---------------------------------------------------------------------------


def _resolve_sweep_id(sweep_id: str) -> str:
    """Ensure a sweep ID is fully qualified as ``entity/project/sweep_id``.

    W&B's API requires the full three-part path.  If ``sweep_id`` is already
    fully qualified (contains exactly two ``/``), it is returned unchanged.
    If it is a bare ID (no ``/``), the current entity's projects are searched
    until a matching sweep is found.

    Args:
        sweep_id: A bare sweep ID (e.g. ``"abc123"``) or a fully-qualified
            path (``"entity/project/abc123"``).

    Returns:
        Fully-qualified sweep ID as ``entity/project/sweep_id``.

    Raises:
        ValueError: If the ID cannot be resolved to any known sweep.
    """
    if sweep_id.count("/") == 2:
        return sweep_id
    if sweep_id.count("/") != 0:
        raise ValueError(f"Ambiguous sweep ID '{sweep_id}'. " "Use the fully-qualified form: entity/project/sweep_id.")

    api = wandb.Api()
    entity = api.default_entity
    for project in api.projects(entity):
        candidate = f"{entity}/{project.name}/{sweep_id}"
        try:
            api.sweep(candidate)
            return candidate
        except Exception:
            continue

    raise ValueError(
        f"Could not find sweep '{sweep_id}' in any project for entity '{entity}'. "
        "Pass the fully-qualified form: entity/project/sweep_id."
    )


# ---------------------------------------------------------------------------
# Single-run helpers
# ---------------------------------------------------------------------------


def fetch_run(run_id: str) -> Any:
    """Fetch a single W&B run by its fully-qualified ID.

    Args:
        run_id: Fully-qualified run ID (``entity/project/run_id``).

    Returns:
        A ``wandb.Run`` object.
    """
    return wandb.Api().run(run_id)


def config_from_wandb_run(run: Any) -> Config:
    """Reconstruct a Config from a W&B run object.

    W&B stores the full ``model_dump()`` nested dict in ``run.config``.
    Filters W&B-internal metadata and unwraps sweep param wrappers before
    passing to ``Config.model_validate``.

    Args:
        run: A ``wandb.Run`` object (from e.g. ``wandb.Api().run(...)``).

    Returns:
        A validated ``Config`` instance.
    """
    raw: dict[str, Any] = dict(run.config)
    normalised = _unwrap_wandb_value(raw)
    assert isinstance(normalised, dict)
    nested = _unflatten_dotted_keys(normalised)
    config = Config.model_validate(nested)

    # The wandb section is intentionally stripped from run.config by _to_flat_config
    # (it's run metadata, not experiment config). Restore project and entity from the
    # run object so that seed-eval writes back to the same project.
    if config.wandb is None:
        config = config.model_copy(update={"wandb": WandbConfig(project=run.project, entity=run.entity)})
    else:
        updates: dict[str, Any] = {}
        if config.wandb.project is None:
            updates["project"] = run.project
        if config.wandb.entity is None:
            updates["entity"] = run.entity
        if updates:
            config = config.model_copy(update={"wandb": config.wandb.model_copy(update=updates)})

    return config


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------


def fetch_sweep_runs(sweep_id: str, *, state: str | None = None) -> list[Any]:
    """Fetch runs from a W&B sweep, optionally filtered by state.

    Args:
        sweep_id: Fully-qualified sweep ID (``entity/project/sweep_id``).
        state: If provided, only return runs with this state (e.g. ``"finished"``,
            ``"running"``, ``"crashed"``). If ``None``, return all runs.

    Returns:
        List of ``wandb.Run`` objects.
    """
    sweep = wandb.Api().sweep(_resolve_sweep_id(sweep_id))
    if state is None:
        return list(sweep.runs)
    return [r for r in sweep.runs if r.state == state]


def fetch_top_k_runs(
    sweep_id: str,
    metric: str,
    top_k: int,
    *,
    maximize: bool,
) -> list[Any]:
    """Return the top-K finished runs from a W&B sweep, sorted by metric.

    Args:
        sweep_id: Fully-qualified sweep ID (``entity/project/sweep_id``).
        metric: W&B summary metric name to rank by (e.g. ``eval/episode_return/mean``).
        top_k: Number of top runs to return.
        maximize: If True, sort descending (higher is better). If False, ascending.

    Returns:
        List of ``wandb.Run`` objects, length ≤ ``top_k``.
    """
    finished = fetch_sweep_runs(sweep_id, state="finished")

    if len(finished) < top_k:
        warnings.warn(
            f"Requested top-{top_k} runs but only {len(finished)} finished runs exist "
            f"in sweep '{sweep_id}'. Returning all {len(finished)}.",
            UserWarning,
            stacklevel=2,
        )

    def _sort_key(run: Any) -> tuple[bool, float]:
        val = run.summary.get(metric)
        if val is None:
            return (True, 0.0)  # missing values sort last
        return (False, -float(val) if maximize else float(val))

    finished.sort(key=_sort_key)
    return finished[:top_k]


# ---------------------------------------------------------------------------
# DataFrame helper (notebook-friendly)
# ---------------------------------------------------------------------------


def runs_to_dataframe(runs: list[Any], metrics: list[str] | None = None) -> pl.DataFrame:
    """Convert a list of W&B runs to a Polars DataFrame.

    Each row corresponds to one run. Config fields are flattened with dot-separated
    keys (e.g. ``agent.lr``). Summary metrics are included as-is.

    Args:
        runs: List of ``wandb.Run`` objects.
        metrics: If provided, include only these summary metric keys. If ``None``,
            include all summary keys that don't start with ``_``.

    Returns:
        A ``polars.DataFrame`` with one row per run.
    """
    rows = []
    for run in runs:
        row: dict[str, Any] = {"run_id": run.id, "run_name": run.name, "state": run.state}

        config = _unwrap_wandb_value(dict(run.config))
        row.update(_flatten_dict(config, sep="."))

        summary = {k: v for k, v in run.summary.items() if not k.startswith("_") and (metrics is None or k in metrics)}
        row.update(summary)
        rows.append(row)

    return pl.DataFrame(rows)


def _flatten_dict(d: dict[str, Any], *, sep: str = ".", prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict with ``sep``-joined keys."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        full_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, sep=sep, prefix=full_key))
        else:
            out[full_key] = v
    return out
