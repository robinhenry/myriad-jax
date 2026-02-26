"""W&B helper utilities for fetching and inspecting runs and sweeps.

Useful both internally (seed-eval pipeline) and interactively in notebooks.
"""

import warnings
from typing import Any

import polars as pl
import wandb  # type: ignore[import]

from myriad.configs.default import Config

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
    return Config.model_validate(normalised)


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
    sweep = wandb.Api().sweep(sweep_id)
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
