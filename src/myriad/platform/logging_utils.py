from __future__ import annotations

from typing import Any

import jax
import numpy as np

from myriad.configs.default import Config

try:
    import wandb  # type: ignore[import]
except ImportError as import_error:  # pragma: no cover - handled at runtime
    wandb = None  # type: ignore[assignment]
    _wandb_import_error = import_error
else:
    _wandb_import_error = None


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    """Removes items with None values from a dictionary."""

    return {key: value for key, value in values.items() if value is not None}


def prepare_metrics_host(metrics_history: Any, steps_this_chunk: int) -> dict[str, Any]:
    """Transform scan metrics to host arrays, keeping only the active steps."""

    if not isinstance(metrics_history, dict) or not metrics_history or steps_this_chunk <= 0:
        return {}

    sliced_history = {name: values[:steps_this_chunk] for name, values in metrics_history.items()}
    return {name: jax.device_get(values) for name, values in sliced_history.items()}


def summarize_metric(prefix: str, name: str, value: Any) -> dict[str, float]:
    """Expands an array-like metric into scalar statistics for logging."""

    try:
        array = np.asarray(value)
    except Exception:  # pragma: no cover - defensive guard
        return {}

    if array.size == 0:
        return {}

    if array.dtype == np.bool_:
        array = array.astype(np.float32)

    if not np.issubdtype(array.dtype, np.number):
        return {}

    array = np.asarray(array, dtype=np.float64)

    if array.ndim == 0:
        try:
            scalar = float(array.item())
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            return {}
        return {f"{prefix}{name}": scalar}

    mean = float(np.nanmean(array))
    std = float(np.nanstd(array))

    return {
        f"{prefix}{name}/mean": mean,
        f"{prefix}{name}/std": std,
        f"{prefix}{name}/mean-std": mean - std,
        f"{prefix}{name}/mean+std": mean + std,
        f"{prefix}{name}/max": float(np.nanmax(array)),
        f"{prefix}{name}/min": float(np.nanmin(array)),
    }


def build_train_payload(metrics_host: dict[str, Any]) -> dict[str, float]:
    """Aggregates training metrics into scalar summaries suitable for W&B."""

    payload: dict[str, float] = {}
    for name, history in metrics_host.items():
        if hasattr(history, "__getitem__") and len(history) > 0:
            value = history[-1]
            payload.update(summarize_metric("train/", name, value))
    return payload


def maybe_init_wandb(config: Config):
    """Initializes a Weights & Biases run when enabled in the config."""

    wandb_config = config.wandb
    if not wandb_config.enabled:
        return None

    if wandb is None:
        message = (
            "Weights & Biases tracking is enabled but the `wandb` package is not installed. "
            "Install it with `pip install wandb` to proceed."
        )
        raise RuntimeError(message) from _wandb_import_error

    init_kwargs: dict[str, Any] = _drop_none(
        {
            "project": wandb_config.project,
            "entity": wandb_config.entity,
            "group": wandb_config.group,
            "job_type": wandb_config.job_type,
            "mode": wandb_config.mode,
            "dir": wandb_config.dir,
        }
    )

    if wandb_config.run_name:
        init_kwargs["name"] = wandb_config.run_name

    if wandb_config.tags:
        init_kwargs["tags"] = list(wandb_config.tags)

    init_kwargs["config"] = config.model_dump()

    return wandb.init(**init_kwargs)
