"""Shared utilities for the logging system."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np


def prepare_metrics_host(metrics_history: dict, steps_this_chunk: int) -> dict[str, Any]:
    """Transform scan metrics to host arrays, keeping only the active steps.

    Args:
        metrics_history: Dictionary of metric arrays from the training loop
        steps_this_chunk: Number of active steps in this chunk

    Returns:
        Dictionary of host numpy arrays sliced to active steps
    """
    if steps_this_chunk <= 0:
        return {}

    sliced_history = {name: values[:steps_this_chunk] for name, values in metrics_history.items()}
    return {name: jax.device_get(values) for name, values in sliced_history.items()}


def summarize_metric(prefix: str, name: str, value: Any) -> dict[str, float]:
    """Expand an array-like metric into scalar statistics for logging.

    Args:
        prefix: Prefix for metric names (e.g., "train/", "eval/")
        name: Base name of the metric
        value: Metric value (scalar or array)

    Returns:
        Dictionary of scalar statistics (mean, std, min, max for arrays)
    """
    try:
        array = np.asarray(value)
    except Exception:
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
        except (TypeError, ValueError):
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
    """Aggregate training metrics into scalar summaries for remote logging.

    Args:
        metrics_host: Dictionary of host numpy arrays

    Returns:
        Flattened dictionary of scalar metrics with "train/" prefix
    """
    payload: dict[str, float] = {}
    for name, history in metrics_host.items():
        if hasattr(history, "__getitem__") and len(history) > 0:
            value = history[-1]
            payload.update(summarize_metric("train/", name, value))
    return payload
