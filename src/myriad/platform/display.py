"""Terminal display formatting for training and evaluation runs."""

import jax

from myriad.configs.default import Config, EvalConfig

from .metadata import _get_detailed_device_info


def _fmt_fields(model: object) -> str:
    """Format non-default fields of a Pydantic model as 'key=value | key=value'."""
    from pydantic import BaseModel

    if not isinstance(model, BaseModel):
        return str(model)
    non_defaults = model.model_dump(exclude_defaults=True)
    if hasattr(model, "name") and "name" not in non_defaults:
        non_defaults = {"name": model.name} | non_defaults  # type: ignore[attr-defined]
    return " | ".join(f"{k}={v}" for k, v in non_defaults.items()) if non_defaults else "(defaults)"


def _fmt_device_info() -> str:
    """Format device backend and model for display (e.g. 'cuda | A100 x1')."""
    backend = jax.default_backend()
    devices = jax.devices()
    if backend == "cpu":
        model = _get_detailed_device_info()
    elif devices:
        model = devices[0].device_kind
    else:
        model = "unknown"
    return f"{backend} | {model} x{len(devices)}"


def format_train_config(config: "Config") -> str:
    """Format a training config summary for terminal output."""
    wandb_status = "disabled" if (config.wandb is None or not config.wandb.enabled) else _fmt_fields(config.wandb)
    lines = [
        f"Training {config.agent.name} on {config.env.name}",
        f"  Agent : {_fmt_fields(config.agent)}",
        f"  Env   : {_fmt_fields(config.env)}",
        f"  Run   : {_fmt_fields(config.run)}",
        f"  W&B   : {wandb_status}",
        f"  Device: {_fmt_device_info()}",
    ]
    return "\n".join(lines)


def format_eval_config(config: "EvalConfig", config_path: str | None = None) -> str:
    """Format an evaluation config summary for terminal output."""
    wandb_status = "disabled" if (config.wandb is None or not config.wandb.enabled) else _fmt_fields(config.wandb)
    lines = [
        f"Evaluating {config.agent.name} on {config.env.name}",
        f"  Agent : {_fmt_fields(config.agent)}",
        f"  Env   : {_fmt_fields(config.env)}",
        f"  Run   : {_fmt_fields(config.run)}",
        f"  W&B   : {wandb_status}",
        f"  Device: {_fmt_device_info()}",
    ]
    if config_path is not None:
        lines.append(f"  Config: {config_path}")
    return "\n".join(lines)


def format_eval_results(results: object) -> str:
    """Format evaluation results summary for terminal output."""
    lines = [
        "Evaluation results",
        f"  Episodes     : {results.num_episodes}",  # type: ignore[attr-defined]
        f"  Mean return  : {results.mean_return:.2f} ± {results.std_return:.2f}",  # type: ignore[attr-defined]
        f"  Min / Max    : {results.min_return:.2f} / {results.max_return:.2f}",  # type: ignore[attr-defined]
        f"  Mean length  : {results.mean_length:.2f} ± {results.std_length:.2f}",  # type: ignore[attr-defined]
    ]
    return "\n".join(lines)
