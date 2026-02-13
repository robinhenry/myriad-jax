"""Hydra-decorated runner functions for CLI and scripts.

This module contains the @hydra.main decorated entry points for training,
evaluation, and sweeps. Both the CLI and scripts/ import from here to avoid
duplication and maintain a single source of truth.
"""

import logging
import os
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from myriad.configs.default import Config, EvalConfig
from myriad.envs import get_env_info

from .evaluation import evaluate
from .logging.backends.disk import render_episodes_to_videos
from .training import train_and_evaluate

# Suppress excessive JAX logging when running on CPU
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Shorten Hydra's verbose log prefix to timestamp + level."""
    fmt = logging.Formatter(fmt="[%(asctime)s %(levelname)s] %(message)s", datefmt="%H:%M:%S")
    for handler in logging.root.handlers:
        handler.setFormatter(fmt)


def _fmt_fields(model: object) -> str:
    """Format non-default fields of a Pydantic model as 'key=value | key=value'."""
    from pydantic import BaseModel

    if not isinstance(model, BaseModel):
        return str(model)
    non_defaults = model.model_dump(exclude_defaults=True)
    # Always include 'name' if present (it's required, has no default, but exclude_defaults may still miss it)
    if hasattr(model, "name") and "name" not in non_defaults:
        non_defaults = {"name": model.name} | non_defaults  # type: ignore[attr-defined]
    return " | ".join(f"{k}={v}" for k, v in non_defaults.items()) if non_defaults else "(defaults)"


def _format_eval_config(config: "EvalConfig") -> str:
    wandb_status = "disabled" if (config.wandb is None or not config.wandb.enabled) else _fmt_fields(config.wandb)
    config_path = Path.cwd() / ".hydra" / "config.yaml"
    lines = [
        f"Evaluating {config.agent.name} on {config.env.name}",
        f"  Agent : {_fmt_fields(config.agent)}",
        f"  Env   : {_fmt_fields(config.env)}",
        f"  Run   : {_fmt_fields(config.run)}",
        f"  W&B   : {wandb_status}",
        f"  Config: {config_path}",
    ]
    return "\n".join(lines)


def _format_eval_results(results: object) -> str:
    lines = [
        "Evaluation results",
        f"  Episodes     : {results.num_episodes}",  # type: ignore[attr-defined]
        f"  Mean return  : {results.mean_return:.2f} ± {results.std_return:.2f}",  # type: ignore[attr-defined]
        f"  Min / Max    : {results.min_return:.2f} / {results.max_return:.2f}",  # type: ignore[attr-defined]
        f"  Mean length  : {results.mean_length:.2f} ± {results.std_length:.2f}",  # type: ignore[attr-defined]
    ]
    return "\n".join(lines)


def _format_train_config(config: "Config") -> str:
    wandb_status = "disabled" if (config.wandb is None or not config.wandb.enabled) else _fmt_fields(config.wandb)
    config_path = Path.cwd() / ".hydra" / "config.yaml"
    lines = [
        f"Training {config.agent.name} on {config.env.name}",
        f"  Agent : {_fmt_fields(config.agent)}",
        f"  Env   : {_fmt_fields(config.env)}",
        f"  Run   : {_fmt_fields(config.run)}",
        f"  W&B   : {wandb_status}",
        f"  Config: {config_path}",
    ]
    return "\n".join(lines)


def _get_config_path() -> str:
    """Get the absolute path to the configs directory with robust fallback logic.

    Priority:
    1. Environment variable MYRIAD_CONFIG_PATH
    2. A 'configs' directory in the current working directory (for development)
    3. The 'configs' directory relative to this package source (for repository use)
    """
    # 1. Check environment variable
    if env_path := os.environ.get("MYRIAD_CONFIG_PATH"):
        return env_path

    # 2. Try current working directory
    cwd_configs = Path.cwd() / "configs"
    if cwd_configs.exists() and cwd_configs.is_dir():
        return str(cwd_configs)

    # 3. Fall back to repository root (assuming standard myriad-jax layout)
    # This module is at src/myriad/platform/hydra_runners.py
    repo_root = Path(__file__).resolve().parents[3]
    repo_configs = repo_root / "configs"
    if repo_configs.exists() and repo_configs.is_dir():
        return str(repo_configs)

    # Last resort: return relative path and let Hydra attempt discovery
    return "../configs"


_CONFIG_PATH = _get_config_path()


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def train_main(cfg: DictConfig) -> None:
    """Main entry point for training, decorated by Hydra."""
    _configure_logging()
    # Convert Hydra configuration to Pydantic model for validation and typing
    config_dict = OmegaConf.to_object(cfg)
    config = Config.model_validate(config_dict)

    logger.info(_format_train_config(config))

    train_and_evaluate(config)


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def evaluate_main(cfg: DictConfig) -> None:
    """Main entry point for evaluation-only runs."""
    _configure_logging()
    # Convert Hydra configuration to Pydantic model
    config_dict = OmegaConf.to_object(cfg)
    config = EvalConfig.model_validate(config_dict)

    logger.info(_format_eval_config(config))

    # Run evaluation
    results = evaluate(config=config, return_episodes=False)

    logger.info(_format_eval_results(results))

    # Render videos if enabled
    if config.run.eval_render_videos and config.run.eval_episode_save_frequency > 0:
        # Use run directory (Hydra sets cwd to the output directory)
        episodes_path = Path.cwd() / "episodes"
        videos_path = Path.cwd() / "videos"

        # Get the renderer from the environment registry
        env_info = get_env_info(config.env.name)
        render_frame_fn = env_info.render_frame_fn if env_info else None

        if render_frame_fn is None:
            logger.warning(f"No renderer available for environment '{config.env.name}'. Skipping video rendering.")
        else:
            logger.info(f"Rendering episode videos to: {videos_path}")
            render_episodes_to_videos(
                episodes_dir=episodes_path,
                render_frame_fn=render_frame_fn,
                output_dir=videos_path,
                fps=config.run.eval_video_fps,
            )


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def sweep_main(cfg: DictConfig) -> None:
    """Main entry point for sweep training.

    This function:
    1. Initializes a W&B run (which pulls sweep parameters)
    2. Overrides Hydra config with sweep parameters
    3. Runs training with the combined configuration
    """
    _configure_logging()
    # Initialize W&B run - this will pull parameters from the sweep
    wandb.init()

    # Update Hydra config with W&B sweep parameters
    # W&B config keys use dots (e.g., "agent.learning_rate")
    for key, value in wandb.config.items():
        if "." in key:
            # Handle nested keys like "agent.learning_rate"
            parts = key.split(".")
            config_part = cfg
            for part in parts[:-1]:
                if part not in config_part:
                    config_part[part] = {}
                config_part = config_part[part]
            config_part[parts[-1]] = value
        else:
            cfg[key] = value

    # Also update the wandb section of config to ensure W&B integration works
    cfg.wandb.enabled = True
    cfg.wandb.mode = wandb.config.get("wandb.mode", "online")

    # Convert Hydra configuration to Pydantic model
    config_dict = OmegaConf.to_object(cfg)
    config = Config.model_validate(config_dict)

    logger.info(_format_train_config(config))

    # Call the runner with the configuration
    train_and_evaluate(config)

    # Finish the W&B run
    wandb.finish()
