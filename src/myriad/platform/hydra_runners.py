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
from myriad.platform.evaluation import evaluate
from myriad.platform.logging.backends.disk import render_episodes_to_videos
from myriad.platform.training import train_and_evaluate

# Suppress excessive JAX logging when running on CPU
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


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
    # Convert Hydra configuration to Pydantic model for validation and typing
    config_dict = OmegaConf.to_object(cfg)
    config = Config.model_validate(config_dict)

    logger.info("=" * 60)
    logger.info("Running with the following configuration:")
    logger.info(str(config))
    logger.info("=" * 60)

    train_and_evaluate(config)


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def evaluate_main(cfg: DictConfig) -> None:
    """Main entry point for evaluation-only runs."""
    # Convert Hydra configuration to Pydantic model
    config_dict = OmegaConf.to_object(cfg)
    config = EvalConfig.model_validate(config_dict)

    logger.info("Running evaluation with the following configuration:")
    logger.info(str(config))

    # Run evaluation
    results = evaluate(config=config, return_episodes=False)

    # Log summary statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Episodes: {results.num_episodes}")
    logger.info(f"Mean return: {results.mean_return:.2f} ± {results.std_return:.2f}")
    logger.info(f"Min return: {results.min_return:.2f}")
    logger.info(f"Max return: {results.max_return:.2f}")
    logger.info(f"Mean episode length: {results.mean_length:.2f}")
    logger.info("=" * 60)

    # Render videos if enabled
    if config.run.eval_render_videos and config.run.eval_episode_save_frequency > 0:
        episodes_path = Path("episodes").resolve()
        videos_path = Path("videos").resolve()

        # Get the renderer from the environment registry
        env_info = get_env_info(config.env.name)
        render_frame_fn = env_info.render_frame_fn if env_info else None

        if render_frame_fn is None:
            logger.warning(f"No renderer available for environment '{config.env.name}'. Skipping video rendering.")
        else:
            logger.info("")
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

    logger.info("-" * 60)
    logger.info("Running sweep with the following configuration:")
    logger.info(str(config))
    logger.info("-" * 60)

    # Call the runner with the configuration
    train_and_evaluate(config)

    # Finish the W&B run
    wandb.finish()
