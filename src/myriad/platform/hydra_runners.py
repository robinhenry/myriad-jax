"""Hydra-decorated runner functions for CLI and scripts.

This module contains the @hydra.main decorated entry points for training,
evaluation, and sweeps. Both the CLI and scripts/ import from here to avoid
duplication and maintain a single source of truth.
"""

import logging
import os
import signal
import sys
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict

from myriad.configs.default import Config, EvalConfig
from myriad.envs import get_env_info

from .display import format_eval_results
from .evaluation import evaluate
from .logging.backends.disk import render_episodes_to_videos
from .training import train_and_evaluate

# Suppress excessive JAX logging when running on CPU
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _apply_auto_tune(cfg: DictConfig) -> None:
    """Call suggest_scan_chunk_size and patch cfg.run.scan_chunk_size in-place."""
    from myriad.platform.autotune import suggest_scan_chunk_size

    buffer_size = getattr(cfg.run, "buffer_size", None)
    chunk_size = suggest_scan_chunk_size(
        num_envs=cfg.run.num_envs,
        env=cfg.env.name,
        agent=cfg.agent.name,
        buffer_size=buffer_size,
    )
    OmegaConf.update(cfg, "run.scan_chunk_size", chunk_size, force_add=True)
    logger.info(f"  Auto-tune: scan_chunk_size={chunk_size}")


def _configure_logging() -> None:
    """Shorten Hydra's verbose log prefix to timestamp + level."""
    fmt = logging.Formatter(fmt="[%(asctime)s %(levelname)s] %(message)s", datefmt="%H:%M:%S")
    for handler in logging.root.handlers:
        handler.setFormatter(fmt)


def _get_config_path() -> str:
    """Get the absolute path to the configs directory.

    Priority:
    1. Environment variable MYRIAD_CONFIG_PATH  (examples and scripts set this)
    2. Package-internal configs shipped with myriad  (fallback for sweeps etc.)
    """
    # 1. Check environment variable
    if env_path := os.environ.get("MYRIAD_CONFIG_PATH"):
        return str(Path(env_path).resolve())

    # 2. Package-internal fallback: configs/ lives alongside this module
    return str(Path(__file__).resolve().parent / "configs")


_CONFIG_PATH = _get_config_path()


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def train_main(cfg: DictConfig) -> None:
    """Main entry point for training, decorated by Hydra."""
    _configure_logging()
    if os.environ.pop("MYRIAD_AUTO_TUNE", None):
        _apply_auto_tune(cfg)
    # Convert Hydra configuration to Pydantic model for validation and typing
    config_dict = OmegaConf.to_object(cfg)
    config = Config.model_validate(config_dict)

    train_and_evaluate(config)


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def evaluate_main(cfg: DictConfig) -> None:
    """Main entry point for evaluation-only runs."""
    _configure_logging()
    # Convert Hydra configuration to Pydantic model
    config_dict = OmegaConf.to_object(cfg)
    config = EvalConfig.model_validate(config_dict)

    # Run evaluation
    results = evaluate(config=config, return_episodes=False)

    logger.info(format_eval_results(results))

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
    if os.environ.pop("MYRIAD_AUTO_TUNE", None):
        _apply_auto_tune(cfg)
    # Initialize W&B run — agent sets WANDB_SWEEP_ID so wandb.init() automatically
    # picks up the sampled hyperparameters for this run.
    wandb.init()

    # Handle SIGTERM (sent by the W&B agent for hyperband early termination).
    # Python's default SIGTERM handler kills the process immediately, skipping all
    # cleanup. Catching SIGTERM and raising SystemExit lets context managers unwind
    # normally; train_and_evaluate() treats SystemExit as an intentional stop and
    # calls wandb.finish(exit_code=0) so W&B marks the run as "killed", not "crashed".
    def _handle_sigterm(signum: int, frame: object) -> None:
        sys.exit(128 + signum)  # conventional: 128 + signal number

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # Apply W&B sweep parameters to the Hydra config. Keys use dot notation
    # (e.g. "agent.learning_rate"). Use open_dict to allow adding keys that are
    # not pre-declared in the base config (Hydra 1.x locks the config into struct
    # mode after composition, which would otherwise reject new keys).
    with open_dict(cfg):
        for key, value in wandb.config.items():
            parts = key.split(".")
            node = cfg
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value

        cfg.wandb.enabled = True

    # Convert Hydra configuration to Pydantic model
    config_dict = OmegaConf.to_object(cfg)
    config = Config.model_validate(config_dict)

    train_and_evaluate(config)
