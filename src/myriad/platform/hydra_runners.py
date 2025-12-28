"""Hydra-decorated runner functions for CLI and scripts.

This module contains the @hydra.main decorated entry points for training,
evaluation, and sweeps. Both the CLI and scripts/ import from here to avoid
duplication and maintain a single source of truth.
"""

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from myriad.configs.default import Config, EvalConfig
from myriad.platform.runner import evaluate, train_and_evaluate
from myriad.utils.rendering import render_cartpole_frame, render_episode_to_video

# Suppress excessive JAX logging when running on CPU
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# Find configs directory - works for both development and installed package
# Priority: 1) Environment variable, 2) CWD/configs, 3) Package location
def _get_config_path() -> str:
    """Get the absolute path to the configs directory."""
    # Check environment variable first
    if "MYRIAD_CONFIG_PATH" in os.environ:
        return os.environ["MYRIAD_CONFIG_PATH"]

    # Try current working directory (for development)
    cwd_configs = Path.cwd() / "configs"
    if cwd_configs.exists():
        return str(cwd_configs)

    # Fall back to package-relative path
    # This assumes configs is at the same level as src/ in the repo
    package_dir = Path(__file__).resolve().parent.parent.parent.parent
    config_path = package_dir / "configs"
    if config_path.exists():
        return str(config_path)

    # Last resort: return relative path and let Hydra error
    return "../configs"


_CONFIG_PATH = _get_config_path()


# Environment-specific rendering functions
# Add new environments here as rendering functions are implemented
ENV_RENDERERS = {
    "cartpole-control": render_cartpole_frame,  # Same renderer for both CartPole variants
}


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def train_main(cfg: DictConfig) -> None:
    """Main entry point for training, decorated by Hydra.

    Hydra will automatically:
    1. Find the `configs/config.yaml` file.
    2. Compose the configuration based on the `defaults` list.
    3. Allow overrides from the command line.
    4. Pass the final configuration as the `cfg` argument.
    """
    # Convert the Hydra configuration into a Pydantic configuration
    config_dict = OmegaConf.to_object(cfg)
    config: Config = Config(**config_dict)  # type: ignore

    logger.info("=" * 60)
    logger.info("Running with the following configuration:")
    logger.info(str(config))
    logger.info("=" * 60)

    # Call your existing runner with the fully-typed and populated config object
    train_and_evaluate(config)


def render_videos_from_episodes(
    episodes_dir: str | Path,
    env_name: str,
    output_dir: str | Path = "videos",
    fps: int = 50,
) -> int:
    """Render saved episodes to video files.

    Args:
        episodes_dir: Directory containing .npz episode files
        env_name: Name of the environment (to select the appropriate renderer)
        output_dir: Directory where videos will be saved
        fps: Frames per second for rendered videos

    Returns:
        Number of videos successfully rendered
    """
    import numpy as np

    episodes_path = Path(episodes_dir).resolve()
    if not episodes_path.exists():
        logger.warning(f"Episodes directory not found: {episodes_path}")
        return 0

    # Get the appropriate renderer for this environment
    render_frame_fn = ENV_RENDERERS.get(env_name)
    if render_frame_fn is None:
        logger.warning(f"No renderer available for environment '{env_name}'. Skipping video rendering.")
        logger.warning(f"Available environments: {list(ENV_RENDERERS.keys())}")
        return 0

    # Find all episode files
    episode_files = sorted(episodes_path.rglob("*.npz"))
    if not episode_files:
        logger.warning(f"No episode files found in {episodes_path}")
        return 0

    # Create output directory
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Rendering {len(episode_files)} episode(s) to video...")

    # Render each episode
    rendered_count = 0
    for episode_file in episode_files:
        try:
            # Load episode data
            episode_data = np.load(episode_file)

            # Generate output filename (preserve directory structure)
            relative_path = episode_file.relative_to(episodes_path)
            video_name = relative_path.with_suffix(".mp4")
            video_path = output_path / video_name

            # Ensure parent directory exists
            video_path.parent.mkdir(parents=True, exist_ok=True)

            # Render episode to video
            render_episode_to_video(
                episode_data,
                render_frame_fn,
                video_path,
                fps=fps,
            )

            logger.info(f"  → {video_name}")
            rendered_count += 1

        except Exception as e:
            logger.error(f"Failed to render {episode_file.name}: {e}")
            continue

    logger.info(f"Successfully rendered {rendered_count}/{len(episode_files)} videos to {output_path}")
    return rendered_count


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="config")
def evaluate_main(cfg: DictConfig) -> None:
    """Main entry point for evaluation-only runs (no training).

    Use this for:
    - Classical controllers (random, bang-bang, PID)
    - Pre-trained models
    - Baseline comparisons
    - Debugging and visualization

    Hydra will automatically:
    1. Find the configuration file (default: configs/config.yaml)
    2. Compose the configuration based on the `defaults` list
    3. Allow overrides from the command line
    4. Pass the final configuration as the `cfg` argument

    Examples:
        # Run an evaluation config
        python scripts/evaluate.py --config-name=experiments/eval_bangbang_cartpole

        # Override parameters
        python scripts/evaluate.py --config-name=experiments/eval_bangbang_cartpole eval_rollouts=100
    """
    # Convert the Hydra configuration into a Pydantic configuration
    config_dict = OmegaConf.to_object(cfg)
    config: EvalConfig = EvalConfig(**config_dict)  # type: ignore

    logger.info("Running evaluation with the following configuration:")
    logger.info(str(config))

    # Run evaluation and get results
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

    # Render videos if enabled and episodes were saved
    if config.run.eval_render_videos and config.run.eval_episode_save_frequency > 0:
        episodes_path = Path("episodes").resolve()
        videos_path = Path("videos").resolve()

        logger.info("")
        logger.info("Rendering episode videos...")
        logger.info(f"Episodes directory: {episodes_path}")
        logger.info(f"Videos directory: {videos_path}")

        render_videos_from_episodes(
            episodes_dir=episodes_path,
            env_name=config.env.name,
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
    # We need to set them in the nested DictConfig structure
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

    # Convert the Hydra configuration into a Pydantic configuration
    config_dict = OmegaConf.to_object(cfg)
    config: Config = Config(**config_dict)  # type: ignore

    print("--- Running sweep with the following configuration ---")
    print(config)
    print("------------------------------------------------------")

    # Call the runner with the configuration
    train_and_evaluate(config)

    # Finish the W&B run
    wandb.finish()
