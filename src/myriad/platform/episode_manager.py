"""Episode persistence and rendering management.

This module handles:
- Saving episode trajectories to disk (.npz format)
- Loading episodes from disk
- Rendering episodes to video files
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from myriad.configs.default import Config, EvalConfig

logger = logging.getLogger(__name__)


def save_episodes_to_disk(
    episode_data: dict[str, Any],
    global_step: int,
    save_count: int,
    config: Config | EvalConfig,
) -> str | None:
    """Save episode trajectories to disk for later analysis.

    Episodes are saved as compressed numpy archives (.npz) with the following structure:
    - episodes/step_{global_step}/episode_{i}.npz (relative to Hydra output dir)

    After saving, episodes can be rendered to videos using render_episodes_to_videos(),
    which will create MP4 files in a videos/ directory.

    Args:
        episode_data: Dictionary containing episode data from evaluation
        global_step: Current training step (for naming/organization)
        save_count: Number of episodes to save (saves first N from eval_rollouts)
        config: Training configuration (for metadata)

    Returns:
        Path to the episode directory if successful, None otherwise
    """
    # Extract episode trajectories
    if "episodes" not in episode_data:
        return None  # No episode data to save

    episodes = episode_data["episodes"]
    episode_lengths = episode_data["episode_length"]
    episode_returns = episode_data["episode_return"]

    # Create output directory (hard-coded to "episodes" relative to Hydra run dir)
    base_dir = Path("episodes")
    episodes_dir = base_dir / f"step_{global_step:08d}"

    try:
        episodes_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"Failed to create episode directory {episodes_dir}: {e}")
        return None

    # Save only the first save_count episodes
    num_to_save = min(save_count, len(episode_lengths))
    saved_count = 0

    for i in range(num_to_save):
        ep_len = int(episode_lengths[i])
        ep_file = episodes_dir / f"episode_{i}.npz"

        try:
            # Extract valid portion of each episode (no padding)
            np.savez_compressed(
                ep_file,
                observations=episodes["observations"][i, :ep_len],
                actions=episodes["actions"][i, :ep_len],
                rewards=episodes["rewards"][i, :ep_len],
                dones=episodes["dones"][i, :ep_len],
                # Metadata
                episode_length=ep_len,
                episode_return=float(episode_returns[i]),
                global_step=global_step,
                seed=config.run.seed,
            )
            saved_count += 1
        except (OSError, IOError) as e:
            logger.warning(f"Failed to save episode {i} to {ep_file}: {e}")
            continue

    if saved_count > 0:
        logger.info(f"Saved {saved_count}/{num_to_save} episodes to {episodes_dir}")
        return str(episodes_dir)
    else:
        return None


def render_episodes_to_videos(
    episodes_dir: str | Path,
    render_frame_fn: Callable[[np.ndarray], np.ndarray],
    output_dir: str | Path = "videos",
    fps: int = 50,
) -> int:
    """Render saved episodes to video files.

    Args:
        episodes_dir: Directory containing .npz episode files
        render_frame_fn: Function that converts observation to RGB frame
            Signature: (observation: np.ndarray) -> np.ndarray (H, W, 3)
        output_dir: Directory where videos will be saved
        fps: Frames per second for rendered videos

    Returns:
        Number of videos successfully rendered
    """
    from myriad.utils.rendering import render_episode_to_video

    episodes_path = Path(episodes_dir).resolve()
    if not episodes_path.exists():
        logger.warning(f"Episodes directory not found: {episodes_path}")
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
