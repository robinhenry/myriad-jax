"""Disk persistence backend for episode data.

Handles saving episode trajectories to disk as compressed numpy archives.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# Centralized formats for episode storage (defined once, used everywhere)
STEP_DIR_FORMAT = "step_{:08d}"  # 8 digits to support large step counts
EPISODE_FILE_FORMAT = "episode_{:04d}.npz"  # 4 digits for consistency


class DiskBackend:
    """Backend for persisting episodes to disk.

    Episodes are saved as compressed numpy archives (.npz) with the structure:
    - `{base_dir}/step_{steps_per_env:08d}/episode_{i:04d}.npz`

    Each episode file contains:
    - observations, actions, rewards, dones (trajectory data)
    - episode_length, episode_return, global_step, seed (metadata)
    """

    def __init__(self, base_dir: Path, seed: int = 0) -> None:
        """Initialize the disk backend.

        Args:
            base_dir: Base directory for episode storage (e.g., run_dir/episodes)
            seed: Random seed for metadata
        """
        self.base_dir = base_dir
        self.seed = seed

    def save_episodes(
        self,
        episode_data: dict[str, Any],
        global_step: int,
        steps_per_env: int,
        save_count: int,
    ) -> Path | None:
        """Save episode trajectories to disk.

        Args:
            episode_data: Dictionary containing 'episodes', 'episode_length', 'episode_return'
            global_step: Current global training step (total across all envs)
            steps_per_env: Training steps per individual environment (for directory naming)
            save_count: Number of episodes to save (saves first N from eval_rollouts)

        Returns:
            Path to the episode directory if successful, None otherwise
        """
        if "episodes" not in episode_data:
            return None

        episodes = episode_data["episodes"]
        episode_lengths = episode_data["episode_length"]
        episode_returns = episode_data["episode_return"]

        # Use steps_per_env for more intuitive directory naming
        episodes_dir = self.base_dir / STEP_DIR_FORMAT.format(steps_per_env)

        try:
            episodes_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Failed to create episode directory {episodes_dir}: {e}")
            return None

        num_to_save = min(save_count, len(episode_lengths))
        saved_count = 0

        for i in range(num_to_save):
            ep_len = int(episode_lengths[i])
            ep_file = episodes_dir / EPISODE_FILE_FORMAT.format(i)

            try:
                np.savez_compressed(
                    ep_file,
                    observations=episodes["observations"][i, :ep_len],
                    actions=episodes["actions"][i, :ep_len],
                    rewards=episodes["rewards"][i, :ep_len],
                    dones=episodes["dones"][i, :ep_len],
                    episode_length=ep_len,
                    episode_return=float(episode_returns[i]),
                    global_step=global_step,
                    seed=self.seed,
                )
                saved_count += 1
            except (OSError, IOError) as e:
                logger.warning(f"Failed to save episode {i} to {ep_file}: {e}")
                continue

        if saved_count > 0:
            logger.debug(f"Saved {saved_count}/{num_to_save} episodes to {episodes_dir}")
            return episodes_dir
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
    from myriad.utils import rendering

    episodes_path = Path(episodes_dir).resolve()
    if not episodes_path.exists():
        logger.warning(f"Episodes directory not found: {episodes_path}")
        return 0

    episode_files = sorted(episodes_path.rglob("*.npz"))
    if not episode_files:
        logger.warning(f"No episode files found in {episodes_path}")
        return 0

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Rendering {len(episode_files)} episode(s) to video...")

    rendered_count = 0
    for episode_file in episode_files:
        try:
            episode_data = np.load(episode_file)
            relative_path = episode_file.relative_to(episodes_path)
            video_name = relative_path.with_suffix(".mp4")
            video_path = output_path / video_name

            video_path.parent.mkdir(parents=True, exist_ok=True)

            rendering.render_episode_to_video(
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
