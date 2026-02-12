"""Video generation utilities for episode rendering.

This module provides functions to convert sequences of frames into videos
and to render complete episodes from saved trajectory data.
"""

import warnings
from pathlib import Path
from typing import Callable

import imageio
import numpy as np


def frames_to_video(
    frames: list[np.ndarray] | np.ndarray,
    output_path: str | Path,
    fps: int = 50,
    codec: str = "libx264",
    quality: int = 8,
) -> Path:
    """Convert sequence of RGB frames to MP4 video.

    Args:
        frames: List or array of RGB frames with shape (num_frames, height, width, 3)
        output_path: Path where video will be saved (should end in .mp4)
        fps: Frames per second for the output video
        codec: Video codec to use (default: libx264 for H.264)
        quality: Quality parameter (1-10, where 10 is highest quality/largest file)

    Returns:
        Path object pointing to the saved video file

    Example:
        >>> frames = [render_cartpole_frame(obs) for obs in observations]
        >>> video_path = frames_to_video(frames, "episode.mp4", fps=50)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert list to array if needed
    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)

    # Suppress the os.fork() warning from subprocess (used by ffmpeg via imageio)
    # This is safe here because JAX computation is complete before video rendering
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*os.fork.*", category=RuntimeWarning)

        # Write video using imageio
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec=codec,
            quality=quality,
            pixelformat="yuv420p",  # Ensures compatibility with most video players
        )

        for frame in frames:
            writer.append_data(frame)

        writer.close()

    return output_path


def render_episode_to_video(
    episode_data: dict[str, np.ndarray],
    render_frame_fn: Callable[[np.ndarray], np.ndarray],
    output_path: str | Path,
    fps: int = 50,
    max_frames: int | None = None,
) -> Path:
    """Render a complete episode to video from saved trajectory data.

    This is a convenience function that combines frame rendering and video generation.
    It handles episodes from .npz files saved by the Myriad platform.

    Args:
        episode_data: Dictionary containing episode trajectory with keys:
            - 'observations': Array of shape (max_steps, obs_dim)
            - 'actions': Array of shape (max_steps, ...)
            - 'rewards': Array of shape (max_steps,)
            - 'dones': Array of shape (max_steps,)
        render_frame_fn: Function that takes observation and returns RGB frame
        output_path: Path where video will be saved
        fps: Frames per second for the output video
        max_frames: Optional limit on number of frames to render (for long episodes)

    Returns:
        Path object pointing to the saved video file

    Example:
        >>> episode_data = np.load("episodes/step_001000/episode_0.npz")
        >>> video_path = render_episode_to_video(
        ...     episode_data,
        ...     render_cartpole_frame,
        ...     "cartpole_episode.mp4"
        ... )
    """
    observations = episode_data["observations"]
    dones = episode_data["dones"]

    # Find the actual episode length (up to first done=True)
    done_indices = np.where(dones)[0]
    episode_length = done_indices[0] + 1 if len(done_indices) > 0 else len(dones)

    # Limit frames if requested
    if max_frames is not None:
        episode_length = min(episode_length, max_frames)

    # Render all frames
    frames = []
    for t in range(episode_length):
        frame = render_frame_fn(observations[t])
        frames.append(frame)

    # Convert frames to video
    return frames_to_video(frames, output_path, fps=fps)


def render_saved_episodes(
    env_name: str,
    episodes_dir: str | Path = "episodes",
    steps: int | list[int] | None = None,
    output_dir: str | Path = "videos",
    fps: int = 50,
    episode_index: int = 0,
) -> tuple[list[Path], list[dict[str, np.ndarray]]]:
    """Render saved episodes from training checkpoints to videos.

    Convenience function for batch rendering episodes saved during training.
    Automatically finds the render function for the environment and loads
    episodes from disk.

    Args:
        env_name: Environment name (e.g., "cartpole-control")
        episodes_dir: Base directory containing saved episodes (default: "./episodes")
        steps: Steps per env to render. Can be:
            - None: render all available checkpoints
            - int: render single checkpoint (e.g., 500)
            - list: render multiple checkpoints (e.g., [500, 2500, 5000])
        output_dir: Directory where videos will be saved (default: "./videos")
        fps: Frames per second for output videos
        episode_index: Which episode to render if multiple were saved (default: 0)

    Returns:
        Tuple of (video_paths, episode_data_list):
        - video_paths: List of Path objects pointing to rendered video files
        - episode_data_list: List of dicts containing episode metadata:
            - 'observations': trajectory observations
            - 'actions': trajectory actions
            - 'rewards': trajectory rewards
            - 'dones': trajectory done flags
            - 'episode_return': total return
            - 'episode_length': episode length
            - 'global_step': global training step
            - 'seed': random seed

    Example:
        >>> # Render 1st, 5th, and 10th eval checkpoints
        >>> from myriad.utils.rendering import render_saved_episodes
        >>> video_paths, episode_data = render_saved_episodes(
        ...     "cartpole-control",
        ...     steps=[500, 2500, 5000]
        ... )
        >>> for path, data in zip(video_paths, episode_data):
        ...     print(f"{path.name}: return={data['episode_return']:.1f}")
    """
    from myriad.envs import get_env_info

    episodes_dir = Path(episodes_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get render function for this environment
    env_info = get_env_info(env_name)
    render_fn = env_info.render_frame_fn

    # Determine which checkpoints to render
    if steps is None:
        # Find all available checkpoint directories
        checkpoint_dirs = sorted([d for d in episodes_dir.iterdir() if d.is_dir() and d.name.startswith("step_")])
        steps_list = [int(d.name.split("_")[1]) for d in checkpoint_dirs]
    elif isinstance(steps, int):
        steps_list = [steps]
    else:
        steps_list = list(steps)

    video_paths = []
    episode_data_list = []

    for step in steps_list:
        # Load episode from disk
        episode_file = episodes_dir / f"step_{step:06d}" / f"episode_{episode_index}.npz"
        if not episode_file.exists():
            warnings.warn(f"Episode file not found: {episode_file}, skipping")
            continue

        episode_data = np.load(episode_file)

        # Render to video
        output_path = output_dir / f"{env_name.replace('-', '_')}_step_{step:06d}.mp4"
        video_path = render_episode_to_video(episode_data, render_fn, output_path, fps=fps)

        video_paths.append(video_path)
        episode_data_list.append(dict(episode_data))

    return video_paths, episode_data_list
