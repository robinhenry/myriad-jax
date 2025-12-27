"""Video generation utilities for episode rendering.

This module provides functions to convert sequences of frames into videos
and to render complete episodes from saved trajectory data.
"""

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
        >>> episode_data = np.load("episodes/step_1000000/episode_0.npz")
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
