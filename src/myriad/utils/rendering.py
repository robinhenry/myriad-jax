"""Video generation utilities for episode rendering.

This module provides functions to convert sequences of frames into videos
and to render complete episodes from saved trajectory data.

The high-level `render_episodes()` function provides a unified API for rendering
episodes from multiple sources (EvaluationResults, episode dicts, or saved files)
with support for batch rendering.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

import imageio
import numpy as np

from myriad.platform.logging.backends.disk import EPISODE_FILE_FORMAT, STEP_DIR_FORMAT


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


def _detect_mode(
    results: Any | None,
    episode: dict | list | None,
    run_dir: str | Path | None,
    env_name: str | None,
    step: int | list | None,
) -> str:
    """Detect rendering mode from parameters and validate.

    Args:
        results: EvaluationResults object (Mode 1)
        episode: Episode dict or list of dicts (Mode 2)
        run_dir: Run directory (Mode 3)
        env_name: Environment name (required for Mode 2)
        step: Training step (required for Mode 3)

    Returns:
        Mode string: "results", "episode", or "disk"

    Raises:
        ValueError: If invalid combination of parameters
    """
    modes_provided = sum([results is not None, episode is not None, run_dir is not None])

    if modes_provided == 0:
        raise ValueError(
            "Must provide one of: 'results', 'episode', or 'run_dir'. " "See docstring for usage examples."
        )

    if modes_provided > 1:
        raise ValueError(
            "Cannot mix input modes. Provide only one of: "
            "'results', 'episode' (with env_name), or 'run_dir' (with step)"
        )

    # Mode 1: From EvaluationResults
    if results is not None:
        if results.episodes is None:
            raise ValueError("EvaluationResults.episodes is None. " "Call evaluate() with return_episodes=True")
        return "results"

    # Mode 2: From episode dict(s)
    if episode is not None:
        if env_name is None:
            raise ValueError(
                "When using 'episode' mode, you must provide 'env_name'. "
                "Example: render_episodes(episode=data, env_name='cartpole-control', ...)"
            )
        return "episode"

    # Mode 3: From disk
    if run_dir is not None:
        if step is None:
            raise ValueError(
                "When using 'run_dir' mode, you must provide 'step'. "
                "Example: render_episodes(run_dir='outputs/...', step=5000, ...)"
            )
        return "disk"

    raise ValueError("Invalid parameter combination")


def render_episodes(
    *,
    # Mode 1: From EvaluationResults
    results: Any | None = None,
    episode_index: int | list[int] = 0,
    # Mode 2: From episode dict(s)
    episode: dict[str, np.ndarray] | list[dict[str, np.ndarray]] | None = None,
    env_name: str | None = None,
    # Mode 3: From disk
    run_dir: str | Path | None = None,
    step: int | list[int] | None = None,
    # Common parameters
    output_path: str | Path | list[str | Path],
    fps: int = 50,
    max_frames: int | None = None,
) -> tuple[Path, dict[str, Any]] | tuple[list[Path], list[dict[str, Any]]]:
    """Render episode(s) to video with automatic environment detection.

    Supports three input modes (provide exactly one):

    1. **From EvaluationResults**: results + episode_index → extracts episode(s), auto-discovers env_name
    2. **From episode dict(s)**: episode + env_name → renders pre-extracted episode(s)
    3. **From disk**: run_dir + step → loads episode(s), auto-discovers env_name from config

    **Batch Rendering**: Pass lists for `episode_index`, `episode`, or `step` to render multiple
    episodes. When rendering multiple episodes, `output_path` can be:
    - A directory path (e.g., "videos/") → auto-generates filenames
    - A list of paths (same length as episodes) → explicit paths for each video

    Args:
        results: EvaluationResults object (Mode 1). Requires return_episodes=True.
        episode_index: Episode index/indices to render. Single int or list of ints (default: 0).
        episode: Episode dict(s) with keys: observations, actions, rewards, dones (Mode 2).
                 Can be single dict or list of dicts.
        env_name: Environment name. Required for Mode 2. For Mode 1, required if results
                 doesn't have a config attribute (e.g., from evaluate() instead of train_and_evaluate()).
        run_dir: Run directory containing episodes/ subdirectory and config (Mode 3)
        step: Training step(s) to render from run_dir. Single int or list of ints.
        output_path: Where to save video(s). For batch rendering:
            - Directory: "videos/" → auto-names as "env_name_step_NNNN.mp4" or "env_name_episode_NNNN.mp4"
            - List of paths: ["vid1.mp4", "vid2.mp4"] → explicit path for each episode
        fps: Frames per second
        max_frames: Optional limit on frames to render

    Returns:
        - **Single episode**: tuple[Path, dict] - (video_path, metadata)
        - **Batch**: tuple[list[Path], list[dict]] - (video_paths, metadata_list)

        Metadata dict includes:
            - episode_return: Total episode return
            - episode_length: Episode length
            - env_name: Environment name
            - global_step: Training step (if from disk)
            - seed: Random seed (if available)

    Examples:
        # Single episode from EvaluationResults
        >>> results = evaluate(config, return_episodes=True)
        >>> path, meta = render_episodes(results=results, output_path="video.mp4")

        # Batch episodes from EvaluationResults
        >>> paths, metas = render_episodes(
        ...     results=results,
        ...     episode_index=[0, 1, 2],
        ...     output_path="videos/"  # Auto-names files
        ... )

        # Single episode from dict
        >>> episode_data = {k: v[0] for k, v in results.episodes.items()}
        >>> path, meta = render_episodes(
        ...     episode=episode_data,
        ...     env_name="cartpole-control",
        ...     output_path="video.mp4"
        ... )

        # Single checkpoint from disk
        >>> path, meta = render_episodes(
        ...     run_dir="outputs/2026-02-12/12-56-45",
        ...     step=5000,
        ...     output_path="video.mp4"
        ... )

        # Batch checkpoints from disk
        >>> paths, metas = render_episodes(
        ...     run_dir="outputs/2026-02-12/12-56-45",
        ...     step=[500, 2500, 5000],
        ...     output_path="videos/"
        ... )

    Raises:
        ValueError: If invalid combination of parameters provided
        FileNotFoundError: If run_dir or episode file doesn't exist
    """
    from myriad.envs import get_env_info
    from myriad.platform.artifact_loader import load_run_config

    # Detect and validate mode
    mode = _detect_mode(results, episode, run_dir, env_name, step)

    # Detect if batch rendering is requested
    is_batch = False
    num_episodes = 1

    if mode == "results":
        is_batch = isinstance(episode_index, list)
        num_episodes = len(episode_index) if is_batch else 1
    elif mode == "episode":
        is_batch = isinstance(episode, list)
        num_episodes = len(episode) if is_batch else 1
    elif mode == "disk":
        is_batch = isinstance(step, list)
        num_episodes = len(step) if is_batch else 1

    # Handle output paths
    output_paths: list[Path]
    if is_batch:
        if isinstance(output_path, list):
            # Explicit list of paths
            if len(output_path) != num_episodes:
                raise ValueError(
                    f"output_path list length ({len(output_path)}) must match " f"number of episodes ({num_episodes})"
                )
            output_paths = [Path(p) for p in output_path]
        else:
            # Directory path - auto-generate filenames
            output_dir = Path(output_path)
            # Detect if it's meant to be a directory
            if not (str(output_path).endswith("/") or output_dir.is_dir()):
                raise ValueError(
                    "For batch rendering, output_path must be a directory (ending with '/') "
                    "or a list of paths matching the number of episodes"
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_paths = []  # Will be populated per-episode below
    else:
        # Single episode
        if isinstance(output_path, list):
            raise ValueError("output_path cannot be a list when rendering a single episode")
        output_paths = [Path(output_path)]

    # Get environment name and render function
    discovered_env_name: str
    if mode == "results":
        # Try to get from config first, then fall back to explicit env_name
        if env_name is not None:
            discovered_env_name = env_name
        elif hasattr(results, "config") and hasattr(results.config, "env"):
            discovered_env_name = results.config.env.name
        else:
            raise ValueError("Cannot determine env_name from EvaluationResults. " "Please provide env_name explicitly.")
    elif mode == "episode":
        discovered_env_name = env_name  # Required, already validated
    elif mode == "disk":
        # Load config to get env_name
        config = load_run_config(run_dir)
        discovered_env_name = config.env.name

    # Get render function
    env_info = get_env_info(discovered_env_name)
    if env_info is None:
        raise ValueError(f"Unknown environment: {discovered_env_name}")
    if env_info.render_frame_fn is None:
        raise ValueError(f"Environment '{discovered_env_name}' does not support rendering")
    render_fn = env_info.render_frame_fn

    # Render episodes
    video_paths: list[Path] = []
    metadata_list: list[dict[str, Any]] = []

    # Prepare iteration based on mode
    if mode == "results":
        indices = episode_index if is_batch else [episode_index]
        for i, idx in enumerate(indices):
            # Extract episode from results
            episode_data = {k: v[idx] for k, v in results.episodes.items()}

            # Auto-generate filename if needed
            if is_batch and not isinstance(output_path, list):
                fname = f"{discovered_env_name.replace('-', '_')}_episode_{idx:04d}.mp4"
                out_path = output_dir / fname
            else:
                out_path = output_paths[i]

            # Render
            video_path = render_episode_to_video(episode_data, render_fn, out_path, fps, max_frames)
            video_paths.append(video_path)

            # Collect metadata
            meta = {
                "episode_return": float(episode_data.get("episode_return", episode_data["rewards"].sum())),
                "episode_length": int(episode_data.get("episode_length", episode_data["dones"].sum())),
                "env_name": discovered_env_name,
                "global_step": None,
                "seed": getattr(results, "seed", None),
            }
            metadata_list.append(meta)

    elif mode == "episode":
        episodes = episode if is_batch else [episode]
        for i, ep in enumerate(episodes):
            # Auto-generate filename if needed
            if is_batch and not isinstance(output_path, list):
                fname = f"{discovered_env_name.replace('-', '_')}_episode_{i:04d}.mp4"
                out_path = output_dir / fname
            else:
                out_path = output_paths[i]

            # Render
            video_path = render_episode_to_video(ep, render_fn, out_path, fps, max_frames)
            video_paths.append(video_path)

            # Collect metadata
            meta = {
                "episode_return": float(ep.get("episode_return", ep["rewards"].sum())),
                "episode_length": int(ep.get("episode_length", ep["dones"].sum())),
                "env_name": discovered_env_name,
                "global_step": ep.get("global_step"),
                "seed": ep.get("seed"),
            }
            metadata_list.append(meta)

    elif mode == "disk":
        run_dir_path = Path(run_dir)
        episodes_dir = run_dir_path / "episodes"
        steps = step if is_batch else [step]

        for i, s in enumerate(steps):
            # Load episode from disk
            ep_idx = episode_index if isinstance(episode_index, int) else episode_index[i]
            episode_file = episodes_dir / STEP_DIR_FORMAT.format(s) / EPISODE_FILE_FORMAT.format(ep_idx)
            if not episode_file.exists():
                # Try to provide helpful error message
                available_dirs = sorted([d.name for d in episodes_dir.iterdir() if d.is_dir()])
                raise FileNotFoundError(
                    f"Episode file not found: {episode_file}\n" f"Available checkpoint directories: {available_dirs}"
                )

            episode_data = dict(np.load(episode_file))

            # Auto-generate filename if needed
            if is_batch and not isinstance(output_path, list):
                fname = f"{discovered_env_name.replace('-', '_')}_{STEP_DIR_FORMAT.format(s)}.mp4"
                out_path = output_dir / fname
            else:
                out_path = output_paths[i]

            # Render
            video_path = render_episode_to_video(episode_data, render_fn, out_path, fps, max_frames)
            video_paths.append(video_path)

            # Collect metadata
            meta = {
                "episode_return": float(episode_data.get("episode_return", episode_data["rewards"].sum())),
                "episode_length": int(episode_data.get("episode_length", episode_data["dones"].sum())),
                "env_name": discovered_env_name,
                "global_step": int(episode_data.get("global_step", s)),
                "seed": episode_data.get("seed"),
            }
            metadata_list.append(meta)

    # Return single tuple or lists based on batch mode
    if is_batch:
        return video_paths, metadata_list
    else:
        return video_paths[0], metadata_list[0]
