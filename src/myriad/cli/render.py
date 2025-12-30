"""Render CLI command for converting episode trajectories to videos.

This module provides the Click-based CLI for rendering saved episode files.
"""

import logging
from pathlib import Path

import click
import numpy as np

from myriad.utils.rendering import render_cartpole_frame, render_ccas_ccar_frame, render_episode_to_video

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


ENV_RENDERERS = {
    "cartpole": render_cartpole_frame,
    "ccas_ccar": render_ccas_ccar_frame,
}


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--env",
    type=click.Choice(list(ENV_RENDERERS.keys())),
    default="cartpole",
    help="Environment type (determines rendering function)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for videos (default: videos/)",
)
@click.option(
    "--fps",
    type=int,
    default=50,
    help="Frames per second for output video",
)
@click.option(
    "--max-frames",
    type=int,
    default=None,
    help="Maximum number of frames to render per episode",
)
@click.option(
    "--wandb-project",
    type=str,
    default=None,
    help="W&B project name for uploading videos",
)
@click.option(
    "--wandb-run-id",
    type=str,
    default=None,
    help="W&B run ID to attach videos to",
)
def render(
    input_path: Path,
    env: str,
    output_dir: Path | None,
    fps: int,
    max_frames: int | None,
    wandb_project: str | None,
    wandb_run_id: str | None,
):
    """Render episode trajectories to MP4 videos.

    INPUT_PATH can be either:
    - A single .npz episode file
    - A directory containing .npz episode files
    """
    # Get the appropriate renderer for this environment
    render_frame_fn = ENV_RENDERERS[env]

    # Find all episode files to render
    if input_path.is_file():
        episode_files = [input_path]
    else:
        episode_files = sorted(input_path.rglob("*.npz"))

    if not episode_files:
        logger.error(f"No .npz files found in {input_path}")
        return

    logger.info(f"Found {len(episode_files)} episode(s) to render")

    # Set output directory (default to videos/ in current working directory)
    if output_dir is None:
        output_dir = Path("videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render each episode
    video_paths = []
    for episode_file in episode_files:
        logger.info(f"Rendering {episode_file.name}...")

        try:
            # Load episode data
            episode_data = np.load(episode_file)

            # Generate output filename
            video_name = episode_file.stem + ".mp4"
            video_path = output_dir / video_name

            # Render episode to video
            render_episode_to_video(
                episode_data,
                render_frame_fn,
                video_path,
                fps=fps,
                max_frames=max_frames,
            )

            logger.info(f"  → Saved to {video_path}")
            video_paths.append(video_path)

        except Exception as e:
            logger.error(f"  → Failed to render {episode_file.name}: {e}")
            continue

    logger.info(f"Successfully rendered {len(video_paths)}/{len(episode_files)} episodes")

    # Upload to W&B if requested
    if wandb_project is not None:
        logger.info("Uploading videos to W&B...")
        upload_videos_to_wandb(video_paths, wandb_project, wandb_run_id)


def upload_videos_to_wandb(
    video_paths: list[Path],
    project: str,
    run_id: str | None = None,
):
    """Upload rendered videos to Weights & Biases.

    Args:
        video_paths: List of paths to video files
        project: W&B project name
        run_id: Optional run ID to attach videos to (creates new run if None)
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Install with: pip install wandb")
        return

    # Initialize or resume W&B run
    if run_id is not None:
        run = wandb.init(project=project, id=run_id, resume="allow")
    else:
        run = wandb.init(project=project)

    # Log each video
    for video_path in video_paths:
        video_name = video_path.stem
        wandb.log({f"videos/{video_name}": wandb.Video(str(video_path), fps=50, format="mp4")})
        logger.info(f"  → Uploaded {video_name}")

    run.finish()
    logger.info("W&B upload complete")


__all__ = ["render"]
