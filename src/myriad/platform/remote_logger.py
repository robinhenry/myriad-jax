"""Remote logging handlers for external services (W&B, etc)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .logging_utils import build_train_payload, summarize_metric, wandb


class RemoteLogger:
    """Handles sending metrics to remote logging services.

    Supports multiple destinations (W&B, etc.) without coupling
    the core metrics capture logic to specific logging implementations.
    """

    def __init__(self, wandb_run: Any | None = None):
        """Initialize remote logger with configured destinations.

        Args:
            wandb_run: W&B run instance (None to disable)
        """
        self.wandb_run = wandb_run
        self.use_wandb = wandb_run is not None and wandb is not None

    def log_train(self, metrics_host: dict[str, Any], global_step: int, steps_per_env: int) -> None:
        """Send training metrics to configured remote services.

        Args:
            metrics_host: Raw metrics dictionary
            global_step: Global environment steps
            steps_per_env: Steps per individual environment
        """
        if not self.use_wandb:
            return

        train_payload = build_train_payload(metrics_host)
        if train_payload:
            train_payload["train/global_env_steps"] = float(global_step)
            train_payload["train/steps_per_env"] = float(steps_per_env)
            wandb.log(train_payload, step=global_step)

    def log_eval(self, eval_results: dict[str, Any], global_step: int) -> None:
        """Send evaluation metrics to configured remote services.

        Args:
            eval_results: Dictionary with 'episode_return', 'episode_length', 'dones'
            global_step: Global environment steps
        """
        if not self.use_wandb:
            return

        eval_payload: dict[str, float] = {}

        eval_returns = eval_results.get("episode_return")
        eval_lengths = eval_results.get("episode_length")
        eval_dones = eval_results.get("dones")

        if eval_returns is not None:
            eval_payload.update(summarize_metric("eval/", "episode_return", eval_returns))

        if eval_lengths is not None:
            eval_payload.update(summarize_metric("eval/", "episode_length", eval_lengths))

        if eval_dones is not None:
            termination_rate = float(np.asarray(eval_dones, dtype=np.float32).mean())
            eval_payload["eval/termination_rate"] = termination_rate
            eval_payload["eval/non_termination_rate"] = float(1.0 - termination_rate)

        if eval_payload:
            eval_payload["eval/global_env_steps"] = float(global_step)
            wandb.log(eval_payload, step=global_step)

    def log_episodes(self, episode_dir: str, global_step: int) -> None:
        """Log saved episodes to W&B as artifacts.

        Args:
            episode_dir: Path to directory containing saved episodes
            global_step: Global environment steps (for artifact versioning)
        """
        if not self.use_wandb:
            return

        try:
            # Log episode directory as a W&B artifact
            artifact = wandb.Artifact(
                name=f"episodes_step_{global_step}",
                type="evaluation_episodes",
                description=f"Episode trajectories from evaluation at step {global_step}",
            )
            artifact.add_dir(episode_dir)
            self.wandb_run.log_artifact(artifact)  # type: ignore[union-attr]
        except Exception as e:
            # Don't fail training if episode logging fails
            print(f"Warning: Failed to log episodes to W&B: {e}")

    def log_videos(
        self,
        episode_dir: str | Path,
        render_frame_fn: Callable[[np.ndarray], np.ndarray],
        global_step: int,
        fps: int = 50,
        max_episodes: int | None = None,
        max_frames: int | None = None,
    ) -> None:
        """Render saved episodes to videos and log to W&B.

        This method finds .npz episode files, renders them to MP4 videos,
        and uploads them to W&B for visualization in the dashboard.

        Args:
            episode_dir: Path to directory containing .npz episode files
            render_frame_fn: Function that takes observation array and returns RGB frame
            global_step: Global environment steps (for W&B logging step)
            fps: Frames per second for rendered videos
            max_episodes: Maximum number of episodes to render (None = all)
            max_frames: Maximum frames per episode (None = full episode)

        Example:
            >>> from myriad.utils.rendering import render_cartpole_frame
            >>> logger.log_videos("episodes/step_1000000", render_cartpole_frame, 1000000)
        """
        if not self.use_wandb:
            return

        try:
            from myriad.utils.rendering import render_episode_to_video

            episode_dir = Path(episode_dir)

            # Find all episode files
            episode_files = sorted(episode_dir.glob("*.npz"))
            if not episode_files:
                print(f"Warning: No .npz files found in {episode_dir}")
                return

            # Limit number of episodes if requested
            if max_episodes is not None:
                episode_files = episode_files[:max_episodes]

            # Render and log each episode
            for i, episode_file in enumerate(episode_files):
                try:
                    # Load episode data
                    episode_data = np.load(episode_file)

                    # Create temporary video file
                    video_path = episode_dir / f"{episode_file.stem}_video.mp4"

                    # Render episode to video
                    render_episode_to_video(
                        episode_data,
                        render_frame_fn,
                        video_path,
                        fps=fps,
                        max_frames=max_frames,
                    )

                    # Log to W&B
                    wandb.log(
                        {f"videos/episode_{i}": wandb.Video(str(video_path), fps=fps, format="mp4")},
                        step=global_step,
                    )

                    # Clean up temporary video file
                    video_path.unlink()

                except Exception as e:
                    print(f"Warning: Failed to render/log {episode_file.name}: {e}")
                    continue

        except Exception as e:
            # Don't fail training if video logging fails
            print(f"Warning: Failed to log videos to W&B: {e}")

    def log_final(self, total_env_steps: int) -> None:
        """Send final completion metrics to remote services.

        Args:
            total_env_steps: Total environment steps completed
        """
        if self.use_wandb:
            wandb.log({"train/final_env_steps": float(total_env_steps)}, step=total_env_steps)
