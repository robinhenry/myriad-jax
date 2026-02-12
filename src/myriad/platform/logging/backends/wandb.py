"""Weights & Biases remote logging backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from myriad.configs.default import Config, EvalConfig
from myriad.platform.logging.utils import build_train_payload, summarize_metric

# Only import wandb if installed
try:
    import wandb  # type: ignore[import]
except ImportError as import_error:
    wandb = None  # type: ignore[assignment]
    _wandb_import_error: ImportError | None = import_error
else:
    _wandb_import_error = None


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    """Remove items with None values from a dictionary."""
    return {key: value for key, value in values.items() if value is not None}


class WandbBackend:
    """Backend for logging to Weights & Biases.

    Handles:
    - Metric logging (training and evaluation)
    - Episode artifact uploads
    - Video rendering and logging
    """

    def __init__(self, wandb_run: Any | None = None) -> None:
        """Initialize the W&B backend.

        Args:
            wandb_run: W&B run instance (None to disable)
        """
        self.wandb_run = wandb_run
        self.use_wandb = wandb_run is not None and wandb is not None

    @property
    def local_dir(self) -> Path | None:
        """Get the W&B local directory for this run.

        Returns:
            Path to W&B local directory, or None if W&B is disabled
        """
        if not self.use_wandb or self.wandb_run is None:
            return None
        try:
            return Path(self.wandb_run.dir)
        except (AttributeError, TypeError):
            return None

    def log_training(
        self,
        metrics_host: dict[str, Any],
        global_step: int,
        steps_per_env: int,
    ) -> None:
        """Send training metrics to W&B.

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
            wandb.log(train_payload, step=global_step)  # type: ignore[union-attr]

    def log_evaluation(
        self,
        eval_results: dict[str, Any],
        global_step: int,
        steps_per_env: int,
    ) -> None:
        """Send evaluation metrics to W&B.

        Args:
            eval_results: Dictionary with 'episode_return', 'episode_length', 'dones'
            global_step: Global environment steps
            steps_per_env: Steps per individual environment
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
            eval_payload["eval/steps_per_env"] = float(steps_per_env)
            wandb.log(eval_payload, step=global_step)  # type: ignore[union-attr]

    def log_episodes(self, episode_dir: Path, global_step: int) -> None:
        """Log saved episodes to W&B as artifacts.

        Args:
            episode_dir: Path to directory containing saved episodes
            global_step: Global environment steps (for artifact versioning)
        """
        if not self.use_wandb:
            return

        try:
            artifact = wandb.Artifact(  # type: ignore[union-attr]
                name=f"episodes_step_{global_step}",
                type="evaluation_episodes",
                description=f"Episode trajectories from evaluation at step {global_step}",
            )
            artifact.add_dir(str(episode_dir))
            self.wandb_run.log_artifact(artifact)  # type: ignore[union-attr]
        except Exception as e:
            print(f"Warning: Failed to log episodes to W&B: {e}")

    def log_videos(
        self,
        episode_dir: str | Path,
        render_frame_fn: Callable[[np.ndarray], np.ndarray],
        global_step: int,
        fps: int = 50,
        max_episodes: int | None = None,
        max_frames: int | None = None,
        video_dir: Path | None = None,
    ) -> None:
        """Render saved episodes to videos and log to W&B.

        Args:
            episode_dir: Path to directory containing .npz episode files
            render_frame_fn: Function that takes observation array and returns RGB frame
            global_step: Global environment steps (for W&B logging step)
            fps: Frames per second for rendered videos
            max_episodes: Maximum number of episodes to render (None = all)
            max_frames: Maximum frames per episode (None = full episode)
            video_dir: Optional output directory for videos. If provided, videos are saved
                there permanently. Otherwise, temporary videos are created and cleaned up.
        """
        if not self.use_wandb:
            return

        try:
            from myriad.utils.rendering import render_episode_to_video

            episode_dir = Path(episode_dir)

            episode_files = sorted(episode_dir.glob("*.npz"))
            if not episode_files:
                print(f"Warning: No .npz files found in {episode_dir}")
                return

            if max_episodes is not None:
                episode_files = episode_files[:max_episodes]

            # Determine if we should save videos permanently or create temporary ones
            save_videos = video_dir is not None
            if save_videos:
                assert video_dir is not None  # type narrowing for mypy
                video_dir.mkdir(parents=True, exist_ok=True)

            for i, episode_file in enumerate(episode_files):
                try:
                    episode_data = np.load(episode_file)

                    # Save to video_dir if provided, otherwise create temporary in episode_dir
                    if save_videos:
                        assert video_dir is not None  # type narrowing for mypy
                        video_path = video_dir / f"{episode_file.stem}.mp4"
                    else:
                        video_path = episode_dir / f"{episode_file.stem}_video.mp4"

                    render_episode_to_video(
                        episode_data,
                        render_frame_fn,
                        video_path,
                        fps=fps,
                        max_frames=max_frames,
                    )

                    wandb.log(  # type: ignore[union-attr]
                        {f"videos/episode_{i}": wandb.Video(str(video_path), fps=fps, format="mp4")},  # type: ignore[union-attr]
                        step=global_step,
                    )

                    # Only delete temporary videos, keep permanent ones
                    if not save_videos:
                        video_path.unlink()

                except Exception as e:
                    print(f"Warning: Failed to render/log {episode_file.name}: {e}")
                    continue

        except Exception as e:
            print(f"Warning: Failed to log videos to W&B: {e}")

    def log_final(self, total_env_steps: int) -> None:
        """Send final completion metrics to W&B.

        Args:
            total_env_steps: Total environment steps completed
        """
        if self.use_wandb:
            wandb.log({"train/final_env_steps": float(total_env_steps)}, step=total_env_steps)  # type: ignore[union-attr]

    def finish(self) -> None:
        """Close the W&B run."""
        if self.use_wandb and wandb is not None:
            wandb.finish()


def init_wandb(config: Config | EvalConfig) -> Any | None:
    """Initialize a Weights & Biases run when enabled in the config.

    Args:
        config: Training or evaluation config

    Returns:
        W&B run instance or None if disabled
    """
    wandb_config = config.wandb
    if wandb_config is None or not wandb_config.enabled:
        return None

    if wandb is None:
        message = (
            "Weights & Biases tracking is enabled but the `wandb` package is not installed. "
            "Install it with `pip install wandb` to proceed."
        )
        raise RuntimeError(message) from _wandb_import_error

    init_kwargs: dict[str, Any] = _drop_none(
        {
            "project": wandb_config.project,
            "entity": wandb_config.entity,
            "group": wandb_config.group,
            "job_type": wandb_config.job_type,
            "mode": wandb_config.mode,
            "dir": wandb_config.dir,
        }
    )

    if wandb_config.run_name:
        init_kwargs["name"] = wandb_config.run_name

    if wandb_config.tags:
        init_kwargs["tags"] = list(wandb_config.tags)

    init_kwargs["config"] = config.model_dump()

    return wandb.init(**init_kwargs)
