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


def _to_flat_config(config: Config | EvalConfig) -> dict[str, Any]:
    """Flatten a nested config dict to dot-notation keys for W&B.

    Converts {"agent": {"learning_rate": 0.001}} to {"agent.learning_rate": 0.001},
    matching the flat dotted-key format the W&B sweep agent uses. Excludes the
    "wandb" sub-config (meta-config, not experiment config).
    """
    raw = config.model_dump()
    raw.pop("wandb", None)

    flat: dict[str, Any] = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat[f"{section}.{key}"] = value
        else:
            flat[section] = values
    return flat


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
        self._best_eval_return: float = float("-inf")

        if self.use_wandb:
            # Declare steps/env as the x-axis for all train/* and eval/* charts.
            # hidden=True prevents it from appearing as its own noise chart in the UI.
            wandb.define_metric("steps/env", hidden=True)  # type: ignore[union-attr]
            wandb.define_metric("train/*", step_metric="steps/env")  # type: ignore[union-attr]
            wandb.define_metric("eval/*", step_metric="steps/env")  # type: ignore[union-attr]
            # eval metrics are two levels deep; W&B only supports single-level glob suffixes.
            wandb.define_metric("eval/return/*", step_metric="steps/env")  # type: ignore[union-attr]
            wandb.define_metric("eval/length/*", step_metric="steps/env")  # type: ignore[union-attr]
            # Secondary metrics: always logged but hidden by default to keep charts clean.
            for _metric in [
                "eval/return/std",
                "eval/return/min",
                "eval/return/max",
                "eval/length/std",
                "eval/length/min",
                "eval/length/max",
                "eval/termination_rate",
            ]:
                wandb.define_metric(_metric, step_metric="steps/env", hidden=True)  # type: ignore[union-attr]

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
        _global_step: int,
        steps_per_env: int,
    ) -> None:
        """Send training metrics to W&B.

        Args:
            metrics_host: Raw metrics dictionary
            _global_step: Unused; kept for API compatibility with callers
            steps_per_env: Steps per individual environment (used as W&B x-axis step)
        """
        if not self.use_wandb:
            return

        train_payload = build_train_payload(metrics_host)
        if train_payload:
            train_payload["steps/env"] = steps_per_env
            wandb.log(train_payload, step=steps_per_env)  # type: ignore[union-attr]

    def log_evaluation(
        self,
        eval_results: dict[str, Any],
        _global_step: int,
        steps_per_env: int,
    ) -> None:
        """Send evaluation metrics to W&B.

        Args:
            eval_results: Dictionary with 'episode_return', 'episode_length', 'dones'
            _global_step: Unused; kept for API compatibility with callers
            steps_per_env: Steps per individual environment (used as W&B x-axis step)
        """
        if not self.use_wandb:
            return

        eval_payload: dict[str, float] = {}

        eval_returns = eval_results.get("episode_return")
        eval_lengths = eval_results.get("episode_length")
        eval_dones = eval_results.get("dones")

        if eval_returns is not None:
            eval_payload.update(summarize_metric("eval/", "return", eval_returns))

        if eval_lengths is not None:
            eval_payload.update(summarize_metric("eval/", "length", eval_lengths))

        if eval_dones is not None:
            eval_payload["eval/termination_rate"] = float(np.asarray(eval_dones, dtype=np.float32).mean())

        if eval_payload:
            eval_payload["steps/env"] = steps_per_env
            wandb.log(eval_payload, step=steps_per_env)  # type: ignore[union-attr]

        # Track best-ever mean eval return as a summary key. Sweep agents and the
        # runs table use this instead of the last logged value.
        current_mean = eval_payload.get("eval/return/mean")
        if current_mean is not None and current_mean > self._best_eval_return:
            self._best_eval_return = current_mean
            try:
                self.wandb_run.summary["eval/return/best"] = self._best_eval_return  # type: ignore[union-attr]
            except (AttributeError, TypeError):
                pass

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

    def log_run_summary(self, config: Config) -> None:
        """Write key config values to W&B summary for easy runs-table visibility.

        Values written to wandb.run.summary appear as columns in the W&B runs table
        by default, making it easy to filter/sort runs by config parameters.

        Args:
            config: Training configuration
        """
        if not self.use_wandb or self.wandb_run is None:
            return
        try:
            self.wandb_run.summary.update(
                {
                    "cfg/num_envs": config.run.num_envs,
                    "cfg/steps_per_env": config.run.steps_per_env,
                    "cfg/rollout_steps": config.run.rollout_steps,
                    "cfg/eval_rollouts": config.run.eval_rollouts,
                    "cfg/agent": config.agent.name,
                    "cfg/env": config.env.name,
                }
            )
        except (AttributeError, TypeError):
            pass  # Gracefully skip if wandb_run doesn't support summary (e.g. in tests)

    def finish(self, exit_code: int = 0) -> None:
        """Close the W&B run."""
        if self.use_wandb and wandb is not None:
            wandb.finish(exit_code=exit_code)


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

    # If a run is already active (e.g. sweep_main called wandb.init() to read
    # wandb.config), reuse it and add only the keys the sweep agent didn't set.
    # Sweep-controlled keys are locked and will warn if we try to overwrite them.
    if wandb.run is not None:
        existing = set(dict(wandb.run.config).keys())
        new_keys = {k: v for k, v in _to_flat_config(config).items() if k not in existing}
        if new_keys:
            wandb.run.config.update(new_keys)
        return wandb.run

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

    init_kwargs["config"] = _to_flat_config(config)

    return wandb.init(**init_kwargs)
