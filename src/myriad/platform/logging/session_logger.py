"""Unified logger for training and evaluation sessions.

SessionLogger is the main entry point for all logging in Myriad. It composes
three backends to handle different destinations:
1. Memory - Captures metrics for return values
2. Disk - Saves episode trajectories
3. Remote (W&B) - Logs metrics and artifacts
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from myriad.configs.default import Config, EvalConfig
from myriad.platform.types import EvaluationMetrics, TrainingMetrics

from .backends.disk import DiskBackend
from .backends.memory import MemoryBackend
from .backends.wandb import WandbBackend, init_wandb


class SessionLogger:
    """Unified logger for training and evaluation sessions.

    Handles three destinations automatically:
    1. Memory - Captures metrics for return values
    2. Disk - Saves episode trajectories
    3. Remote - Logs to W&B (metrics + artifacts)

    Example:
        >>> logger = SessionLogger.for_training(config)
        >>> logger.log_training_step(...)
        >>> logger.log_evaluation(..., save_episodes=True)
        >>> training_metrics, eval_metrics = logger.finalize()
    """

    def __init__(
        self,
        wandb_run: Any | None,
        run_dir: Path,
        seed: int = 0,
    ) -> None:
        """Initialize the session logger.

        Args:
            wandb_run: W&B run instance (None to disable remote logging)
            run_dir: Base directory for outputs (episode files, etc.)
            seed: Random seed for metadata
        """
        self._wandb_run = wandb_run
        self._run_dir = run_dir
        self._seed = seed

        # Initialize backends
        self._memory = MemoryBackend()
        self._wandb = WandbBackend(wandb_run=wandb_run)

        # Disk backend base dir: prefer W&B local dir, fallback to run_dir
        episode_base_dir = self._get_episode_base_dir()
        self._disk = DiskBackend(base_dir=episode_base_dir, seed=seed)

    @classmethod
    def for_training(cls, config: Config) -> "SessionLogger":
        """Create a logger for training sessions.

        Args:
            config: Training configuration

        Returns:
            Configured SessionLogger instance
        """
        wandb_run = init_wandb(config)
        run_dir = Path.cwd()  # Hydra sets this to the output dir
        return cls(wandb_run=wandb_run, run_dir=run_dir, seed=config.run.seed)

    @classmethod
    def for_evaluation(cls, config: EvalConfig) -> "SessionLogger":
        """Create a logger for evaluation-only sessions.

        Args:
            config: Evaluation configuration

        Returns:
            Configured SessionLogger instance
        """
        wandb_run = init_wandb(config)
        run_dir = Path.cwd()
        return cls(wandb_run=wandb_run, run_dir=run_dir, seed=config.run.seed)

    def _get_episode_base_dir(self) -> Path:
        """Get the base directory for episode storage.

        Prefers W&B local directory when available (keeps everything together),
        otherwise falls back to run_dir/episodes/.
        """
        if self._wandb.local_dir is not None:
            return self._wandb.local_dir / "episodes"
        return self._run_dir / "episodes"

    # --- Training API ---

    def log_training_step(
        self,
        global_step: int,
        steps_per_env: int,
        metrics_history: dict[str, Any],
        steps_this_chunk: int,
    ) -> None:
        """Log training metrics.

        Handles memory capture + W&B logging.

        Args:
            global_step: Global environment steps
            steps_per_env: Steps per individual environment
            metrics_history: Raw metrics from the training loop
            steps_this_chunk: Number of steps in this chunk
        """
        # Memory backend processes and returns host metrics
        metrics_host = self._memory.log_training_step(global_step, steps_per_env, metrics_history, steps_this_chunk)

        # Send to W&B
        if metrics_host:
            self._wandb.log_training(metrics_host, global_step, steps_per_env)

    # --- Evaluation API ---

    def log_evaluation(
        self,
        global_step: int,
        steps_per_env: int,
        eval_results: dict[str, Any],
        save_episodes: bool = False,
        episode_save_count: int | None = None,
    ) -> Path | None:
        """Log evaluation results.

        One call handles:
        - Captures metrics to memory
        - Saves episodes to disk (if save_episodes=True)
        - Logs metrics to W&B
        - Uploads episode artifacts to W&B

        Args:
            global_step: Global environment steps
            steps_per_env: Steps per individual environment
            eval_results: Dictionary with 'episode_return', 'episode_length', 'dones',
                          and optionally 'episodes' (trajectory data)
            save_episodes: If True, save episodes to disk and log to W&B
            episode_save_count: Number of episodes to save (None = all available)

        Returns:
            Path to saved episodes directory (if saved), else None
        """
        # Memory capture
        self._memory.log_evaluation(global_step, steps_per_env, eval_results)

        # W&B metrics
        self._wandb.log_evaluation(eval_results, global_step, steps_per_env)

        # Episode persistence
        episode_dir = None
        if save_episodes and "episodes" in eval_results:
            # Default to all available episodes
            if episode_save_count is None:
                episode_lengths = eval_results.get("episode_length")
                episode_save_count = len(episode_lengths) if episode_lengths is not None else 0

            if episode_save_count > 0:
                episode_dir = self._disk.save_episodes(eval_results, global_step, steps_per_env, episode_save_count)

                # Log artifact to W&B
                if episode_dir is not None:
                    self._wandb.log_episodes(episode_dir, global_step)

        return episode_dir

    # --- Lifecycle ---

    def log_final(self, total_env_steps: int) -> None:
        """Log final training completion.

        Args:
            total_env_steps: Total environment steps completed
        """
        self._wandb.log_final(total_env_steps)

    def finalize(self) -> tuple[TrainingMetrics, EvaluationMetrics]:
        """Finalize session: close W&B and return captured metrics.

        Returns:
            Tuple of (training_metrics, eval_metrics)
        """
        self._wandb.finish()
        return self._memory.get_results()

    # --- Video Rendering ---

    def log_videos(
        self,
        episode_dir: Path,
        render_frame_fn: Callable[[np.ndarray], np.ndarray],
        global_step: int,
        fps: int = 50,
        max_episodes: int | None = None,
    ) -> None:
        """Render saved episodes to videos and log to W&B.

        Args:
            episode_dir: Path to directory containing .npz episode files
            render_frame_fn: Function that takes observation array and returns RGB frame
            global_step: Global environment steps (for W&B logging step)
            fps: Frames per second for rendered videos
            max_episodes: Maximum number of episodes to render (None = all)
        """
        self._wandb.log_videos(
            episode_dir=episode_dir,
            render_frame_fn=render_frame_fn,
            global_step=global_step,
            fps=fps,
            max_episodes=max_episodes,
        )

    # --- Properties ---

    @property
    def wandb_run(self) -> Any | None:
        """Get the underlying W&B run instance."""
        return self._wandb_run

    @property
    def episode_base_dir(self) -> Path:
        """Get the base directory for episode storage."""
        return self._disk.base_dir
