"""Metrics logger that handles both local capture and remote logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .logging_utils import prepare_metrics_host
from .remote_logger import RemoteLogger
from .types import EvaluationMetrics, TrainingMetrics


class MetricsLogger:
    """Unified logger for training and evaluation metrics.

    Handles local metric capture (for return values) and delegates remote logging
    (W&B, etc.) to RemoteLogger. Metrics are processed once and sent
    to all configured destinations.
    """

    def __init__(self, wandb_run: Any | None = None):
        """Initialize the metrics logger.

        Args:
            wandb_run: W&B run instance (None to disable remote logging)
        """
        # Remote logging handler
        self.remote_logger = RemoteLogger(wandb_run=wandb_run)

        # Local metric storage
        self.training_metrics = TrainingMetrics(
            global_steps=[],
            steps_per_env=[],
            agent_metrics={},
        )

        self.eval_metrics = EvaluationMetrics(
            global_steps=[],
            steps_per_env=[],
            episode_returns=[],
            episode_lengths=[],
            mean_return=[],
            std_return=[],
            mean_length=[],
        )

    def log_training_step(
        self,
        global_step: int,
        steps_per_env: int,
        metrics_history: Any,
        steps_this_chunk: int,
    ) -> None:
        """Log training metrics for a single checkpoint.

        Args:
            global_step: Global environment steps
            steps_per_env: Steps per individual environment
            metrics_history: Raw metrics from the training loop
            steps_this_chunk: Number of steps in this chunk
        """
        # Convert metrics to host
        metrics_host = prepare_metrics_host(metrics_history, steps_this_chunk)

        if not metrics_host:
            return

        # Capture to local storage
        self.training_metrics.global_steps.append(global_step)
        self.training_metrics.steps_per_env.append(steps_per_env)

        # TODO: do we really need to manually specify which metrics we want to extract here? This hardcoding seems
        # not-ideal. Surely we just want to extract all available metrics? I'm not sure what's best.
        # Extract common metrics
        if "loss" in metrics_host:
            if self.training_metrics.loss is None:
                self.training_metrics.loss = []
            self.training_metrics.loss.append(float(metrics_host["loss"][-1]))

        if "reward" in metrics_host:
            if self.training_metrics.reward is None:
                self.training_metrics.reward = []
            self.training_metrics.reward.append(float(metrics_host["reward"][-1]))

        # Store agent-specific metrics
        for metric_name, metric_values in metrics_host.items():
            if metric_name not in ["loss", "reward"]:  # Already handled
                if self.training_metrics.agent_metrics is None:
                    self.training_metrics.agent_metrics = {}
                if metric_name not in self.training_metrics.agent_metrics:
                    self.training_metrics.agent_metrics[metric_name] = []
                self.training_metrics.agent_metrics[metric_name].append(float(metric_values[-1]))

        # Send to remote logging services (W&B, etc.)
        self.remote_logger.log_train(metrics_host, global_step, steps_per_env)

    def log_evaluation(
        self,
        global_step: int,
        steps_per_env: int,
        eval_results: dict[str, Any],
    ) -> None:
        """Log evaluation metrics for a single checkpoint.

        Args:
            global_step: Global environment steps
            steps_per_env: Steps per individual environment
            eval_results: Dictionary with 'episode_return', 'episode_length', 'dones'
        """
        eval_returns = eval_results.get("episode_return")
        eval_lengths = eval_results.get("episode_length")

        # Capture to local storage
        self.eval_metrics.global_steps.append(global_step)
        self.eval_metrics.steps_per_env.append(steps_per_env)

        # TODO: it looks as though we may be repeating the mean/std computations in multiple places. For example,
        # they also happen in evaluation.py . We probably want to be consistent and only do it in one place.
        if eval_returns is not None:
            self.eval_metrics.episode_returns.append(np.asarray(eval_returns))
            self.eval_metrics.mean_return.append(float(np.mean(eval_returns)))
            self.eval_metrics.std_return.append(float(np.std(eval_returns)))

        if eval_lengths is not None:
            self.eval_metrics.episode_lengths.append(np.asarray(eval_lengths))
            self.eval_metrics.mean_length.append(float(np.mean(eval_lengths)))

        # Send to remote logging services (W&B, etc.)
        self.remote_logger.log_eval(eval_results, global_step, steps_per_env)

    def log_episodes(self, episode_dir: Path, steps_per_env: int) -> None:
        """Log saved episodes to W&B as artifacts.

        Args:
            episode_dir: Path to directory containing saved episodes
            steps_per_env: Global environment steps (for artifact versioning)
        """
        # TODO: we also save episodes to local files in evaluate() from evaluation.py, for example. It seems we're not
        # being very consistent with the way we log things. Generally speaking, we want to log to memory (so that our
        # train/eval) functions can return the data/metrics, and also to the remote wandb server. In doing so, we're
        # also saving those to the wandb folder on the local machine, which is great. We need to make this process
        # more consistent across the platform codebase. There should be a single clean and clear workflow for logging
        # things.

        # Send to remote logging services (W&B, etc.)
        self.remote_logger.log_episodes(episode_dir, steps_per_env)

    def log_final(self, total_env_steps: int) -> None:
        """Log final training completion.

        Args:
            total_env_steps: Total environment steps completed
        """
        # Send to remote logging services (W&B, etc.)
        self.remote_logger.log_final(total_env_steps)

    def get_results(self) -> tuple[TrainingMetrics, EvaluationMetrics]:
        """Get captured metrics.

        Returns:
            Tuple of (training_metrics, eval_metrics)
        """
        return self.training_metrics, self.eval_metrics
