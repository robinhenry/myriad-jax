"""In-memory metric capture backend.

Stores metrics in TrainingMetrics/EvaluationMetrics dataclasses for return values.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from myriad.platform.logging.utils import prepare_metrics_host
from myriad.platform.types import EvaluationMetrics, TrainingMetrics


class MemoryBackend:
    """Backend for capturing metrics in memory.

    Metrics are stored in TrainingMetrics and EvaluationMetrics dataclasses
    and can be retrieved at the end of a session.
    """

    def __init__(self) -> None:
        """Initialize empty metric storage."""
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
    ) -> dict[str, Any]:
        """Capture training metrics to memory.

        Args:
            global_step: Global environment steps
            steps_per_env: Steps per individual environment
            metrics_history: Raw metrics from the training loop
            steps_this_chunk: Number of steps in this chunk

        Returns:
            Processed metrics dict (for passing to other backends)
        """
        metrics_host = prepare_metrics_host(metrics_history, steps_this_chunk)

        if not metrics_host:
            return {}

        self.training_metrics.global_steps.append(global_step)
        self.training_metrics.steps_per_env.append(steps_per_env)

        # Extract all metrics - "loss" and "reward" are stored in dedicated
        # fields for backward compatibility, all others go to agent_metrics
        for metric_name, metric_values in metrics_host.items():
            metric_value = float(metric_values[-1])

            if metric_name == "loss":
                if self.training_metrics.loss is None:
                    self.training_metrics.loss = []
                self.training_metrics.loss.append(metric_value)
            elif metric_name == "reward":
                if self.training_metrics.reward is None:
                    self.training_metrics.reward = []
                self.training_metrics.reward.append(metric_value)
            else:
                if self.training_metrics.agent_metrics is None:
                    self.training_metrics.agent_metrics = {}
                if metric_name not in self.training_metrics.agent_metrics:
                    self.training_metrics.agent_metrics[metric_name] = []
                self.training_metrics.agent_metrics[metric_name].append(metric_value)

        return metrics_host

    def log_evaluation(
        self,
        global_step: int,
        steps_per_env: int,
        eval_results: dict[str, Any],
    ) -> None:
        """Capture evaluation metrics to memory.

        Args:
            global_step: Global environment steps
            steps_per_env: Steps per individual environment
            eval_results: Dictionary with 'episode_return', 'episode_length', 'dones'
        """
        eval_returns = eval_results.get("episode_return")
        eval_lengths = eval_results.get("episode_length")

        self.eval_metrics.global_steps.append(global_step)
        self.eval_metrics.steps_per_env.append(steps_per_env)

        if eval_returns is not None:
            self.eval_metrics.episode_returns.append(np.asarray(eval_returns))
            self.eval_metrics.mean_return.append(float(np.mean(eval_returns)))
            self.eval_metrics.std_return.append(float(np.std(eval_returns)))

        if eval_lengths is not None:
            self.eval_metrics.episode_lengths.append(np.asarray(eval_lengths))
            self.eval_metrics.mean_length.append(float(np.mean(eval_lengths)))

    def get_results(self) -> tuple[TrainingMetrics, EvaluationMetrics]:
        """Get captured metrics.

        Returns:
            Tuple of (training_metrics, eval_metrics)
        """
        return self.training_metrics, self.eval_metrics
