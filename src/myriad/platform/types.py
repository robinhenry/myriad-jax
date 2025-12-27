"""Result types returned from platform training functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from myriad.configs.default import Config


@dataclass
class TrainingMetrics:
    """Training metrics collected at each logging checkpoint.

    Metrics are captured at intervals defined by log_frequency in the run config.
    Each list contains one entry per logging checkpoint.
    """

    global_steps: list[int]
    """Global environment steps at each checkpoint (total across all envs)."""

    steps_per_env: list[int]
    """Steps per individual environment at each checkpoint."""

    loss: list[float] | None = None
    """Training loss values (if available from agent)."""

    reward: list[float] | None = None
    """Mean reward per step (if available)."""

    agent_metrics: dict[str, list[float]] | None = None
    """Agent-specific metrics (e.g., q_value, td_error for DQN; policy_loss, value_loss for PPO)."""


@dataclass
class EvaluationMetrics:
    """Evaluation metrics collected at each evaluation checkpoint.

    Metrics are captured at intervals defined by eval_frequency in the run config.
    Each list contains one entry per evaluation checkpoint.
    """

    global_steps: list[int]
    """Global environment steps at each evaluation (total across all envs)."""

    steps_per_env: list[int]
    """Steps per individual environment at each evaluation."""

    episode_returns: list[np.ndarray]
    """Raw episode returns from each evaluation. Each array contains returns from eval_rollouts episodes."""

    episode_lengths: list[np.ndarray]
    """Raw episode lengths from each evaluation. Each array contains lengths from eval_rollouts episodes."""

    mean_return: list[float]
    """Mean episode return at each evaluation."""

    std_return: list[float]
    """Standard deviation of episode returns at each evaluation."""

    mean_length: list[float]
    """Mean episode length at each evaluation."""


@dataclass
class TrainingResults:
    """Complete results from a training run.

    Returned by train_and_evaluate() and contains everything needed to:
    - Use the trained agent for inference
    - Analyze training progress
    - Reproduce the run
    - Resume training (optional)
    """

    agent_state: Any
    """Trained agent state (can be used for inference with agent.select_action())."""

    training_metrics: TrainingMetrics
    """Training metrics history (loss, reward, etc.)."""

    eval_metrics: EvaluationMetrics
    """Evaluation metrics history (episode returns, lengths)."""

    config: Config
    """Configuration used for this training run (for reproducibility)."""

    final_env_state: Any | None = None
    """Final state of training environments (can be used to resume training)."""

    def summary(self) -> dict[str, float]:
        """Get summary statistics for quick inspection.

        Returns:
            Dictionary with key metrics:
            - final_eval_return_mean: Mean return from last evaluation
            - final_eval_return_std: Std deviation from last evaluation
            - total_training_steps: Total global environment steps
            - num_eval_checkpoints: Number of evaluations performed
        """
        return {
            "final_eval_return_mean": self.eval_metrics.mean_return[-1] if self.eval_metrics.mean_return else 0.0,
            "final_eval_return_std": self.eval_metrics.std_return[-1] if self.eval_metrics.std_return else 0.0,
            "total_training_steps": self.training_metrics.global_steps[-1] if self.training_metrics.global_steps else 0,
            "num_eval_checkpoints": len(self.eval_metrics.global_steps),
        }

    def __repr__(self) -> str:
        """Human-readable summary of training results."""
        summary = self.summary()
        return (
            f"TrainingResults(\n"
            f"  final_eval_return={summary['final_eval_return_mean']:.1f} ± {summary['final_eval_return_std']:.1f},\n"
            f"  total_steps={summary['total_training_steps']:,},\n"
            f"  num_evals={summary['num_eval_checkpoints']}\n"
            f")"
        )


@dataclass
class EvaluationResults:
    """Results from an evaluation-only run.

    Returned by evaluate() and contains:
    - Summary statistics (mean, std, min, max)
    - Raw episode data (for custom analysis)
    - Optional trajectory data (if return_episodes=True)
    - Metadata (seed, num_episodes)
    """

    # --- Summary Statistics ---
    mean_return: float
    """Mean episode return across all episodes."""

    std_return: float
    """Standard deviation of episode returns."""

    min_return: float
    """Minimum episode return."""

    max_return: float
    """Maximum episode return."""

    mean_length: float
    """Mean episode length (number of steps)."""

    std_length: float
    """Standard deviation of episode lengths."""

    min_length: int
    """Minimum episode length."""

    max_length: int
    """Maximum episode length."""

    # --- Raw Data ---
    episode_returns: np.ndarray
    """Raw episode returns. Shape: (num_episodes,)"""

    episode_lengths: np.ndarray
    """Raw episode lengths. Shape: (num_episodes,)"""

    # --- Metadata ---
    num_episodes: int
    """Number of episodes evaluated."""

    seed: int
    """Random seed used for evaluation."""

    # --- Optional Trajectory Data ---
    episodes: dict[str, np.ndarray] | None = None
    """Full episode trajectories (if return_episodes=True).
    Contains:
    - observations: Shape (num_episodes, max_steps, obs_dim)
    - actions: Shape (num_episodes, max_steps, ...)
    - rewards: Shape (num_episodes, max_steps)
    - dones: Shape (num_episodes, max_steps)
    """

    def summary(self) -> dict[str, float]:
        """Get summary statistics for quick inspection.

        Returns:
            Dictionary with key metrics:
            - mean_return: Mean episode return
            - std_return: Standard deviation of returns
            - min_return: Minimum return
            - max_return: Maximum return
            - mean_length: Mean episode length
            - num_episodes: Number of episodes evaluated
        """
        return {
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "min_return": self.min_return,
            "max_return": self.max_return,
            "mean_length": self.mean_length,
            "num_episodes": self.num_episodes,
        }

    def __repr__(self) -> str:
        """Human-readable summary of evaluation results."""
        return (
            f"EvaluationResults(\n"
            f"  mean_return={self.mean_return:.1f} ± {self.std_return:.1f},\n"
            f"  range=[{self.min_return:.1f}, {self.max_return:.1f}],\n"
            f"  mean_length={self.mean_length:.1f},\n"
            f"  num_episodes={self.num_episodes}\n"
            f")"
        )
