"""Result types returned from platform training functions."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chex
import numpy as np
from flax import struct

from myriad.configs.default import Config, EvalConfig
from myriad.envs.environment import EnvironmentState
from myriad.utils.config import save_config

from .artifact_helpers import save_results_to_disk
from .constants import RESULTS_FILENAME
from .serialization import load_agent_state, save_agent_state


@struct.dataclass
class TrainingEnvState:
    """Container for the state of a training environment, including observations.

    This struct groups the environment state with the current observations for
    efficient handling in training loops. The observations are stored as arrays
    (not NamedTuples) to ensure compatibility with platform utilities like
    where_mask and mask_tree.
    """

    env_state: EnvironmentState
    obs: chex.Array


@dataclass
class TrainingMetrics:
    """Training metrics collected at each logging checkpoint.

    Metrics are captured at intervals defined by ``log_frequency`` in the run config.
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
    """Agent-specific metrics (e.g., ``q_value``, ``td_error`` for DQN)."""


@dataclass
class EvaluationMetrics:
    """Evaluation metrics collected at each evaluation checkpoint.

    Metrics are captured at intervals defined by ``eval_frequency`` in the run config.
    Each list contains one entry per evaluation checkpoint.
    """

    global_steps: list[int]
    """Global environment steps at each evaluation (total across all envs)."""

    steps_per_env: list[int]
    """Steps per individual environment at each evaluation."""

    episode_returns: list[np.ndarray]
    """Raw episode returns from each evaluation. Each array contains returns from ``eval_rollouts`` episodes."""

    episode_lengths: list[np.ndarray]
    """Raw episode lengths from each evaluation. Each array contains lengths from ``eval_rollouts`` episodes."""

    mean_return: list[float]
    """Mean episode return at each evaluation."""

    std_return: list[float]
    """Standard deviation of episode returns at each evaluation."""

    mean_length: list[float]
    """Mean episode length at each evaluation."""


@dataclass
class TrainingResults:
    """Complete results from a training run.

    Returned by :func:`~myriad.platform.train_and_evaluate` and contains everything needed to:

    - Use the trained agent for inference
    - Analyze training progress
    - Reproduce the run
    - Resume training (optional)
    """

    agent_state: Any
    """Trained agent state (can be used for inference with ``agent.select_action()``)."""

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
            - final_eval_return_mean: Mean return from last evaluation checkpoint
            - final_eval_return_std: Std deviation from last evaluation checkpoint
            - training_steps_per_env: Environment steps per individual environment
            - training_global_steps: Total global environment steps across all envs
            - num_eval_checkpoints: Number of evaluations performed
        """
        return {
            "final_eval_return_mean": self.eval_metrics.mean_return[-1] if self.eval_metrics.mean_return else 0.0,
            "final_eval_return_std": self.eval_metrics.std_return[-1] if self.eval_metrics.std_return else 0.0,
            "training_steps_per_env": self.eval_metrics.steps_per_env[-1] if self.eval_metrics.steps_per_env else 0,
            "training_global_steps": (
                self.training_metrics.global_steps[-1] if self.training_metrics.global_steps else 0
            ),
            "num_eval_checkpoints": len(self.eval_metrics.global_steps),
        }

    def __repr__(self) -> str:
        """Human-readable summary of training results."""
        summary = self.summary()
        return (
            f"TrainingResults(\n"
            f"  final_eval_return={summary['final_eval_return_mean']:.1f} ± {summary['final_eval_return_std']:.1f},\n"
            f"  steps_per_env={summary['training_steps_per_env']:,},\n"
            f"  global_steps={summary['training_global_steps']:,},\n"
            f"  num_evals={summary['num_eval_checkpoints']}\n"
            f")"
        )

    def save(self, directory: Path | str, save_checkpoint: bool = False) -> None:
        """Save results and optionally agent checkpoint to directory.

        Saves:
        - .hydra/config.yaml: Configuration used for the run
        - results.pkl: TrainingResults without agent_state
        - checkpoints/final.msgpack: Agent state (if save_checkpoint=True)

        Note: The agent_state is excluded from results.pkl and saved separately
        using Flax msgpack serialization for reliability with JAX/Flax objects.

        Args:
            directory: Directory to save results to (typically Hydra output directory)
            save_checkpoint: Whether to save agent checkpoint

        Raises:
            RuntimeError: If agent checkpoint serialization fails

        Example:
            >>> results = train_and_evaluate(config)
            >>> results.save(Path.cwd(), save_checkpoint=True)
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save config to .hydra/config.yaml (matches Hydra runner output)
        save_config(self.config, directory / ".hydra" / "config.yaml")

        # Save results without agent_state
        # Create a shallow copy with agent_state and final_env_state set to None
        results_copy = TrainingResults(
            agent_state=None,  # Don't pickle agent state - use checkpoint instead
            training_metrics=self.training_metrics,
            eval_metrics=self.eval_metrics,
            config=self.config,
            final_env_state=None,  # Also exclude env state - not needed for analysis
        )

        # Use shared helper for saving results and checkpoint
        save_results_to_disk(results_copy, directory, self.agent_state, save_checkpoint)

    @staticmethod
    def load(directory: Path | str) -> "TrainingResults":
        """Load results from directory.

        Args:
            directory: Directory containing results.pkl

        Returns:
            Loaded TrainingResults object

        Example:
            >>> results = TrainingResults.load("outputs/2026-02-12/14-30-52")
            >>> print(results.summary())
        """
        with open(Path(directory) / RESULTS_FILENAME, "rb") as f:
            return pickle.load(f)

    def save_agent(self, path: str | Path) -> None:
        """Save trained agent state to file using Flax msgpack serialization.

        Args:
            path: Path to save the agent state (typically with .msgpack extension)

        Raises:
            RuntimeError: If serialization fails

        Example:
            >>> results = train_and_evaluate(config)
            >>> results.save_agent("trained_agent.msgpack")
        """
        save_agent_state(self.agent_state, path)

    @staticmethod
    def load_agent(path: str | Path) -> Any:
        """Load agent state from file.

        Args:
            path: Path to the saved agent state file

        Returns:
            The loaded agent state (can be passed to evaluate())

        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If deserialization fails

        Example:
            >>> agent_state = TrainingResults.load_agent("trained_agent.msgpack")
            >>> results = evaluate(config, agent_state=agent_state)
        """
        return load_agent_state(path)


@dataclass
class EvaluationResults:
    """Results from an evaluation-only run.

    Returned by evaluate() and contains:

    - Summary statistics (mean, std, min, max)
    - Raw episode data (for custom analysis)
    - Optional trajectory data (if return_episodes=True)
    - Metadata (seed, num_episodes, config)
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
    """Raw episode returns. Shape: ``(num_episodes,)``"""

    episode_lengths: np.ndarray
    """Raw episode lengths. Shape: ``(num_episodes,)``"""

    # --- Metadata ---
    num_episodes: int
    """Number of episodes evaluated."""

    seed: int
    """Random seed used for evaluation."""

    # --- Configuration ---
    config: EvalConfig
    """Evaluation configuration used (for reproducibility)."""

    # --- Optional Trajectory Data ---
    episodes: dict[str, np.ndarray] | None = None
    """Full episode trajectories (if return_episodes=True).
    Contains:
    - observations: Shape ``(num_episodes, max_steps, obs_dim)``
    - actions: Shape ``(num_episodes, max_steps, ...)``
    - rewards: Shape ``(num_episodes, max_steps)``
    - dones: Shape ``(num_episodes, max_steps)``
    """

    # --- Optional Agent State ---
    agent_state: Any | None = None
    """Agent state used for evaluation (if provided)."""

    # --- Output Directory ---
    run_dir: Path | None = None
    """Directory where evaluation outputs were saved."""

    def save(self, directory: Path | str, save_checkpoint: bool = False) -> None:
        """Save results and optionally agent checkpoint to directory.

        Saves:
        - .hydra/config.yaml: Configuration used for the run (if config is present)
        - results.pkl: EvaluationResults without agent_state
        - checkpoints/final.msgpack: Agent state (if save_checkpoint=True and agent_state exists)

        Note: The agent_state is excluded from results.pkl and saved separately
        using Flax msgpack serialization for reliability with JAX/Flax objects.

        Args:
            directory: Directory to save results to (typically Hydra output directory)
            save_checkpoint: Whether to save agent checkpoint

        Raises:
            RuntimeError: If agent checkpoint serialization fails

        Example:
            >>> results = evaluate(config, agent_state=agent_state)
            >>> results.save(Path.cwd(), save_checkpoint=True)
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save config to .hydra/config.yaml (matches Hydra runner output)
        save_config(self.config, directory / ".hydra" / "config.yaml")

        # Save results without agent_state
        # Create a shallow copy with agent_state set to None
        results_copy = EvaluationResults(
            mean_return=self.mean_return,
            std_return=self.std_return,
            min_return=self.min_return,
            max_return=self.max_return,
            mean_length=self.mean_length,
            std_length=self.std_length,
            min_length=self.min_length,
            max_length=self.max_length,
            episode_returns=self.episode_returns,
            episode_lengths=self.episode_lengths,
            num_episodes=self.num_episodes,
            seed=self.seed,
            config=self.config,  # Include config in pickled results
            episodes=self.episodes,
            agent_state=None,  # Don't pickle agent state - use checkpoint instead
        )

        # Use shared helper for saving results and checkpoint
        save_results_to_disk(results_copy, directory, self.agent_state, save_checkpoint)

    @staticmethod
    def load(directory: Path | str) -> "EvaluationResults":
        """Load results from directory.

        Args:
            directory: Directory containing results.pkl

        Returns:
            Loaded EvaluationResults object

        Example:
            >>> results = EvaluationResults.load("outputs/2026-02-12/14-30-52")
            >>> print(results.summary())
        """
        with open(Path(directory) / RESULTS_FILENAME, "rb") as f:
            return pickle.load(f)

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
