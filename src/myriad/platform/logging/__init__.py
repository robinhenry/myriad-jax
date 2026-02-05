"""Unified logging system for training and evaluation sessions.

This module provides a single unified logger (SessionLogger) that handles:
1. Memory - Captures metrics for return values (TrainingMetrics, EvaluationMetrics)
2. Disk - Saves episode trajectories to files
3. Remote - Logs to W&B (metrics + artifacts)

Usage:
    # For training
    logger = SessionLogger.for_training(config)
    logger.log_training_step(global_step, steps_per_env, metrics_history, steps_this_chunk)
    logger.log_evaluation(global_step, steps_per_env, eval_results, save_episodes=True)
    training_metrics, eval_metrics = logger.finalize()

    # For evaluation-only
    logger = SessionLogger.for_evaluation(config)
    logger.log_evaluation(0, 0, eval_results, save_episodes=True)
    logger.finalize()
"""

from .session_logger import SessionLogger

__all__ = ["SessionLogger"]
