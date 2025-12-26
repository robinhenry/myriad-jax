"""Platform module for training and evaluation infrastructure."""

from .runner import evaluate, train_and_evaluate
from .types import EvaluationMetrics, TrainingMetrics, TrainingResults

__all__ = [
    "train_and_evaluate",
    "evaluate",
    "TrainingResults",
    "TrainingMetrics",
    "EvaluationMetrics",
]
