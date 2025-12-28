"""Platform module for training and evaluation infrastructure."""

from .evaluation import evaluate
from .training import train_and_evaluate
from .types import EvaluationMetrics, EvaluationResults, TrainingMetrics, TrainingResults

__all__ = [
    "train_and_evaluate",
    "evaluate",
    "TrainingResults",
    "TrainingMetrics",
    "EvaluationMetrics",
    "EvaluationResults",
]
