"""Myriad: JAX-based Digital Twin Engine for Control & Decision-Making.

Massively parallel experiments on GPU/TPU for biological/physics systems,
active system identification, model-based, and model-free control.

Main Programmatic API:
    - create_config: Create training configurations easily
    - create_eval_config: Create evaluation-only configurations
    - train_and_evaluate: Run a training experiment
    - evaluate: Evaluate agents without training

Example:
    >>> from myriad import create_config, train_and_evaluate
    >>> config = create_config(
    ...     env="cartpole-control",
    ...     agent="dqn",
    ...     num_envs=1000,
    ...     steps_per_env=100
    ... )
    >>> results = train_and_evaluate(config)
    >>> print(results.summary())
"""

from myriad.configs.builder import create_config, create_eval_config
from myriad.platform import evaluate, train_and_evaluate

__all__ = [
    "create_config",
    "create_eval_config",
    "train_and_evaluate",
    "evaluate",
]
