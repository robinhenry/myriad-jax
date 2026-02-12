"""Myriad: JAX-based Digital Twin Engine for Control & Decision-Making.

Massively parallel experiments on GPU/TPU for biological/physics systems,
active system identification, model-based, and model-free control.

Main Programmatic API:
    - create_config: Create training configurations easily
    - create_eval_config: Create evaluation-only configurations
    - train_and_evaluate: Run a training experiment
    - evaluate: Evaluate agents without training
    - load_run: Load saved run artifacts (config, results, metadata)

Example:
    >>> from myriad import create_config, train_and_evaluate, load_run
    >>>
    >>> # Train an agent
    >>> config = create_config(
    ...     env="cartpole-control",
    ...     agent="dqn",
    ...     num_envs=1000,
    ...     steps_per_env=100
    ... )
    >>> results = train_and_evaluate(config)
    >>> print(results.summary())
    >>>
    >>> # Load a saved run
    >>> run = load_run("outputs/2026-02-12/14-30-52")
    >>> print(run.results.summary())
"""

from myriad.configs.builder import create_config, create_eval_config
from myriad.configs.default import Config, EvalConfig, config_to_eval_config
from myriad.platform import (
    RunArtifacts,
    evaluate,
    load_run,
    load_run_checkpoint,
    load_run_config,
    load_run_metadata,
    load_run_results,
    train_and_evaluate,
)

__all__ = [
    # Config creation
    "create_config",
    "create_eval_config",
    # Training and evaluation
    "train_and_evaluate",
    "evaluate",
    # Config types
    "Config",
    "EvalConfig",
    "config_to_eval_config",
    # Run artifact loading (new)
    "load_run",
    "load_run_config",
    "load_run_results",
    "load_run_checkpoint",
    "load_run_metadata",
    "RunArtifacts",
]
