# Platform API

High-level training and evaluation infrastructure.

## Training

### train_and_evaluate

```{eval-rst}
.. autofunction:: myriad.platform.train_and_evaluate
```

**Basic Example:**

```{literalinclude} ../../examples/02_basic_training.py
:language: python
:caption: examples/02_basic_training.py
```

**Advanced Example with Custom Hyperparameters:**

```{literalinclude} ../../examples/03_advanced_training.py
:language: python
:caption: examples/03_advanced_training.py
```

## Evaluation

### evaluate

```{eval-rst}
.. autofunction:: myriad.platform.evaluate
```

**Random Baseline Example:**

```{literalinclude} ../../examples/05_random_baseline.py
:language: python
:caption: examples/05_random_baseline.py
```

**Pre-trained Agent Example:**

```{literalinclude} ../../examples/04_evaluate_pretrained.py
:language: python
:caption: examples/04_evaluate_pretrained.py
```

## Result Types

### TrainingResults

```{eval-rst}
.. autoclass:: myriad.platform.types.TrainingResults
   :members:
   :undoc-members:
   :show-inheritance:
```

Contains:
- `agent_state`: Trained agent ready for inference
- `training_metrics`: Training history (loss, Q-values, etc.)
- `eval_metrics`: Evaluation history (returns, lengths)
- `config`: Configuration used (for reproducibility)
- `final_env_state`: Final environment states

### TrainingMetrics

```{eval-rst}
.. autoclass:: myriad.platform.types.TrainingMetrics
   :members:
   :undoc-members:
   :show-inheritance:
```

### EvaluationResults

```{eval-rst}
.. autoclass:: myriad.platform.types.EvaluationResults
   :members:
   :undoc-members:
   :show-inheritance:
```

Contains:
- `summary`: Statistics (mean, std, min, max return)
- `episode_returns`: Per-episode returns
- `episode_lengths`: Per-episode lengths
- `episodes`: Full trajectories (if `return_episodes=True`)
- `metadata`: Configuration details

### EvaluationMetrics

```{eval-rst}
.. autoclass:: myriad.platform.types.EvaluationMetrics
   :members:
   :undoc-members:
   :show-inheritance:
```

## Configuration

### create_config

```{eval-rst}
.. autofunction:: myriad.configs.builder.create_config
```

### create_eval_config

```{eval-rst}
.. autofunction:: myriad.configs.builder.create_eval_config
```

**Note:** For advanced use cases, you can still use the lower-level Pydantic models directly.
See [Configuration Guide](../contributing/configuration.md) for details.

## Next Steps

- [Running Experiments](../user-guide/running_experiments.md): Detailed training guide
- [Configuration System](../contributing/configuration.md): Hydra config reference
- [Quickstart](../getting-started/quickstart.md): Your first training run
