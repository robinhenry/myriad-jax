# Platform API

High-level training and evaluation infrastructure.

## Training

### train_and_evaluate

```{eval-rst}
.. autofunction:: myriad.platform.train_and_evaluate
```

**Example:**

```python
from myriad.configs.default import Config
from myriad.platform import train_and_evaluate

config = Config(
env={"_target_": "cartpole-control"},
agent={"_target_": "dqn"},
run={
    "num_envs": 10000,
    "total_timesteps": 1_000_000,
    "eval_frequency": 10000,
}
)

results = train_and_evaluate(config)

# Access trained agent
agent_state = results.agent_state

# View training metrics
print(results.training_metrics.loss)
print(results.eval_metrics.mean_return)

# Save for later
import pickle
with open("trained_agent.pkl", "wb") as f:
pickle.dump(results.agent_state, f)
```

## Evaluation

### evaluate

```{eval-rst}
.. autofunction:: myriad.platform.evaluate
```

**Example:**

```python
from myriad.configs.default import EvalConfig
from myriad.platform import evaluate

# Evaluate a random agent
config = EvalConfig(
env_id="cartpole-control",
agent_id="random",
num_eval_rollouts=100,
seed=42
)

results = evaluate(config)
print(f"Mean return: {results.summary.mean_return:.2f}")
print(f"Std return: {results.summary.std_return:.2f}")

# Evaluate a pre-trained agent
import pickle
with open("trained_agent.pkl", "rb") as f:
agent_state = pickle.load(f)

results = evaluate(config, agent_state=agent_state)

# Get full episode trajectories
results = evaluate(config, agent_state=agent_state, return_episodes=True)
print(results.episodes.observations.shape)  # (num_rollouts, max_steps, obs_dim)
print(results.episodes.actions.shape)       # (num_rollouts, max_steps, act_dim)
print(results.episodes.rewards.shape)       # (num_rollouts, max_steps)
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

Training and evaluation are configured via Hydra and Pydantic:

```python
from myriad.configs.default import Config, RunConfig

config = Config(
env={"_target_": "cartpole-control"},
agent={"_target_": "dqn", "learning_rate": 1e-3},
run=RunConfig(
    num_envs=10000,
    total_timesteps=1_000_000,
    eval_frequency=10000,
    seed=42
)
)
```

See [Configuration Guide](../contributing/configuration.md) for details.

## Next Steps

- [Running Experiments](../user-guide/running_experiments.md): Detailed training guide
- [Configuration System](../contributing/configuration.md): Hydra config reference
- [Quickstart](../getting-started/quickstart.md): Your first training run
