# Running experiments

## Basic training

```bash
python scripts/train.py
```

## Override config values

```bash
python scripts/train.py \
  run.num_envs=10000 \
  run.total_timesteps=1e6 \
  agent.learning_rate=3e-4
```

## Switch config groups

```bash
python scripts/train.py \
  env=cartpole_sysid \
  agent=pqn
```

## Parameter sweeps

Coming soon: W&B sweep integration.

## Programmatic usage

```python
from myriad.configs.default import Config
from myriad.platform.runner import train_and_evaluate

config = Config(
    env={"_target_": "cartpole-control"},
    agent={"_target_": "dqn"},
    run={"num_envs": 10000, "total_timesteps": 1_000_000}
)

train_and_evaluate(config)
```

## Next steps

- [Configuration System](../contributing/configuration.md): Hydra config details (for advanced users)
- [Custom Environment Guide](custom_env.md): Implement your own environments
- [Custom Agent Guide](custom_agent.md): Implement learning algorithms
