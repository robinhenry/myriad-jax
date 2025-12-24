# Quickstart

Train your first agent on CartPole in under 2 minutes.

## Run with default config

```bash
python scripts/train.py
```

This trains DQN on CartPole with the default configuration.

## Monitor with W&B

Edit `configs/config.yaml`:

```yaml
wandb:
  enabled: true
  mode: online  # Change from offline
  entity: your-username
```

Run again:

```bash
python scripts/train.py
```

View results at [wandb.ai](https://wandb.ai).

## Scale up to 10,000 environments

```bash
python scripts/train.py run.num_envs=10000
```

## Try a different environment

```bash
python scripts/train.py env=cartpole_sysid
```

## Try a different agent

```bash
python scripts/train.py agent=pqn
```

## Combine options

```bash
python scripts/train.py \
  env=cartpole_sysid \
  agent=pqn \
  run.num_envs=50000 \
  run.total_timesteps=1e6
```

## What's happening

1. Hydra composes config from `configs/`
2. Pydantic validates the config
3. Platform initializes vectorized environments
4. JAX jit-compiles the training loop
5. Training runs (first step is slow due to compilation)
6. Metrics logged to W&B

## Next steps

- [Core Concepts](../user-guide/concepts.md): Understand the three-layer architecture
- [Custom Environment](../user-guide/custom_env.md): Implement your own physics
- [Running Experiments](../user-guide/running_experiments.md): Advanced configuration options
