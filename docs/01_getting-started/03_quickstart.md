# Quickstart

Train your first agent on CartPole in under 2 minutes.

## Run with default config

```bash
myriad train
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
myriad train
```

View results at [wandb.ai](https://wandb.ai).

## Scale up to 10,000 environments

```bash
myriad train run.num_envs=10000
```

## Try a different environment

```bash
myriad train env=cartpole_sysid
```

## Try a different agent

```bash
myriad train agent=pqn
```

## Combine options

```bash
myriad train \
  env=cartpole_sysid \
  agent=pqn \
  run.num_envs=50000 \
  run.steps_per_env=20
```

## What's happening

1. Hydra composes config from `configs/`
2. Pydantic validates the config
3. Platform initializes vectorized environments
4. JAX jit-compiles the training loop
5. Training runs (first step is slow due to compilation)
6. Metrics logged to W&B

## Next steps

- [Core Concepts](../02_user-guide/01_concepts.md): Understand the three-layer architecture
- [Custom Environment](../02_user-guide/02_custom_env.md): Implement your own physics
- [Running Experiments](../02_user-guide/04_running_experiments.md): Advanced configuration options
