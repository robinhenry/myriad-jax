# Example 02: PQN on CartPole

PQN with 32 parallel environments on `cartpole-control`. Compared to example 01,
this shows how population-scale parallelism speeds up data collection and
stabilises training — PQN collects a full minibatch per update step without a
replay buffer.

## How to run

```bash
cd examples/02_pqn_cartpole_control
./run.sh
```

Or with Hydra overrides:

```bash
MYRIAD_CONFIG_PATH=$(pwd)/configs myriad train --config-name pqn_cartpole \
    run.num_envs=64
```

## Viewing results

Training output is written to `outputs/` under a timestamped subdirectory.
To enable W&B logging, set `wandb.enabled=true`.
