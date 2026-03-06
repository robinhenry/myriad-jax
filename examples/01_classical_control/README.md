# Example 01: DQN on CartPole

A single-environment DQN run on `cartpole-control`. This is the minimal
starting point — one agent, one environment, no parallelism. Use it to verify
your installation and to understand the training loop before scaling up.

## How to run

```bash
cd examples/01_classical_control
./run.sh
```

Or run directly with Hydra overrides:

```bash
MYRIAD_CONFIG_PATH=$(pwd)/configs myriad train --config-name dqn run.seed=1
```

## Viewing results

Training output (episode returns, checkpoints) is written to `outputs/` under a
timestamped subdirectory. To enable W&B logging, set `wandb.enabled=true` in
the config or pass it as a CLI override.
