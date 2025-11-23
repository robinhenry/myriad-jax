# W&B Sweep Configurations

This directory contains W&B sweep configurations for hyperparameter optimization.

## DQN CartPole Sweep

The `dqn_cartpole_sweep.yaml` uses Bayesian optimization to find the fastest DQN configuration to solve CartPole (195+ mean return).

### Key Hyperparameters

- **Learning rate**: 0.0001-0.01 (log-uniform)
- **Gamma**: 0.95-0.999
- **Epsilon decay**: 10k-100k steps
- **Target network frequency**: 500-5000 steps
- **Batch size**: 32-256
- **Buffer size**: 10k-100k
- **Num envs**: 1-16 (key parameter for analyzing parallelization impact)

### Quick Start

```bash
# Initialize sweep
wandb sweep configs/sweeps/dqn_cartpole_sweep.yaml

# Run agent(s) - use the sweep ID from above
wandb agent lugagne-lab/aion/<sweep_id>
```

Monitor at: `https://wandb.ai/lugagne-lab/aion/sweeps/<sweep_id>`
