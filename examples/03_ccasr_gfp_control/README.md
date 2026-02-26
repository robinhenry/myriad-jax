# Example 03: PQN on CcaSR-GFP Control

Approximately reproduces the main results from the CDC 2025 paper ("Control of a bi-stable
genetic system via parallelised RL"). The script evaluates three classical
baselines (random, bang-bang, PI) then trains PQN with 1024 parallel
environments for ~30 M total timesteps. Frame stacking (16 frames) is enabled
to give the agent memory of recent GFP dynamics.

## How to run

```bash
cd examples/03_ccasr_gfp_control
./run.sh
```

The script runs evaluation for the three baselines first (fast), then launches
PQN training (~30 M timesteps).

Individual agents can also be run directly:

```bash
MYRIAD_CONFIG_PATH=$(pwd)/configs myriad evaluate --config-name bangbang
MYRIAD_CONFIG_PATH=$(pwd)/configs myriad train    --config-name pqn
```

## Viewing results

Evaluation and training output is written to `outputs/`. Open `figures.ipynb`
to reproduce the paper figures from those outputs. To enable W&B logging, set
`wandb.enabled=true` in any config or as a CLI override.
