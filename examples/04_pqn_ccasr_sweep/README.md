# Example 04: PQN Hyperparameter Sweep on ccasr-gfp-control

Bayesian hyperparameter optimization for PQN on the GFP control task, using W&B
sweeps with Hyperband early stopping. The sweep is repeated across three parallelism
levels (`num_envs` ∈ {512, 1024, 16384}) to characterize the tradeoff between
population size and final policy quality.

## What this example does

Each `num_envs` level gets its own W&B sweep. Within each sweep:

- **Bayesian optimization** proposes hyperparameter configurations, learning from
  the performance of previous runs to focus budget on promising regions.
- **Hyperband early stopping** culls poor configs at ~20% of training budget,
  concentrating compute on configs that show early promise.
- **3 seeds** are included as a categorical parameter. W&B treats seed like any
  other categorical — promising configs naturally get sampled across all three seeds.
  Group by `run.seed` in the W&B UI to inspect per-config variance.

### Parameters swept

See `configs/pqn_ccasr_sweep.yaml` for the full parameter list with distributions
and ranges. `epsilon_decay_fraction` and `lr_decay_fraction` are resolved to absolute
step counts at training time based on `steps_per_env`, `rollout_steps`, `num_epochs`,
and `num_minibatches` — so the fractions remain meaningful across all `num_envs`
levels and all sampled `rollout_steps` values.

### Training budget

Each run trains for `steps_per_env=5_000` steps per environment, giving
`5_000 × num_envs` total environment steps (e.g. ~5M at 1024 envs, ~82M at 16384).
Evaluations run every 500 steps/env, giving 10 eval points per run.

## Files

```
04_pqn_ccasr_sweep/
├── configs/
│   └── pqn_ccasr_sweep.yaml   # W&B sweep definition (all parameters, 1024-env canonical)
├── create_sweeps.py            # registers one W&B sweep per num_envs level
├── run_sweep.sh                # convenience: create sweeps + launch local agents
└── README.md
```

## How to run

### Prerequisites

```bash
wandb login                         # authenticate once
export WANDB_ENTITY=your-entity     # your W&B username or team
```

### 1. Create sweeps (registers with W&B, no training starts)

```bash
python examples/04_pqn_ccasr_sweep/create_sweeps.py --project myriad-ccasr
# Output: one sweep ID per num_envs level
```

To create sweeps for specific levels only:

```bash
python examples/04_pqn_ccasr_sweep/create_sweeps.py --project myriad-ccasr --levels 512 1024
```

### 2. Launch agents

Each agent runs one training job at a time. Launch one per available GPU:

```bash
wandb agent <sweep_id>
```

### All-in-one (create + launch local agents)

```bash
# Defaults: all 3 levels, 1 agent per level, project=myriad-ccasr
./examples/04_pqn_ccasr_sweep/run_sweep.sh

# Custom:
WANDB_PROJECT=my-project NUM_AGENTS=2 ./examples/04_pqn_ccasr_sweep/run_sweep.sh
```

## Analyzing results

In the W&B UI, filter runs by `wandb.group` to isolate a specific `num_envs` level.
Group by `run.seed` to separate variance from hyperparameter signal.

The sweep objective is `eval/episode_return/mean` (logged every 500 steps/env,
giving 10 eval points per run). Hyperband first culls at 2 logged values (~20% of
training), so weak configs are killed early without wasting the full budget.
