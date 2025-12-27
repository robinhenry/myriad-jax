# Experiment Configs

Self-contained experiment configurations. Each experiment config includes all parameters needed to run that experiment.

## Philosophy

- **Defaults live in Python code** (factory functions and dataclasses)
- **Experiment configs are self-contained** - they include agent, env, run, and wandb settings
- **Base configs** (`agent/`, `env/`) provide composition targets for Hydra
- **YAML files specify experiment-specific parameters** - override Python defaults only when needed

## Usage

Run an experiment using `--config-name`:

```bash
python scripts/train.py --config-name=experiments/dqn_fast_exploration
```

Or run the default experiment (no flag needed):

```bash
python scripts/train.py
```

Override parameters via CLI:

```bash
python scripts/train.py \
  --config-name=experiments/dqn_fast_exploration \
  agent.learning_rate=1e-4 \
  run.num_envs=10000
```

## Available Experiments

### DQN CartPole Default
**File:** `dqn_cartpole_default.yaml`

Standard DQN baseline on CartPole.

**Settings:**
- Single environment (num_envs=1)
- 150k steps total
- Buffer size: 50k

### PQN CartPole Default
**File:** `pqn_cartpole_default.yaml`

Standard PQN baseline on CartPole (matches PureJaxQL reference).

**Settings:**
- 32 parallel environments
- 500k steps total (15625 steps/env × 32)
- On-policy rollout_steps=2

### DQN Fast Exploration
**File:** `dqn_fast_exploration.yaml`

Aggressive exploration decay for simple control tasks.

**Key overrides:**
- `epsilon_decay_steps: 5000` (2x faster)
- `learning_rate: 3e-4` (higher LR)

### DQN Slow Exploration
**File:** `dqn_slow_exploration.yaml`

Conservative exploration for complex tasks requiring more data.

**Key overrides:**
- `epsilon_decay_steps: 100000` (10x slower)
- `epsilon_end: 0.01` (lower final epsilon)

### CartPole Heavy Pole
**File:** `cartpole_heavy_pole.yaml`

Harder control task with heavier pole and weaker actuator.

**Key overrides:**
- `pole_mass: 0.2` (2x heavier)
- `force_magnitude: 8.0` (weaker actuator)
- `max_steps: 1000` (longer episodes)

### Gene Circuit Sinewave Tracking
**File:** `ccas_sinewave_tracking.yaml`

Dynamic target tracking with sinewave reference.

**Key overrides:**
- `target_type: "sinewave"`
- `sinewave_period_minutes: 480.0` (8h period)
- `n_horizon: 3` (see 3 future timesteps)

## Creating Custom Experiments

1. Copy an existing experiment config as a template
2. Modify agent, env, run, or wandb parameters
3. Update wandb tags/group for organization
4. Document key parameter changes in comments

Example:

```yaml
# @package _global_
# My custom experiment

defaults:
  - /agent: dqn
  - /env: cartpole_control
  - _self_

# Run parameters (required)
run:
  seed: 42
  steps_per_env: 150000
  num_envs: 1
  buffer_size: 50000
  scan_chunk_size: 2048
  eval_frequency: 1000
  eval_rollouts: 10
  eval_max_steps: 500
  log_frequency: 1000

# Override agent parameters
agent:
  learning_rate: 1e-4

# Override environment parameters
env:
  physics:
    pole_mass: 0.15

# W&B settings
wandb:
  enabled: true
  entity: lugagne-lab
  project: myriad
  group: my-experiment
  tags: ["custom"]
```
