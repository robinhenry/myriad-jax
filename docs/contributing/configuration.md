# Configuration System

!!! info "Advanced Topic"
    This document explains Myriad's Hydra-based configuration system in detail. Most users only need to know how to override config values (see [Running Experiments](../user-guide/running_experiments.md)). This guide is for contributors or advanced users who need to understand the configuration architecture.

Myriad uses [Hydra](https://hydra.cc) for composable configuration management. Configs are organized into three categories: `env/`, `agent/`, and `run/`.

## Configuration hierarchy

```
configs/
├── config.yaml              # Main config with defaults
├── env/                     # Environment configurations
│   ├── cartpole_control.yaml
│   ├── cartpole_sysid.yaml
│   └── ccas_ccar_v1.yaml
├── agent/                   # Agent configurations
│   ├── dqn.yaml
│   ├── pqn.yaml
│   └── random.yaml
└── run/                     # Combined run configurations
    ├── dqn_cartpole_control.yaml
    └── pqn_cartpole_control.yaml
```

## Main config (`config.yaml`)

```yaml
defaults:
  - agent: dqn
  - env: cartpole_control
  - run: dqn_cartpole_control
  - _self_

wandb:
  enabled: true
  entity: your-team         # W&B team/user
  project: myriad            # W&B project name
  group: experiment-1      # Group runs together
  job_type: train          # train, eval, etc.
  run_name: null           # Auto-generated if null
  mode: offline            # online | offline
  dir: null                # Local logging dir
  tags: []                 # Search tags
```

### Defaults list

Hydra composes configs from the `defaults` list. Order matters:

1. `agent: dqn` → Load `agent/dqn.yaml`
2. `env: cartpole_control` → Load `env/cartpole_control.yaml`
3. `run: dqn_cartpole_control` → Load `run/dqn_cartpole_control.yaml`
4. `_self_` → Apply current file's values (override previous)

### W&B integration

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | `bool` | Enable W&B logging |
| `entity` | `str` | Team or username |
| `project` | `str` | Project name (groups runs) |
| `group` | `str` | Sub-group within project |
| `job_type` | `str` | Run type (train, eval, etc.) |
| `run_name` | `str\|null` | Custom name or auto-generate |
| `mode` | `str` | `online` or `offline` |
| `dir` | `str\|null` | Local logging directory |
| `tags` | `list[str]` | Searchable tags |

## Environment config (`env/`)

### CartPole control (`cartpole_control.yaml`)

```yaml
name: cartpole-control

# Physics configuration
physics:
  gravity: 9.8              # m/s^2
  cart_mass: 1.0            # kg
  pole_mass: 0.1            # kg
  pole_length: 0.5          # m (half-length)
  force_magnitude: 10.0     # N
  dt: 0.02                  # s (timestep)

# Task configuration
task:
  max_steps: 500
  theta_threshold: 0.2094395102393195  # 12 degrees (rad)
  x_threshold: 2.4          # m
```

#### Physics parameters

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `gravity` | `float` | m/s² | Gravitational acceleration |
| `cart_mass` | `float` | kg | Cart mass |
| `pole_mass` | `float` | kg | Pole mass |
| `pole_length` | `float` | m | Half-length of pole |
| `force_magnitude` | `float` | N | Force applied per action |
| `dt` | `float` | s | Integration timestep |

#### Task parameters

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `max_steps` | `int` | - | Maximum episode length |
| `theta_threshold` | `float` | rad | Angle termination threshold |
| `x_threshold` | `float` | m | Position termination threshold |

### CartPole SysID (`cartpole_sysid.yaml`)

Same structure as control task, but observation includes belief state.

### Gene circuit (`ccas_ccar_v1.yaml`)

Contains ODE parameters for protein expression and cell growth dynamics.

## Agent config (`agent/`)

### DQN (`dqn.yaml`)

```yaml
name: dqn

# Training settings
batch_size: 64              # Transitions per update

# Optimizer settings
learning_rate: 1e-3

# RL algorithm parameters
gamma: 0.99                 # Discount factor

# Exploration schedule (epsilon-greedy)
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay_steps: 50000

# Target network updates
target_network_frequency: 1000  # Steps between target updates
tau: 1.0                    # Hard update (1.0) vs soft update (<1.0)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | 64 | Replay buffer sample size |
| `learning_rate` | `float` | 1e-3 | Adam learning rate |
| `gamma` | `float` | 0.99 | Discount factor |
| `epsilon_start` | `float` | 1.0 | Initial exploration rate |
| `epsilon_end` | `float` | 0.05 | Final exploration rate |
| `epsilon_decay_steps` | `int` | 50000 | Steps to decay epsilon |
| `target_network_frequency` | `int` | 1000 | Target network update interval |
| `tau` | `float` | 1.0 | Target update rate (1.0 = hard) |

### PQN (`pqn.yaml`)

Parametric Q-Network for continuous actions. Similar structure to DQN.

### Random (`random.yaml`)

No configurable parameters. Used as baseline.

## Run config (`run/`)

### DQN + CartPole (`dqn_cartpole_control.yaml`)

```yaml
seed: 42
steps_per_env: 150000       # Steps per environment (total = steps_per_env * num_envs)

# Environment parallelization
num_envs: 1                 # Parallel environments

# Training settings
buffer_size: 50000          # Replay buffer capacity
scan_chunk_size: 2048       # JAX scan chunk size

# Evaluation
eval_frequency: 1000        # Steps between evaluations
eval_rollouts: 10           # Episodes to average
eval_max_steps: 500         # Max steps per eval episode

# Episode saving (optional)
eval_episode_save_frequency: 0     # 0 = disabled, >0 = save every N steps
eval_episode_save_count: null      # Number to save (null = all eval_rollouts)
eval_episode_save_dir: episodes    # Output directory

# Logging
log_frequency: 1000         # Steps between log writes
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | `int` | Random seed |
| `steps_per_env` | `int` | Steps each environment will take (total = steps_per_env × num_envs) |
| `num_envs` | `int` | Parallel environments (vectorized) |
| `buffer_size` | `int` | Replay buffer capacity (off-policy only) |
| `scan_chunk_size` | `int` | JAX scan chunk size (affects compilation) |
| `eval_frequency` | `int` | Evaluation interval (steps per env) |
| `eval_rollouts` | `int` | Episodes to average for eval metrics |
| `eval_max_steps` | `int` | Max steps per eval episode |
| `eval_episode_save_frequency` | `int` | Episode save interval (0 = disabled) |
| `eval_episode_save_count` | `int\|null` | Episodes to save per checkpoint (null = all) |
| `eval_episode_save_dir` | `str` | Directory for saved episodes |
| `log_frequency` | `int` | Logging interval (steps per env) |

!!! info "Terminology Note: steps_per_env vs total_timesteps"
    Myriad uses `steps_per_env` as the primary configuration parameter, which differs from standard RL convention:

    - **`steps_per_env`** (primary): Number of steps each environment will take during training
    - **`total_timesteps`** (computed): Total environment interactions = `steps_per_env × num_envs`

    This design aligns with the "digital twin" / parallel experiments mental model: you specify how long to run each experiment, and the total computational budget scales naturally with parallelism.

    For RL sample efficiency comparisons in papers, use `total_timesteps = config.run.total_timesteps` (computed property).

!!! note "Performance tuning"
    - Larger `num_envs` → Better GPU utilization
    - Larger `scan_chunk_size` → Fewer Python overheads, longer compile time
    - Rule of thumb: `scan_chunk_size` ≈ `num_envs` / 10

## Command-line overrides

Override any config value from the command line:

```bash
# Override single values
python scripts/train.py run.num_envs=10000

# Override nested values
python scripts/train.py agent.learning_rate=3e-4

# Override multiple values
python scripts/train.py \
  run.num_envs=10000 \
  run.steps_per_env=100 \
  agent.batch_size=256

# Switch entire config groups
python scripts/train.py \
  env=cartpole_sysid \
  agent=pqn
```

## Programmatic usage

```python
from omegaconf import OmegaConf
from myriad.configs.default import Config

# Load from YAML
cfg_dict = OmegaConf.load("configs/config.yaml")

# Override values
cfg_dict.run.num_envs = 10000
cfg_dict.agent.learning_rate = 3e-4

# Convert to Pydantic
config = Config(**OmegaConf.to_object(cfg_dict))

# Use in training
from myriad.platform.runner import train_and_evaluate
train_and_evaluate(config)
```

## Creating custom configs

### New environment config

Create `configs/env/my_env.yaml`:

```yaml
name: my-env

# Your physics parameters
physics:
  param1: value1
  param2: value2

# Your task parameters
task:
  max_steps: 1000
```

Use it:

```bash
python scripts/train.py env=my_env
```

### New run config

Create `configs/run/my_experiment.yaml`:

```yaml
seed: 123
steps_per_env: 20        # 20 steps/env * 50k envs = 1M total
num_envs: 50000
buffer_size: 100000
scan_chunk_size: 4096
eval_frequency: 5000
eval_rollouts: 20
eval_max_steps: 1000
log_frequency: 2000
```

Use it:

```bash
python scripts/train.py run=my_experiment
```

## Next steps

- [Running Experiments](../user-guide/running_experiments.md): Basic config usage
- [Setup Guide](setup.md): Development environment
- [JAX Architecture](architecture.md): Pure functional design principles
