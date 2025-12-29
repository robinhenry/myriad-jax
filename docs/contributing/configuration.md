# Configuration System

```{note}
**Advanced Topic**

This document explains Myriad's Hydra-based configuration system in detail. Most users only need to know how to override config values (see [Running Experiments](../user-guide/running_experiments.md)). This guide is for contributors or advanced users who need to understand the configuration architecture.
```

Myriad uses [Hydra](https://hydra.cc) for composable configuration management. Configs are organized into three categories: `env/`, `agent/`, and `run/`.

## Configuration Philosophy

### Single Source of Truth

**All default values are defined in Python factory functions, not in YAML files.**

This design follows industry best practices:

- **Python factory functions** (`make_agent()`, `make_env()`): Define all parameter defaults
- **YAML configuration files**: Specify only experiment-specific overrides
- **Pydantic models**: Validate types and structure, not defaults

### Why This Design?

1. **No Duplication** - Defaults live in one place (the code that uses them)
2. **Type Safety** - Python type hints catch errors at development time
3. **IDE Support** - Jump-to-definition, autocomplete, and inline docs work
4. **Programmatic Access** - Direct factory calls use the same defaults as YAML
5. **Clear Intent** - YAML files show what makes an experiment unique

### Where Defaults Live

Different config types use different patterns based on their purpose:

- **Agent/Env configs**: Defaults in factory functions (`make_agent()`, `make_env()`)
  - These construct complex objects with initialization logic
  - Defaults live where they're used
- **Run/Wandb configs**: Defaults in Pydantic models (`RunConfig`, `WandbConfig`)
  - These are pure config schemas for validation
  - Defaults live in the model definition
- **Shared eval parameters**: Defaults in `EvalConfigBase`
  - Inherited by both `RunConfig` (training) and `EvalConfig` (eval-only)
  - Eliminates duplication via inheritance

### Finding Default Values

To find the default value of a parameter:

1. Identify the component type (agent, env, run, wandb)
2. Look up defaults in the appropriate location:
   - **Agents:** `src/myriad/agents/{agent_name}.py` → `make_agent()` signature
   - **Environments:** `src/myriad/envs/{env_name}/*.py` → `make_env()` signature
   - **Run settings:** `src/myriad/configs/default.py` → `RunConfig` class
   - **W&B settings:** `src/myriad/configs/default.py` → `WandbConfig` class

**Example:** DQN learning rate default (factory function)

```python
# src/myriad/agents/rl/dqn.py
def make_agent(
    action_space: Space,
    learning_rate: float = 1e-3,  # ← Default is 1e-3
    gamma: float = 0.99,           # ← Default is 0.99
    ...
) -> Agent:
```

**Example:** Config defaults (Pydantic model with inheritance)

```python
# src/myriad/configs/default.py
class EvalConfigBase(BaseModel):
    """Shared evaluation parameters."""
    seed: int = 42                      # ← Default is 42
    eval_rollouts: int = 10             # ← Default is 10
    eval_max_steps: int                 # ← REQUIRED (no default)

class RunConfig(EvalConfigBase):
    """Training config (extends eval params)."""
    num_envs: PositiveInt = 1           # ← Default is 1
    eval_frequency: PositiveInt = 1000  # ← Default is 1000
    steps_per_env: PositiveInt          # ← REQUIRED (no default)
    # Inherits: seed, eval_rollouts, eval_max_steps

class EvalRunConfig(EvalConfigBase):
    """Eval-only run config (extends eval params)."""
    pass  # Just inherits eval params

class EvalConfig(BaseModel):
    """Eval-only config (matches Config structure)."""
    run: EvalRunConfig  # ← Nested like Config.run
    agent: AgentConfig
    env: EnvConfig
    wandb: WandbConfig | None = None
```

YAML files only specify values that differ from these defaults.

## Configuration hierarchy

```
configs/
├── config.yaml              # Main config (references default experiment)
├── env/                     # Environment base configs (Hydra composition)
│   ├── cartpole_control.yaml
│   ├── cartpole_sysid.yaml
│   └── ccas_ccar_control.yaml
├── agent/                   # Agent base configs (Hydra composition)
│   ├── dqn.yaml
│   ├── pqn.yaml
│   └── bangbang.yaml
└── experiments/             # Self-contained experiment configs
    ├── dqn_cartpole_default.yaml
    ├── pqn_cartpole_default.yaml
    ├── dqn_fast_exploration.yaml
    ├── dqn_slow_exploration.yaml
    ├── cartpole_heavy_pole.yaml
    └── ccas_sinewave_tracking.yaml
```

## Main config (`config.yaml`)

The main config simply references a default experiment:

```yaml
# @package _global_
# Main Configuration

# Load default experiment
defaults:
  - experiments/dqn_cartpole_default
```

This makes `python scripts/train.py` run the DQN CartPole baseline by default.

### Experiment configs

Experiment configs are self-contained and include all settings. They only specify parameters that differ from defaults:

```yaml
# @package _global_
defaults:
  - /agent: dqn
  - /env: cartpole_control
  - _self_

# Only specify overrides and required params
run:
  steps_per_env: 150000  # Required (no default)
  buffer_size: 50000     # Off-policy agents need this
  scan_chunk_size: 2048  # Override default (256)
  eval_max_steps: 500    # Required (environment-specific)
  # seed, num_envs, eval_frequency, etc. use defaults

wandb:
  entity: lugagne-lab    # Override default (None)
  group: baseline        # Override default (None)
  tags: ["dqn", "cartpole"]  # Override default ([])
  # enabled, project, mode, etc. use defaults
```

### W&B integration

**Defaults:** `src/myriad/configs/default.py:WandbConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable W&B logging |
| `project` | `str` | `"myriad"` | Project name (groups runs) |
| `job_type` | `str` | `"train"` | Run type (train, eval, etc.) |
| `mode` | `str` | `"offline"` | `online` or `offline` |
| `entity` | `str\|null` | `None` | Team or username (user-specific) |
| `group` | `str\|null` | `None` | Sub-group within project (experiment-specific) |
| `run_name` | `str\|null` | `None` | Custom name or auto-generate |
| `dir` | `str\|null` | `None` | Local logging directory |
| `tags` | `tuple[str]` | `()` | Searchable tags (experiment-specific) |

## Environment config (`env/`)

Environment base configs provide Hydra composition targets. They contain minimal metadata. YAML files in experiments/ contain experiment-specific overrides; defaults are in factory functions.

### CartPole Control

**Default values:** `src/myriad/envs/cartpole/tasks/control.py:make_env()`

**Base config** (`configs/env/cartpole_control.yaml`):
```yaml
# @package env
name: cartpole-control
```

**Experiment override example** (in an experiment config):
```yaml
env:
  physics:
    pole_mass: 0.15      # Heavier pole (default: 0.1)
    force_magnitude: 15.0  # Stronger actuator (default: 10.0)
  task:
    max_steps: 1000      # Longer episodes (default: 500)
```

#### Available Physics Parameters

All parameters have defaults in the factory function. Specify in YAML only to override.

| Parameter | Type | Units | Default | Description |
|-----------|------|-------|---------|-------------|
| `gravity` | `float` | m/s² | `9.8` | Gravitational acceleration |
| `cart_mass` | `float` | kg | `1.0` | Cart mass |
| `pole_mass` | `float` | kg | `0.1` | Pole mass |
| `pole_length` | `float` | m | `0.5` | Half-length of pole |
| `force_magnitude` | `float` | N | `10.0` | Force applied per action |
| `dt` | `float` | s | `0.02` | Integration timestep |

#### Available Task Parameters

| Parameter | Type | Units | Default | Description |
|-----------|------|-------|---------|-------------|
| `max_steps` | `int` | - | `500` | Maximum episode length |
| `theta_threshold` | `float` | rad | `0.209` | Angle termination threshold (≈12°) |
| `x_threshold` | `float` | m | `2.4` | Position termination threshold |

```{tip}
**Finding Defaults**

Check the factory function and dataclass definitions for authoritative default values. The code is always correct.
```

### CartPole SysID

**Default values:** `src/myriad/envs/cartpole/tasks/sysid.py:make_env()`

Same structure as control task, but observation includes belief state over physics parameters.

### Gene Circuit (CCAS-CCAR)

**Default values:** `src/myriad/envs/ccas_ccar/tasks/control.py:make_env()`

Contains ODE parameters for protein expression and cell growth dynamics.

## Agent config (`agent/`)

Agent base configs provide Hydra composition targets. They contain minimal metadata. YAML files in experiments/ contain experiment-specific overrides; defaults are in factory functions.

### DQN Agent

**Default values:** `src/myriad/agents/rl/dqn.py:make_agent()` (`src/myriad/agents/rl/dqn.py:254`)

**Base config** (`configs/agent/dqn.yaml`):
```yaml
# @package agent
name: dqn
```

**Experiment override example** (in an experiment config):
```yaml
agent:
  epsilon_decay_steps: 50000  # Override default (10000)
  target_network_frequency: 1000  # Override default (500)
```

#### Available Parameters

All parameters have defaults in the factory function. Specify in YAML only to override.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | See factory | Replay buffer sample size |
| `learning_rate` | `float` | `1e-3` | Adam learning rate |
| `gamma` | `float` | `0.99` | Discount factor |
| `epsilon_start` | `float` | `1.0` | Initial exploration rate |
| `epsilon_end` | `float` | `0.05` | Final exploration rate |
| `epsilon_decay_steps` | `int` | `10000` | Steps to decay epsilon |
| `target_network_frequency` | `int` | `500` | Target network update interval |
| `tau` | `float` | `1.0` | Target update rate (1.0 = hard) |

```{tip}
**Finding Defaults**

Check the factory function signature for the authoritative source of default values. Defaults shown in this table may be outdated; the code is always correct.
```

### PQN (`pqn.yaml`)

Parametric Q-Network for continuous actions. Similar structure to DQN.

### Random (`random.yaml`)

No configurable parameters. Used as baseline.

## Run config

Run configurations control training loop execution. Run parameters are specified directly in experiment configs under the `run:` key.

**Defaults:** `src/myriad/configs/default.py:RunConfig` (extends `EvalConfigBase`)

`RunConfig` inherits evaluation parameters from `EvalConfigBase` and adds training-specific settings. This ensures eval parameters are consistent between training and eval-only runs.

### Example Run Config (in an experiment config)

```yaml
run:
  # Required parameters (no defaults)
  steps_per_env: 150000  # Experiment-specific
  eval_max_steps: 500    # Environment-specific

  # Override defaults as needed
  num_envs: 32           # Default: 1
  buffer_size: 50000     # Default: None (required for off-policy)
  scan_chunk_size: 2048  # Default: 256
```

### Default vs Required Parameters

**Parameters with defaults** (override only if needed):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `42` | Random seed for reproducibility |
| `num_envs` | `int` | `1` | Parallel environments (vectorized execution) |
| `scan_chunk_size` | `int` | `256` | JAX scan chunk size (affects compilation/memory) |
| `eval_frequency` | `int` | `1000` | Evaluation interval (steps per env) |
| `eval_rollouts` | `int` | `10` | Episodes to average for eval metrics |
| `log_frequency` | `int` | `1000` | Logging interval (steps per env) |
| `buffer_size` | `int\|null` | `None` | Replay buffer capacity (off-policy only) |
| `rollout_steps` | `int\|null` | `None` | Rollout length (on-policy only) |
| `eval_episode_save_frequency` | `int` | `0` | Episode save interval (0 = disabled) |
| `eval_episode_save_count` | `int\|null` | `None` | Episodes to save per checkpoint (null = all) |

**Required parameters** (must be specified):

| Parameter | Type | Description |
|-----------|------|-------------|
| `steps_per_env` | `int` | Steps each environment will take (total = steps_per_env × num_envs) |
| `eval_max_steps` | `int` | Max steps per eval episode (environment-specific) |

```{note}
**Terminology Note: steps_per_env vs total_timesteps**

Myriad uses `steps_per_env` as the primary configuration parameter, which differs from standard RL convention:

- **`steps_per_env`** (primary): Number of steps each environment will take during training
- **`total_timesteps`** (computed): Total environment interactions = `steps_per_env × num_envs`

This design aligns with the "digital twin" / parallel experiments mental model: you specify how long to run each experiment, and the total computational budget scales naturally with parallelism.

For RL sample efficiency comparisons in papers, use `total_timesteps = config.run.total_timesteps` (computed property).
```

```{note}
**Performance tuning**

- Larger `num_envs` → Better GPU utilization
- Larger `scan_chunk_size` → Fewer Python overheads, longer compile time
- Rule of thumb: `scan_chunk_size` ≈ `num_envs` / 10
```

## Command-line overrides

Override any config value from the command line:

```bash
# Run a specific experiment
python scripts/train.py --config-name=experiments/dqn_fast_exploration

# Override single values
python scripts/train.py run.num_envs=10000

# Override nested values
python scripts/train.py agent.learning_rate=3e-4

# Override multiple values
python scripts/train.py \
  --config-name=experiments/dqn_cartpole_default \
  run.num_envs=10000 \
  run.steps_per_env=100 \
  agent.batch_size=256

# Mix and match with experiment configs
python scripts/train.py \
  --config-name=experiments/dqn_fast_exploration \
  run.num_envs=10000
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

## Creating Custom Configs

### New Experiment Config

Experiment configs are self-contained. Copy an existing experiment as a template.

**Example:** Create `configs/experiments/my_experiment.yaml`:

```yaml
# @package _global_
# My custom experiment

defaults:
  - /agent: dqn
  - /env: cartpole_control
  - _self_

# Run parameters (required)
run:
  seed: 123
  steps_per_env: 20000
  num_envs: 50000
  buffer_size: 100000
  scan_chunk_size: 4096
  eval_frequency: 5000
  eval_rollouts: 20
  eval_max_steps: 1000
  log_frequency: 2000

# Override agent parameters
agent:
  epsilon_decay_steps: 5000
  learning_rate: 3e-4

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

Use it:

```bash
python scripts/train.py --config-name=experiments/my_experiment
```

### New Environment Base Config

When adding a new environment, create a minimal base config for Hydra composition.

Create `configs/env/my_env.yaml`:

```yaml
# @package env
name: my-env
```

**Do NOT repeat all default values in YAML.** Defaults live in the factory function. Override parameters in experiment configs only.

## Adding New Parameters

When adding parameters to agents or environments, follow this workflow:

### 1. Add to Factory Function (Single Source of Truth)

```python
# src/myriad/agents/rl/dqn.py
def make_agent(
    action_space: Space,
    learning_rate: float = 1e-3,
    new_parameter: float = 0.5,  # ← Add with default value
    ...
) -> Agent:
    """
    Args:
        new_parameter: Description of what this parameter does
    """
    params = AgentParams(
        action_space=action_space,
        learning_rate=learning_rate,
        new_parameter=new_parameter,  # ← Pass to params
        ...
    )
```

### 2. Update Pydantic Model (Type Validation)

Pydantic models use `extra: "allow"` so new parameters work automatically. No changes needed unless you want explicit type validation.

**Optional:** Add explicit field for better validation:

```python
# src/myriad/configs/default.py
class AgentConfig(BaseModel):
    model_config = {"extra": "allow"}

    name: str
    batch_size: PositiveInt | None = None
    new_parameter: float | None = None  # ← Optional explicit validation
```

### 3. Document in Config File (Optional)

Add a comment to the YAML showing the parameter exists:

```yaml
# configs/agent/dqn.yaml
# @package agent
# Available parameters (see src/myriad/agents/rl/dqn.py for defaults):
#   learning_rate (float): Adam learning rate [default: 1e-3]
#   new_parameter (float): Description [default: 0.5]  # ← Document availability

name: dqn

# Uncomment to override:
# new_parameter: 0.8
```

### 4. Use in Experiments

Override the default in experiment configs:

```bash
# CLI override
python scripts/train.py agent.new_parameter=0.8

# Or create experiment config
# configs/agent/dqn_high_param.yaml
name: dqn
new_parameter: 0.8
```

```{warning}
**Do Not Duplicate Defaults**

Never put default values in YAML files. Defaults live ONLY in factory functions. YAML files specify experiment-specific overrides only.
```

## Evaluation-Only Configs

For testing controllers without training (classical controllers, pre-trained models, baselines), use `EvalConfig` with `scripts/evaluate.py`.

### EvalConfig Schema

**Defaults:** `src/myriad/configs/default.py:EvalConfig`

```python
# Base class with shared eval parameters
class EvalConfigBase(BaseModel):
    seed: int = 42
    eval_rollouts: int = 10
    eval_max_steps: int  # Required (environment-specific)
    eval_episode_save_frequency: int = 0
    eval_episode_save_count: int | None = None
    # Episodes saved to episodes/ (hard-coded, relative to Hydra run dir)

# Eval-only run config (just inherits base)
class EvalRunConfig(EvalConfigBase):
    pass

# Eval-only config (matches Config structure)
class EvalConfig(BaseModel):
    run: EvalRunConfig   # ← Nested like Config.run
    agent: AgentConfig
    env: EnvConfig
    wandb: WandbConfig | None = None
```

**Key benefits:**
- Eval parameters shared with `RunConfig` via inheritance (no duplication)
- Structure matches `Config` (both have `run`, `agent`, `env`, `wandb`)

### Example Evaluation Config

```yaml
# configs/experiments/eval_bangbang_cartpole.yaml
# @package _global_

defaults:
  - /agent: bangbang
  - /env: cartpole_control
  - _self_

# Run settings (nested like training configs)
run:
  eval_max_steps: 500  # Required
  eval_rollouts: 100   # Override default (10)

# Optional: enable W&B logging
# wandb:
#   entity: lugagne-lab
#   group: classical-controllers
#   tags: ["bangbang", "cartpole"]
```

**Note:** Eval configs now use `run:` nesting to match training config structure.

### Usage

```bash
# Run evaluation
python scripts/evaluate.py --config-name=experiments/eval_bangbang_cartpole

# Override parameters
python scripts/evaluate.py \
  --config-name=experiments/eval_bangbang_cartpole \
  eval_rollouts=200
```

## Next steps

- [Running Experiments](../user-guide/running_experiments.md): Basic config usage
- [Setup Guide](setup.md): Development environment
- [JAX Architecture](architecture.md): Pure functional design principles
