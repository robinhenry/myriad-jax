# Running experiments

Myriad provides both a unified CLI and direct script access for different workflows.

## Command-line interface

The `myriad` CLI is the recommended way to run experiments after installation:

```bash
# Basic training
myriad train

# Override config values
myriad train \
  run.num_envs=10000 \
  run.steps_per_env=100 \
  agent.learning_rate=3e-4

# Evaluation only
myriad evaluate --config-name=experiments/eval_bangbang_cartpole

# Hyperparameter sweeps
myriad sweep

# Render saved episodes to video
myriad render episodes/ --fps 60
```

All Hydra arguments work with the CLI. Use `myriad --help` to see available commands.

## Development workflow

For rapid iteration during development, you can run scripts directly:

```bash
python scripts/train.py
```

This approach is useful when:
- Modifying the core platform code
- Debugging with IDE breakpoints
- Quickly testing configuration changes

## Override config values

Override any parameter from the command line using dot notation:

```bash
myriad train \
  run.num_envs=10000 \
  run.steps_per_env=100 \
  agent.learning_rate=3e-4
```

```{tip}
**Finding Default Values**

To find what the default value of a parameter is:
```

    - **Agent parameters:** Check factory function `src/myriad/agents/{agent_name}.py:make_agent()`
    - **Environment parameters:** Check factory function `src/myriad/envs/{env_name}/*.py:make_env()`
    - **Run parameters:** Check Pydantic model `src/myriad/configs/default.py:RunConfig`

    YAML files contain experiment-specific overrides, not the source of truth for defaults.

## Switch config groups

```bash
myriad train \
  env=cartpole_sysid \
  agent=pqn
```

Load different base configurations from `configs/agent/`, `configs/env/`, or `configs/run/`.

## Parameter sweeps

Use W&B sweeps for hyperparameter optimization:

```bash
# 1. Create a sweep configuration (sweep_config.yaml)
# 2. Initialize the sweep
wandb sweep sweep_config.yaml

# 3. Start sweep agents
wandb agent <sweep-id>
# This automatically calls 'myriad sweep' with W&B-provided parameters
```

The `sweep` command integrates with W&B to override Hydra parameters during hyperparameter search.

## Programmatic usage

```{literalinclude} ../../examples/original/07_quickstart_simple.py
:language: python
:caption: examples/original/07_quickstart_simple.py
:lines: 8-18
```

See the `examples/` directory in the repository for more programmatic usage patterns.

## Evaluation-only mode

Use `evaluate()` for non-learning controllers (PID, MPC, scripted) or benchmarking pre-trained agents.

**Random baseline:**

```{literalinclude} ../../examples/original/05_random_baseline.py
:language: python
:caption: examples/original/05_random_baseline.py
:lines: 8-20
```

**Pre-trained agent:**

See [examples/original/04_evaluate_pretrained.py](../../examples/original/04_evaluate_pretrained.py) for loading and evaluating saved agents.

## Episode collection

Collect full trajectories during evaluation for analysis or visualization:

```{literalinclude} ../../examples/original/08_episode_collection.py
:language: python
:caption: examples/original/08_episode_collection.py
:lines: 9-39
```

```{note}
Episode collection compiles a separate code path on first use. Subsequent calls reuse cached compilation.
```

## Periodic episode saving

Save episodes to disk during training for qualitative monitoring.

**Configuration and usage:**

```{literalinclude} ../../examples/original/09_periodic_episode_saving.py
:language: python
:caption: examples/original/09_periodic_episode_saving.py
:lines: 11-28
```

**Loading saved episodes:**

```{literalinclude} ../../examples/original/09_periodic_episode_saving.py
:language: python
:lines: 33-53
```

Episodes automatically logged to W&B as artifacts when enabled.

```{warning}
Set `eval_episode_save_frequency` >> `eval_frequency` to avoid storage bloat. Recommended: 5-10x eval frequency.
```

## Rendering episodes

Convert saved episode trajectories to MP4 videos:

```bash
# Render a directory of episodes
myriad render episodes/ --fps 60

# Render with custom output directory
myriad render episodes/step_1000000/ --output-dir videos/

# Render and upload to W&B
myriad render episodes/ --wandb-project my-project --wandb-run-id abc123
```

See `myriad render --help` for all options.

## CLI reference

| Command | Description | Example |
|---------|-------------|---------|
| `myriad train` | Train an agent | `myriad train run.total_timesteps=1e6` |
| `myriad evaluate` | Evaluation-only mode | `myriad evaluate --config-name=eval_config` |
| `myriad sweep` | W&B hyperparameter sweep | `wandb agent <sweep-id>` |
| `myriad render` | Render episodes to video | `myriad render episodes/ --fps 60` |

All commands support `--help` for detailed usage information.
