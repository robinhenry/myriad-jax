# Myriad Examples

This directory contains runnable examples demonstrating how to use Myriad programmatically.

**Important:** All Python code examples in the documentation reference these scripts using Sphinx's `literalinclude` directive. This ensures documentation examples are always up-to-date and working.

## Quick Start Examples

### 07_quickstart_simple.py
The absolute simplest example - shown in the main docs index.
- Minimal configuration
- Shows the core API: `create_config()` + `train_and_evaluate()`
- Perfect starting point for new users

**Run:**
```bash
python examples/07_quickstart_simple.py
```

### 02_basic_training.py
The simplest way to train an agent using the Python API.
- Uses `create_config()` for easy configuration
- Trains DQN on CartPole
- Shows how to access results and summary statistics

**Run:**
```bash
python examples/02_basic_training.py
```

### 03_advanced_training.py
Advanced training with custom hyperparameters.
- Shows how to customize agent parameters (learning rate, batch size, etc.)
- Demonstrates using dot notation for nested config overrides
- Example of a more realistic training setup

**Run:**
```bash
python examples/03_advanced_training.py
```

## Evaluation Examples

### 05_random_baseline.py
Evaluate a random agent without any training.
- Uses `create_eval_config()` for evaluation-only configs
- Good baseline for comparing learning algorithms
- Fast and simple

**Run:**
```bash
python examples/05_random_baseline.py
```

### 04_evaluate_pretrained.py
Example of loading and evaluating a saved agent.
- **Note:** Due to JAX/Flax serialization limitations, saving agent state may not work
- This example demonstrates the intended API but may skip execution
- In practice, use `results.agent_state` directly in-memory

**Run:**
```bash
python examples/04_evaluate_pretrained.py
```

## Episode Management Examples

### 08_episode_collection.py
Collect full episode trajectories for analysis.
- Shows how to get observations, actions, rewards, dones
- Demonstrates episode padding and trimming
- Used in "Running Experiments" documentation

**Run:**
```bash
python examples/08_episode_collection.py
```

### 09_periodic_episode_saving.py
Save episodes to disk during training.
- Configure automatic episode saving
- Load and inspect saved episodes
- Useful for qualitative monitoring
- Used in "Running Experiments" documentation

**Run:**
```bash
python examples/09_periodic_episode_saving.py
```

## Legacy Examples

### 01_classical_control.py
Legacy example using YAML configuration files.
- Paired with `01_classical_control.yaml`
- Shows backward compatibility with CLI-based workflows

**Run:**
```bash
python examples/01_classical_control.py
```

### 06_yaml_config.py
Demonstrates loading configs from YAML files.
- Requires `01_classical_control.yaml`
- Useful when migrating from CLI to programmatic API

**Run:**
```bash
python examples/06_yaml_config.py
```

## Common Issues

### Agent Serialization
Due to JAX/Flax limitations with pickling closures, `results.save_agent()` may fail.
The examples handle this gracefully. For production use:
- Keep `agent_state` in memory for the duration of your script
- Use Flax's serialization utilities directly if needed
- Extract and save only the parameter arrays

### Performance Warnings
You may see warnings about `scan_chunk_size`. These are informational and can be
ignored for small experiments. For production runs, adjust `scan_chunk_size`,
`log_frequency`, or `eval_frequency` as suggested in the warning.

## Next Steps

After trying these examples, see:
- [API Documentation](../docs/api/platform.md): Full API reference
- [Quickstart Guide](../docs/getting-started/quickstart.md): CLI usage
- [Custom Environments](../docs/user-guide/custom_env.md): Build your own physics
