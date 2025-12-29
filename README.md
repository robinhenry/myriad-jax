# Myriad

**A JAX-native platform for massively parallel system identification and control.**

Myriad is named after the Greek 'myrias', representing the ten thousand parallel environments the engine simulates simultaneously. It provides a myriad of viewpoints from which to observe, identify, and control complex systems.

## The Problem

Standard RL environments (Gym, Gymnax) give you one robot and ask you to control it.

Myriad gives you **100,000 uncertain physical systems** in parallel and asks you to:

1. **Identify** their hidden parameters (System ID)
2. **Control** them to a target (RL/MPC)
3. **Plan** experiments to reduce uncertainty (Active Learning)

## Key Features

- **Massively Parallel**: Run 100,000+ environments simultaneously on a single GPU. JAX's `vmap` and XLA compilation deliver millions of steps per second.
- **Three-Layer Architecture**: Separate physics from task logic from learning algorithms—reuse the same physics for control, system ID, and model-based planning.
- **Direct Differentiable Access**: Unlike Gym-style environments, Myriad exposes pure physics functions for MPC, gradient-based optimization, and hybrid models.
- **Built for Science**: Domain randomization, parameter sweeps, and active experimental design are first-class features.
- **Pure JAX**: Fully jitted training loops with immutable PyTree state
- **Hydra + Pydantic**: Composable configs with runtime validation
- **W&B Integration**: Built-in experiment tracking

## Installation

**Requirements:** Python 3.11+, JAX 0.7.2+

```bash
git clone https://github.com/robinhenry/myriad.git
cd myriad
poetry install
```

**GPU support (CUDA 12):**
```bash
poetry install --with gpu
```

See [Installation Guide](docs/getting-started/installation.md) for details.

## Quick Start

### Python API (Recommended)

Use Myriad programmatically in your scripts:

```python
from myriad import create_config, train_and_evaluate

# Create config and train
config = create_config(
    env="cartpole-control",
    agent="dqn",
    num_envs=1000,
    steps_per_env=100
)
results = train_and_evaluate(config)

# View results
print(results.summary())
results.save_agent("trained_agent.pkl")
```

See [`examples/`](examples/) for more programmatic usage patterns.

### CLI (Alternative)

Train DQN on CartPole using the command line:

```bash
myriad train
```

Scale up to 10,000 parallel environments:

```bash
myriad train run.num_envs=10000
```

Try system identification instead of control:

```bash
myriad train env=cartpole_sysid agent=pqn
```

Override any configuration parameter:

```bash
myriad train \
  env=cartpole_control \
  agent=dqn \
  run.num_envs=50000 \
  run.steps_per_env=15625
```

See [Quickstart Guide](docs/getting-started/quickstart.md) for more examples.

## What's Implemented

**Environments:**
- `cartpole-control`: Standard stabilization task
- `cartpole-sysid`: Active parameter learning variant
- `ccas_ccar_v1`: Gene circuit with growth/division dynamics (the "stress test")

**Agents:**
- `dqn`: Deep Q-Network (discrete actions)
- `pqn`: Parametric Q-Network (continuous actions)
- `random`: Baseline

## Use Cases

| Domain | Example | What Myriad Provides |
|--------|---------|---------------------|
| **RL Research** | Train PPO on CartPole | 100k environments with randomized masses/lengths in 4 seconds |
| **Control Theory** | MPC over stiff ODEs | Direct access to differentiable physics for gradient-based planning |
| **Scientific ML** | Parameter estimation | 100k short trajectories for system identification |
| **Synthetic Biology** | Gene circuit control | In-silico optimization before lab work |

## Documentation

**New to Myriad?**
- [Installation](docs/getting-started/installation.md): Set up in 5 minutes
- [Quickstart](docs/getting-started/quickstart.md): Your first training run

**Building Experiments?**
- [Core Concepts](docs/user-guide/concepts.md): Understand the three-layer architecture
- [Custom Environment](docs/user-guide/custom_env.md): Implement your own physics
- [Custom Agent](docs/user-guide/custom_agent.md): Add new learning algorithms
- [Running Experiments](docs/user-guide/running_experiments.md): Train agents at scale

**Contributing?**
- [Development Setup](docs/contributing/setup.md): Configure your environment
- [Architecture Guide](docs/contributing/architecture.md): Pure functional design constraints
- [Configuration System](docs/contributing/configuration.md): Hydra and Pydantic patterns

See `CLAUDE.md` for AI assistant development guidelines.

## Development

```bash
# Run tests
python -m pytest

# Format and lint
black src/ tests/
ruff check --fix src/ tests/

# Type checking
mypy src/myriad/
```

## Author

Robin Henry (robin.henry012@gmail.com)

## Contributing

Contributions welcome!
