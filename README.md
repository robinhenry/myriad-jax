# Myriad

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/myriad-jax)](https://pypi.org/project/myriad-jax/)
[![CI](https://github.com/robinhenry/myriad-jax/actions/workflows/ci.yml/badge.svg)](https://github.com/robinhenry/myriad-jax/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/robinhenry/myriad-jax/branch/main/graph/badge.svg)](https://codecov.io/gh/robinhenry/myriad-jax)
[![Documentation](https://github.com/robinhenry/myriad-jax/actions/workflows/docs.yml/badge.svg)](https://github.com/robinhenry/myriad-jax/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![JAX](https://img.shields.io/badge/JAX-0.7.2-orange.svg)](https://github.com/google/jax)

**JAX-native platform for massively parallel system identification and control of uncertain, stochastic systems.**

Myriad is named after the Greek 'myrias' (ten thousand), inspired by microfluidic mother machines that observe 100,000+ cells simultaneously. It provides a myriad of viewpoints from which to learn about and control complex systems.

## The Challenge

Many research domains require learning from populations of systems with uncertain parameters:

- **System identification**: Estimate unknown parameters by observing many variants in parallel
- **Stochastic systems**: Randomness is fundamental (molecular noise, asynchronous events)
- **Active learning**: Design experiments to maximize information gain across parameter variants
- **Robust control**: Learn policies that work across diverse conditions

Myriad is designed for this paradigm: 100,000+ environments with different parameters or initial conditions on a single GPU, enabling population-level system identification and control.

## What Myriad Provides

Inspired by state-of-the-art libraries like Gymnasium and Brax, Myriad adds:

1. **Population-scale parallelism**: 100,000+ environments on a single GPU—not for speed, but for a different research paradigm
2. **System ID as a first-class task**: Learn unknown parameters by observing the population, not just single trajectories
3. **Native stochastic simulation**: Gillespie algorithm, asynchronous events, multi-timescale dynamics
4. **Active learning**: Design experiments to maximize information gain across parameter variants
5. **Three-layer architecture**: Reuse physics across control, system ID, and planning tasks

## From Mother Machines to Parameter Sweeps

Myriad's population-scale parallelism enables new experimental paradigms:

| Domain | Example | Population-Level Insight |
|--------|---------|-------------------------|
| **Synthetic Biology** | Mother machine tracking 100k cells | Identify gene circuit parameters from cell-to-cell variability |
| **Chemical Engineering** | Parallel bioreactor conditions | Optimize kinetics across temperature/pH/substrate gradients |
| **Control Theory** | CartPole with randomized physics | Learn robust policies from 100k parameter combinations |
| **Systems Biology** | Metabolic network variants | Infer reaction rates from population heterogeneity |
| **RL Research** | Stochastic environment suite | Benchmark algorithms across diverse initial conditions |

**Example:** The built-in gene circuit environment simulates stochastic gene expression with asynchronous cell division—just like a microfluidic mother machine. Observe 100,000 cells with different initial conditions or parameters simultaneously, then use the population data for system identification or robust control design.

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
git clone https://github.com/robinhenry/myriad-jax.git
cd myriad-jax
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

## Example Workflows

**System Identification:**
```python
# Learn unknown parameters from 10,000 parallel experiments
config = create_config(
    env="cartpole-sysid",  # or your custom environment
    agent="dqn",
    num_envs=10000
)
results = train_and_evaluate(config)
```

**Parameter Sweep:**
```python
# Test 100,000 parameter combinations in parallel
# Useful for: robustness analysis, sensitivity studies, design optimization
config = create_config(
    env="your_env",
    num_envs=100000,
    randomize_params=True
)
```

**Active Learning:**
```python
# Design experiments to reduce parameter uncertainty
# Built-in support for information-theoretic experiment design
```

### Complementary to Existing Tools

Myriad complements rather than replaces existing libraries:

- **Use Gymnasium** for: Standard RL benchmarks, established baselines, broad ecosystem support
- **Use Brax** for: Rigid-body robotics, differentiable physics engines, fast locomotion
- **Use Myriad** when you need: System identification, stochastic dynamics, or parallel parameter exploration

These libraries work together—prototype your algorithm on Gymnasium, then scale system ID experiments with Myriad.

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
- [Roadmap](ROADMAP.md): Strategic development plan and upcoming features

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
