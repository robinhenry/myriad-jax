# Aion

A scalable platform for large-scale high-throughput control and decision-making experiments built with JAX.

## Features

- **Pure Functional JAX**: Efficient compilation and parallelization via `jit`, `vmap`, and `scan`
- **High-Throughput Training**: Vectorized environments with chunked execution for maximum performance
- **Protocol-Based Architecture**: Flexible components using Python protocols
- **JAX-Native Replay Buffer**: Immutable, pure functional replay buffer with PyTree support
- **Hydra Configuration**: Composable YAML configs with Pydantic validation
- **Experiment Tracking**: Built-in Weights & Biases integration

## Installation

Requires Python ^3.11 and JAX.

```bash
git clone <repository-url>
cd aion
poetry install
```

## Quick Start

Run a training experiment (with option to override configuration parameters):

```bash
# One of these:
python scripts/train.py
python scripts/train.py --config-file=pqn_cartpole
python scripts/train.py run.total_timesteps=1000 agent.gamma=0.98
```

Configurations are in `configs/` and use Hydra's composition system.

## Adding Components

**New Environment:**
1. Implement the `Environment` protocol in `src/aion/envs/`
2. Register in `ENV_REGISTRY` with a `make_env(...)` factory
3. Add config file in `configs/env/`

**New Agent:**
1. Implement the `Agent` protocol in `src/aion/agents/`
2. Register in `AGENT_REGISTRY` with a `make_agent(...)` factory
3. Add config file in `configs/agent/`

**New (Environment, Agent) run configuration**
1. Add run config file in `configs/run/`
2. Add overall config file in `configs/`

## Development

```bash
# Run tests
pytest

# Format and lint
black src/ tests/
ruff check --fix src/ tests/

# Type checking
mypy src/aion/
```

See `CLAUDE.md` for detailed architecture documentation and development guidelines.

## Author

Robin Henry (robin.henry012@gmail.com)

## Contributing

Contributions welcome!
