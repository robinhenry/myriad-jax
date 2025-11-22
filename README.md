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

Run a training experiment:

```bash
python -m aion.main
```

Override configuration:

```bash
python -m aion.main agent=random env=toy_env run.total_timesteps=1000000
```

Configurations are in `configs/` and use Hydra's composition system.

## Adding Components

**New Environment:**
1. Implement the `Environment` protocol in `src/aion/envs/`
2. Register in `ENV_REGISTRY` with a `make_env(**kwargs)` factory
3. Add config file in `configs/env/`

**New Agent:**
1. Implement the `Agent` protocol in `src/aion/agents/`
2. Register in `AGENT_REGISTRY` with a `make_agent(action_space, **kwargs)` factory
3. Add config file in `configs/agent/`

## Development

```bash
# Run tests
python -m pytest

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

Contributions welcome! Ensure tests pass and code follows style guidelines before submitting PRs.
