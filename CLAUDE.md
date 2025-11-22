# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aion is a platform for large-scale high-throughput control & decision-making experiments built with JAX. It provides a scalable reinforcement learning experimental platform that follows industrial standards and best practices, capable of handling large-scale experiments efficiently.

## Development Commands

### Testing
```bash
# Run all tests
python -m pytest

# Run tests in a specific directory
python -m pytest tests/core/
python -m pytest tests/platform/
python -m pytest tests/agents/
python -m pytest tests/envs/

# Run a specific test file
python -m pytest tests/platform/test_runner.py

# Run with coverage
python -m pytest --cov=aion
```

### Code Quality
```bash
# Format code with black
black src/ tests/

# Lint with ruff (auto-fix enabled)
ruff check --fix src/ tests/

# Type checking with mypy
mypy src/aion/

# Run pre-commit hooks manually
pre-commit run --all-files
```

### Configuration
The project uses Hydra for configuration management. Config files are in `configs/`:
- `configs/config.yaml` - Main configuration with defaults
- `configs/agent/` - Agent configurations
- `configs/env/` - Environment configurations

## Architecture

### Core Design Principles

**Pure Functional JAX**: The codebase follows a pure functional style with JAX transformations (jit, vmap, scan). All core functions are stateless and side-effect free, enabling efficient compilation and parallelization.

**Protocol-Based Components**: Environments and agents use small, permissive Protocols (`EnvironmentConfig`, `EnvironmentParams`, `EnvironmentState`, `AgentParams`, `AgentState`) allowing concrete implementations to use dataclasses, Flax structs, or NamedTuples while maintaining type safety.

**Typed Containers**: The `Environment` and `Agent` NamedTuples hold pure functions and configuration, separating behavior from state and enabling easy composition.

### Key Architectural Patterns

#### Training Loop Structure
The training system (`src/aion/platform/runner.py`) implements a high-performance loop using:
- **Vectorized environments**: All environments run in lockstep via `jax.vmap`
- **Chunked execution**: Training steps are batched into chunks and executed via `jax.lax.scan` for efficiency
- **Mask-aware scanning**: Active/inactive steps in chunks are handled via boolean masks to align logging/eval boundaries
- **Auto-reset**: Environments automatically reset when done without breaking the pure functional model

#### Component Factories
New environments and agents are registered via factory patterns:
- `ENV_REGISTRY` in `src/aion/envs/__init__.py`
- `AGENT_REGISTRY` in `src/aion/agents/__init__.py`

To add a new environment:
1. Create a module in `src/aion/envs/` implementing the `Environment` protocol
2. Define a `make_env(**kwargs)` factory function
3. Register it in `ENV_REGISTRY`
4. Add a corresponding config file in `configs/env/`

To add a new agent:
1. Create a module in `src/aion/agents/` implementing the `Agent` protocol
2. Define a `make_agent(action_space, **kwargs)` factory function
3. Register it in `AGENT_REGISTRY`
4. Add a corresponding config file in `configs/agent/`

#### Replay Buffer
`ReplayBuffer` (`src/aion/core/replay_buffer.py`) is a JAX-native, pure functional implementation:
- State is immutable (`ReplayBufferState` NamedTuple)
- All operations return new states
- Uses PyTree structure for flexible transition storage
- Supports batched add and sample operations

#### State Management
The platform uses nested state containers:
- `TrainingEnvState`: Bundles environment states and observations for all parallel environments
- Each component (agent, env, buffer) maintains its own state as a PyTree
- State updates flow through the training loop via pure function returns

### Module Organization

```
src/aion/
├── core/           # Core data structures and utilities
│   ├── types.py          # Base types (Transition, BaseModel)
│   ├── spaces.py         # Action/observation space definitions
│   └── replay_buffer.py  # JAX-native replay buffer
├── platform/       # Training infrastructure
│   ├── runner.py         # Main training loop with chunked execution
│   ├── scan_utils.py     # JAX scan utilities and masking functions
│   ├── logging_utils.py  # W&B integration and metrics handling
│   └── shared.py         # Shared state containers
├── agents/         # Agent implementations
│   ├── agent.py          # Agent protocol and base types
│   └── random.py         # Random agent implementation
├── envs/           # Environment implementations
│   ├── environment.py    # Environment protocol and base types
│   ├── example_env.py    # Example environment template
│   └── ccas_ccar.py      # CCAS-CCAR environment
├── configs/        # Configuration schemas
│   └── default.py        # Pydantic models for config validation
└── utils/          # Utilities
    └── plotting/         # Plotly-based visualization tools
```

### Configuration System

The project uses Hydra with Pydantic validation:
- Hydra YAML configs define runtime parameters (in `configs/`)
- Pydantic models validate and type configs at runtime (`src/aion/configs/default.py`)
- Config composition via Hydra's defaults list enables mixing agents/envs

Key config sections:
- `run`: Training hyperparameters (timesteps, batch size, buffer size, scan chunk size)
- `agent`: Agent selection and parameters
- `env`: Environment selection and parameters
- `wandb`: Weights & Biases logging configuration

### Logging and Monitoring

W&B integration (`src/aion/platform/logging_utils.py`):
- Training metrics logged at configurable frequency
- Evaluation rollouts logged separately
- Metrics are pulled from device to host only when needed
- Supports offline mode for later syncing

## Important Constraints

### JAX Best Practices
- All environment and agent functions must be pure (no side effects)
- Use `jax.tree_util.tree_map` for PyTree operations
- Avoid Python control flow in jitted functions; use `jax.lax` equivalents
- Keep static arguments separate (passed as `static_argnames` to jit)

### Performance Considerations
- The `scan_chunk_size` parameter controls compilation overhead vs step-to-step overhead
- Larger chunks reduce Python overhead but increase XLA compile time
- Mask-aware execution ensures alignment with logging/eval frequencies without dynamic control flow
- Auto-resetting environments avoid scan conditionals by using `jax.lax.select` and masking

### Code Style
- Line length: 120 characters (black and ruff configured)
- Import order: Ruff enforces isort with `aion` as first-party
- Type hints: Use for public APIs; mypy runs on `src/aion/**/*.py`
- Lambda expressions allowed (E731 ignored in ruff config)

## Testing Guidelines

- Tests are organized to mirror source structure (`tests/core/`, `tests/platform/`, etc.)
- Test files must follow `test_*.py` naming convention (enforced by pre-commit)
- Use `pytest` fixtures defined in `tests/conftest.py` for common setup
- Tests should verify JAX compilation works (e.g., test that functions jit correctly)
- Check both functional correctness and shape/dtype of PyTree outputs
