# Development Setup

This guide is for contributors who want to modify Myriad's internals. If you just want to use Myriad to build experiments, see the [Installation guide](../get-started/installation.md) instead.

## Requirements

- Python 3.11+
- JAX 0.7.2+
- Poetry
- Git

## Clone and install

```bash
git clone https://github.com/robinhenry/myriad.git
cd myriad
```

Install all dependencies including dev tools:

```bash
poetry install --with dev,gpu
```

This installs:

- Core dependencies (JAX, Flax, Hydra, etc.)
- Development tools (pytest, ruff, mypy, black)
- GPU support (optional)

## Development workflow

### Running tests

Run the full test suite:

```bash
python -m pytest
```

Run specific test files:

```bash
python -m pytest tests/envs/cartpole/test_physics.py
```

Run with coverage:

```bash
python -m pytest --cov=myriad --cov-report=html
```

### Code formatting

Format code with Black:

```bash
black src/ tests/
```

Lint with Ruff:

```bash
ruff check --fix src/ tests/
```

### Type checking

```bash
mypy src/myriad/
```

### Pre-commit hooks

Install pre-commit hooks to automatically format and lint:

```bash
pre-commit install
```

Now `black`, `ruff`, and `mypy` will run automatically on `git commit`.

## Project structure

```
myriad/
├── src/myriad/
│   ├── core/          # Protocols and base types (Three Layers)
│   ├── envs/          # JAX environments (Physics implementations)
│   ├── agents/        # RL/Control algorithms (PPO, SAC, etc.)
│   ├── platform/      # Infrastructure (logging, W&B)
│   └── utils/         # Utilities (plotting, math)
├── configs/           # Hydra YAML definitions
├── tests/             # Mirrors src/ structure
└── scripts/           # Entry points (train.py, etc.)
```

## Testing philosophy

Tests are organized to mirror the three-layer architecture:

1. **Physics tests** (`tests/envs/*/test_physics.py`): Validate dynamics against analytical solutions
2. **Task tests** (`tests/envs/*/test_task.py`): Verify reward/termination logic
3. **Agent tests** (`tests/agents/test_*.py`): Check learning algorithm correctness

See `tests/conftest.py` for global JAX fixtures.

## JAX development constraints

When modifying environments or agents, you must follow JAX rules:

- **Purity**: All env/agent functions MUST be pure (no side effects)
- **Data**: Use `jax.tree_util.tree_map` for PyTree operations
- **Control Flow**: NO Python `if/else` in jitted paths (use `jax.lax.cond`/`select`)
- **Masking**: Use mask-aware execution for auto-reset
- **JIT**: Keep static args separate (`static_argnames`)

See [JAX Architecture](architecture.md) for details.

## Making changes

### Adding a new environment

1. Create `src/myriad/envs/your_env/physics.py` (pure dynamics)
2. Create `src/myriad/envs/your_env/tasks/control.py` (task wrapper)
3. Register in `src/myriad/envs/__init__.py`
4. Add Hydra config in `configs/env/your_env.yaml`
5. Write tests in `tests/envs/your_env/`

Reference implementation: `src/myriad/envs/cartpole/`

### Adding a new agent

1. Create `src/myriad/agents/your_agent.py`
2. Implement the `Agent` protocol (`select_action`, `update`)
3. Register in `src/myriad/agents/__init__.py`
4. Add Hydra config in `configs/agent/your_agent.yaml`
5. Write tests in `tests/agents/test_your_agent.py`

Reference implementation: `src/myriad/agents/dqn.py`

## Documentation

Build docs locally:

```bash
mkdocs serve
```

View at `http://localhost:8000`

When writing docs, follow the rules in `docs/WRITING_GUIDE.md`:

- No generic AI marketing speak
- Focus on technical accuracy
- Show code over prose

## Continuous Integration

On pull requests, CI runs:

- `pytest` (all tests)
- `ruff check` (linting)
- `black --check` (formatting)
- `mypy` (type checking)

Ensure all checks pass before requesting review.

## Next steps

- [JAX Architecture](architecture.md): Pure functional design principles
- [Configuration System](configuration.md): Hydra setup details
- [User Guide](../user-guide/concepts.md): Three-layer pattern overview
