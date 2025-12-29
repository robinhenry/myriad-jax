# Contributing to Myriad

Thank you for your interest in contributing! We welcome bug reports, documentation improvements, and new features.

## Development Setup

```bash
git clone https://github.com/yourusername/myriad.git
cd myriad
poetry install
poetry run pre-commit install
```

Create a feature branch from `main` for your changes.

## Before Submitting a PR

Run tests and checks locally:

```bash
poetry run pytest tests/ -v
poetry run black src/ tests/
poetry run ruff check --fix src/ tests/
poetry run mypy src/myriad/
```

Pre-commit hooks will run automatically on commit. You may need to run `pre-commit run --all-files` a couple of times initially.

## Code Requirements

- **Pure functions** - Required for JAX/JIT compatibility
- **Type hints** - All public functions
- **Tests** - Mirror source structure, use `@pytest.mark.slow` for slow tests
- **Conventional commits** - `feat:`, `fix:`, `docs:`, etc.
- **Follow patterns in `CLAUDE.md`**

## Documentation

Examples go in `examples/` and are referenced using `literalinclude`:

```rst
.. literalinclude:: ../examples/02_basic_training.py
   :language: python
```

Build docs locally:
```bash
cd docs
poetry run sphinx-autobuild . _build/html --port 8000
```

## Questions?

Check `CLAUDE.md` or open an issue.
