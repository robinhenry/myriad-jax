# Contributing to Myriad

Thank you for your interest in contributing! We welcome bug reports, documentation improvements, and new features.

## Bug Reports

If you find a bug or encounter an issue, we'd be very grateful if you could report it by [opening an issue on the repository](https://github.com/robinhenry/myriad-jax/issues).

Please add as much detail as possible so we can reproduce it.

## What to contribute

We see Myriad as a community-driven tool. If you have ideas about what should be added or improved, we'd love to hear them!

People usually contribute one of the following:

* New environments/tasks inline with the library's main goals
* New agents/algorithms
* Addressing [existing issues](https://github.com/robinhenry/myriad-jax/issues) (e.g., reported bugs)
* Improving [the documentation](https://myriad-jax.readthedocs.io/), such as filling in gaps, adding new tutorials, or clarifying confusing bits.

But, again, this is not an exhaustive list and we welcome new initiatives.

## Development Setup

```bash
git clone https://github.com/yourusername/myriad.git
cd myriad
poetry install --with dev,gpu
poetry run pre-commit install
```

Create a feature branch from `main` for your changes.

## Before Submitting a PR

Pre-commit hooks will run automatically on commit if you've installed them (see above). You may need to run `pre-commit run --all-files` initially.

You can also quickly simulate the CI pipeline without pushing to GitHub by running:
```bash
./scripts/ci-check.sh
```
which will run `mypy`, `ruff`, `pytest`, etc. locally.


## Code Requirements

Wherever possible, please follow the following guidelines:

- **Pure functions** - Required for JAX/JIT compatibility
- **Type hints + docstrings** - At least for all public functions
- **Follow patterns in `CLAUDE.md`**

## Documentation

The documentation lives in `docs/`, including the tutorials in `docs/tutorials/`.

See the [docs README](docs/README.md) for more information on how to build the docs locally, etc.

## Questions?

Check `CLAUDE.md` or open an issue - we'll happily help!
