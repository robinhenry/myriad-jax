:orphan:

# Myriad Documentation

This directory contains the Sphinx documentation for Myriad.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
poetry install --with dev
```

### Build HTML Documentation

Using the Makefile:

```bash
cd docs
make html
```

Or using sphinx-build directly:

```bash
cd docs
sphinx-build -b html . _build/html
```

The built documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser to view it.

### Live Preview with Auto-Rebuild

For development, use sphinx-autobuild to automatically rebuild when files change:

```bash
cd docs
sphinx-autobuild . _build/html --port 8000
```

Then open http://localhost:8000 in your browser.

### Clean Build

To remove all built documentation:

```bash
cd docs
make clean
```

## Documentation Structure

- `conf.py`: Sphinx configuration
- `index.md`: Documentation home page
- `getting-started/`: Installation and quickstart guides
- `user-guide/`: User-facing documentation
- `api-reference/`: API documentation
- `contributing/`: Contributor guides
- `stylesheets/`: Custom CSS
- `_build/`: Generated documentation (git-ignored)

## Theme

The documentation uses the [Furo](https://pradyunsg.me/furo/) theme, which provides a clean, modern appearance with excellent readability.

## Markdown Support

Documentation is written in Markdown and uses [MyST Parser](https://myst-parser.readthedocs.io/) for Sphinx integration. This allows you to use standard Markdown with additional Sphinx directives.
