# Installation

## Requirements

- Python 3.11+
- JAX 0.7.2+
- Poetry (recommended) or pip

## Install from source

Clone the repository:

```bash
git clone https://github.com/robinhenry/aion.git
cd aion
```

Install dependencies with Poetry:

```bash
poetry install
```

Or with pip:

```bash
pip install -e .
```

## GPU support

### CUDA 12

```bash
poetry install --with gpu
```

Or:

```bash
pip install "jax[cuda12]"
```

### macOS (Metal)

JAX Metal support is experimental. Use CPU for now.

## Verify installation

```bash
python -c "import aion; print('Aion installed successfully')"
```

Run tests:

```bash
python -m pytest
```

## Next steps

- [Quickstart](quickstart.md): Your first training run
- [Core Concepts](../user-guide/concepts.md): Understanding Aion's architecture
- [Custom Environment](../user-guide/custom_env.md): Implement your own physics
