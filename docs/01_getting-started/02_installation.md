# Installation

## Requirements

- Python 3.11+
- JAX 0.7.2+
- Poetry (recommended) or pip

## Install via `pip`

```bash
# Standard installation
pip install myriad-jax

# With generic GPU dependencies (checks for nvidia-related packages)
pip install "myriad-jax[gpu]"
```


## Install from source (for development)

Clone the repository:

```bash
git clone https://github.com/robinhenry/myriad.git
cd myriad
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
python -c "import myriad; print('Myriad installed successfully')"
```

Run tests:

```bash
python -m pytest
```
