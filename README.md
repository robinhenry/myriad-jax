# Myriad [1^]

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

**JAX-native platform for massively parallel control, system identification, and active learning of uncertain, stochastic systems.** 💡 → 🤖 → 🔬

> [!TIP]
> **New to Myriad?**
>
> Myriad is in early active development. We aim to build a standard toolkit for the scientific community and warmly welcome contributions!
>
> Have an idea or just want to chat about stochastic control? [Open a discussion](https://github.com/robinhenry/myriad-jax/discussions) or reach out to Robin (robin.henry@eng.ox.ac.uk). 🤝

*Jump straight to [Installation](#installation) or [Quickstart](#quickstart) to see `myriad` in action, or check out the [full documentation](todo: add link).* 🤸🏾

> *TODO*: add 2 videos here: (a) learned agent on ccas-ccar, (b) real-time system ID via active learning.

## At a Glance

Myriad provides the computational backend to control biological and chemical systems where noise is not a bug but a feature. By leveraging JAX, we replace weeks of sequential lab time with minutes of GPU simulation.

### Key Features

* **⚡ Massive GPU Parallelism:** run 100k+ environments simultaneously.

* **🎲 Exact Stochastic Simulations:** native JAX implementation of the Gillespie Algorithm (SSA) for discrete, asynchronous molecular events.

* **∇ Differentiable "White-Box" Physics:** exposes underlying ODEs and jump processes for gradient-based system ID and active learning.

* **🛠 Research-Ready:** pre-configured with [Hydra](https://hydra.cc/), [Pydantic](https://docs.pydantic.dev/), and [W&B](https://wandb.ai/site) support.

*Interested in the story behind Myriad? Read our [Mission Statement & Philosophy](TODO: LINK_TO_DOCS_HERE).*

### Ecosystem & Fit

Myriad is designed to complement existing RL x JAX tools, not replace them.

| Feature | **Gymnasium** | **Brax** | **Myriad** |
| :--- | :--- | :--- | :--- |
| **Best For** | Standard RL benchmarks | Robotics & Locomotion | **Wet-Lab / Scientific Systems** |
| **Physics** | Black Box / Various | Rigid Body (Contacts) | **Stochastic, ODEs, Jump Processes** |
| **Differentiable?** | No | Yes | **Yes** |
| **System ID** | Low support | Low support | **Native** |
| **Primary Goal** | Agent Performance | Fast Physical Control | **Active Learning & Stochastic Control** |

* **Use Myriad if:** you model biological/chemical systems, require Gillespie/SSA stochasticity, or need active learning for parameter uncertainty.

* **Use [Brax](https://github.com/google/brax) if:** you need massive-scale robotics or contact dynamics.

* **Use [Gymnax](https://github.com/RobertTLange/gymnax) or [JaxMARL](https://github.com/FLAI/JaxMARL) if:** you need standard baselines or multi-agent RL.


## Installation

**Requirements:** Python 3.10+, JAX 0.7+

> [!IMPORTANT]
> **GPU Support:** JAX installation can be hardware-specific. We strongly recommend [installing JAX](https://github.com/google/jax#installation) according to your CUDA/cuDNN version *before* installing Myriad if you encounter issues.

### With `pip`

```bash
# Standard installation
pip install myriad-jax

# With generic GPU dependencies (checks for nvidia-related packages)
pip install "myriad-jax[gpu]"
```

### From source (for development)

```bash
git clone https://github.com/robinhenry/myriad-jax.git
cd myriad-jax
poetry install --with dev,gpu
```

## Quickstart

Myriad is designed to be used programmatically (for research loops) or via CLI (for massive sweeps).

### Python API

```python
from myriad import create_config, train_and_evaluate

# Configure a system ID experiment on a stochastic gene network
config = create_config(
    env="gene-circuit-v1",      # A stochastic gene circuit (Gillespie)
    agent="dqn",                # The algorithm to use
    num_envs=10_000,            # 10k parallel simulations (cells)
    sysid_mode=True             # Goal: Infer parameters, not just control
)

# Run the experiment (JIT-compiled & distributed on GPU)
results = train_and_evaluate(config)

# Analyze population statistics
print(f"Parameter Estimate Mean: {results.params.mean():.4f}")
print(f"Population Variance: {results.params.var():.4f}")
```

### CLI Usage

Leverage [Hydra](https://hydra.cc/) to run massive parameter sweeps or system ID experiments directly from the terminal without writing boilerplate.

```bash
# Train a DQN agent on 50,000 parallel cartpole environments
myriad train env=cartpole-control run.num_envs=50000 agent=dqn

# Switch to System Identification mode (inferring hidden physics parameters)
myriad train env=gene-circuit-v1 agent=pqn sysid=true
```

See the [Documentation](add link) for a full list of examples and configuration overrides.

## Flagship Environments

| Environment | Type | Description |
| --- | --- | --- |
| `gene-circuit-v1` | **Bio / Stochastic** | **(Flagship)** Stochastic gene expression with asynchronous division. Ideal for testing noise-robust control policies (SSA). |
| `cartpole-sysid` | **Classic / SysID** | An inverted pendulum where the agent must excite the system to infer the pole's mass and length while balancing. |
| `chem-reactor-v0` | **Chem / ODE** | Continuous Stirred Tank Reactor (CSTR) with uncertain reaction rates. |


## Citation

If you use Myriad in your work, please cite our paper:

```bibtex
@article{...}
```

---

[^1] Myriad is named after the Greek *myrias* ("ten thousand"), inspired by microfluidic "mother machines" that observe 100,000+ cells simultaneously. It brings this paradigm to computational research: providing a myriad of viewpoints from which to learn about and control complex systems—whether they are biological circuits, chemical reactors, or robotic swarms.
