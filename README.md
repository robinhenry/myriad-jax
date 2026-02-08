# Myriad

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

**JAX-native platform for massively parallel control, system identification, and active learning of uncertain, stochastic systems.**

> [!WARNING]
> **Myriad is in early active development — we're building in public.** APIs will change, documentation has gaps, and some features are still taking shape. Things will improve over time. Contributions, feedback, and ideas are very welcome — [open a discussion](https://github.com/robinhenry/myriad-jax/discussions) or reach out to Robin (robin.henry@eng.ox.ac.uk).

*Jump straight to [Installation](#installation) or [Quickstart](#quickstart) to see Myriad in action, or check out the [full documentation](todo: add link).* 🤸🏾

> *TODO*: add 2 videos here: (a) learned agent on ccas-ccar, (b) real-time system ID via active learning.

## At a Glance

Myriad is a **playground to explore RL, traditional control, system identification, and active learning** — with a focus on problems where uncertainty, stochasticity, and rare discrete dynamics play a big role and force us to study very large numbers of variants in parallel (*think: biology → system = cell, chemistry → system = reactor, THIRD EXAMPLE?*). 🛝

It's a **ready-to-go experimental platform**. You can use one of the already-implemented tasks/problems, algorithms, or implement your own and simply plug them in. Myriad will handle the intricacies of JAX/GPU optimization, training/evaluation loops, hyperparameter tracking, metrics logging, and many more not-so-fun things — freeing time for the more fun science and engineering bits. 👩🏾‍🔬👨🏻‍🔬

Last but not least, it yields results that are 100% reproducible. 🌟

> *Interested in the story behind Myriad? Read our [Mission Statement & Philosophy](TODO: LINK_TO_DOCS_HERE).*


### Key Features

* **⚡ Massive GPU Parallelism:** run algorithms on 1M+ of environments simultaneously.

* **🏎️ JAX JIT Optimization:** Myriad is *fast*, even on CPU.

* **✅ 100% Reproducible:** Myriad is fully deterministic. Using the same initial random seed and configuration file will yield the same results → great for science.

* **🎲 Exact Stochastic Simulations:** native JAX implementation of the Gillespie Algorithm (aka SSA) for discrete, asynchronous molecular events.

* **∇ Differentiable "White-Box" Physics:** exposes underlying physics, ODEs, and jump processes for gradient-based system ID and active learning.

* **🛠 Research-Ready:** pre-configured with [Hydra](https://hydra.cc/), [Pydantic](https://docs.pydantic.dev/), and [W&B](https://wandb.ai/site) support.

### Ecosystem

Many amazing RL x JAX tools already exist! Here's how we believe Myriad complements them.

| Feature | **Gymnasium/Gymnax** | **Brax** | **Myriad** |
| :--- | :--- | :--- | :--- |
| **Best For** | Standard RL benchmarks | Robotics & Locomotion | **Wet-Lab / Scientific Systems** |
| **Physics** | Black Box / Various | Rigid Body (Contacts) | **Stochastic, ODEs, Jump Processes** |
| **Differentiable?** | No | Yes | **Yes** |
| **System ID** | Low support | Low support | **Key focus** |
| **Primary Goal** | Agent Performance | Fast Physical Control | **Active Learning & Stochastic Control** |

* **Use Myriad if:** you model biological/chemical systems, require Gillespie/SSA stochasticity, or need active learning for parameter uncertainty.

* **Use [Brax](https://github.com/google/brax) if:** you need massive-scale robotics or contact dynamics.

* **Use [Gymnasium](https://gymnasium.farama.org/), [Gymnax](https://github.com/RobertTLange/gymnax) or [JaxMARL](https://github.com/FLAI/JaxMARL) if:** you need standard baselines or multi-agent RL.


## Installation

**Requirements:** Python 3.10+, JAX 0.7+.

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

# Configure a gene expression control experiment across 10k cells
config = create_config(
    env="gene-circuit-v1",      # A stochastic gene circuit (Gillespie)
    agent="dqn",                # The algorithm to use (eg, 'pid', 'pqn')
    num_envs=10_000,            # 10k parallel simulations (cells)
    scan_autotune=True,         # Automatically optimize GPU training loop parameters
)

# Run the experiment (JIT-compiled & distributed on GPU)
results = train_and_evaluate(config)

# Inspect performance metrics
metrics = results.eval_metrics
print(f"Return (mean +/- std): {metrics.mean_return} +/- {metrics.std_return}")

# Quickly plot convergence curve
TO ADD
```

### CLI Usage

Leverage [Hydra](https://hydra.cc/) to run massive parameter sweeps or experiments directly from the terminal.

```bash
# Train a DQN agent on 50,000 parallel cartpole environments
myriad train env=cartpole-control run.num_envs=50000 agent=dqn

# Switch to System Identification mode (inferring hidden physics parameters)
myriad train env=gene-circuit-v1 agent=pqn sysid=true
```

See the [Documentation](add link) for a full list of examples and configuration overrides.

## Flagship Environments

To add.

<!-- ## Citation

If you use Myriad in your work, please cite the original paper:

```bibtex
@article{...}
``` -->

---

*A little bit of history: Myriad is named after the Greek *myrias* ("ten thousand"), inspired by microfluidic "mother machines" that observe 100,000+ cells simultaneously. It brings this paradigm to computational research: providing a myriad of viewpoints from which to learn about and control complex systems — whether they are biological circuits, chemical reactors, or robotic swarms.*
