<h1 align="center"> Myriad</h1>

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/myriad-jax.svg)](https://pypi.org/project/myriad-jax/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://readthedocs.org/projects/myriad-jax/badge/?version=latest)](https://myriad-jax.readthedocs.io/en/latest/)
[![Build](https://img.shields.io/github/actions/workflow/status/robinhenry/myriad-jax/ci.yml?label=build)](https://github.com/robinhenry/myriad-jax/actions)
[![codecov](https://codecov.io/gh/robinhenry/myriad-jax/branch/main/graph/badge.svg)](https://codecov.io/gh/robinhenry/myriad-jax)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**JAX-native platform for massively parallel control, system identification, and active learning of uncertain, stochastic systems.**

> [!WARNING]
> **Myriad is in early active development.** APIs will change, documentation has gaps, and some features are still taking shape. Contributions, feedback, and ideas are very welcome! [Open a discussion](https://github.com/robinhenry/myriad-jax/discussions) or reach out to Robin (robin.henry@eng.ox.ac.uk).

**Documentation:** [docs](https://myriad-jax.readthedocs.io)

## At a Glance 🌟

Myriad is a **playground to explore RL, traditional control, system identification, and active learning** — with a focus on problems where uncertainty, stochasticity, and rare discrete dynamics play a big role and force us to study very large numbers of variants in parallel (*think: biology → system = cell, chemistry → system = reactor*). 🛝

It's a **ready-to-go experimental platform**. You can use one of the already-implemented tasks, algorithms, or implement your own and simply plug them in. Myriad will handle the intricacies of JAX/GPU optimization, training/evaluation loops, hyperparameter tracking, metrics logging, and many more not-so-fun things — freeing time for the more fun science and engineering bits. 👩🏾‍🔬👨🏻‍🔬

Last but not least, it yields results that are 100% reproducible. 🌟

> *Interested in the story behind Myriad? Read our [Motivation & Philosophy](https://myriad-jax.readthedocs.io/en/latest/introduction/motivation_philosophy.html).*

### Key Features

* **⚡ Massive GPU Parallelism:** run algorithms on 1M+ of environments simultaneously.

* **🏎️ JAX JIT Optimization:** Myriad is *fast*, even on CPU.

* **✅ 100% Reproducible:** Myriad is fully deterministic. Using the same initial random seed and configuration file will yield the same results → great for science.

* **🎲 Exact Stochastic Simulations:** native JAX implementation of the Gillespie Algorithm (aka SSA) for discrete, asynchronous molecular events.

* **∇ Differentiable "White-Box" Physics:** exposes underlying physics, ODEs, and jump processes for gradient-based system ID and active learning.

* **🛠 Research-Ready:** pre-configured with [Hydra](https://hydra.cc/), [Pydantic](https://docs.pydantic.dev/), and [W&B](https://wandb.ai/site) support.

## Installation 🌱

**Requirements:** Python 3.11+, JAX 0.7+.

> [!IMPORTANT]
> **GPU Support:** JAX installation can be hardware-specific. We strongly recommend [installing JAX](https://github.com/google/jax#installation) according to your CUDA/cuDNN version *before* installing Myriad if you encounter issues.

```bash
# Standard installation
pip install myriad-jax

# With generic GPU dependencies (checks for nvidia-related packages)
pip install "myriad-jax[gpu]"
```

## Quickstart 🏁

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
```

See the [Documentation](https://myriad-jax.readthedocs.io) for further information, including tutorials.

## Flagship Environments 🌍

To add.

## Contributing 🛠️

Our goal is for Myriad to become a platform that accelerates RL/control research, especially in the life sciences. As such, we'd love to have others contribute!

Please take a look at the [contributing guide](CONTRIBUTING.md) for instructions on how to add new environments, algorithms, or for other ways to contribute.

If in doubt, always feel free to reach out by [opening a discussion](https://github.com/robinhenry/myriad-jax/discussions).

<!-- ## Citation ✍️

If you use Myriad in your work, please cite the original paper:

```bibtex
@article{...}
``` -->

## See Also 🔎

Here is a non-exhaustive list of other JAX x RL libraries, some of which inspired the development of Myriad.

**Environments:**

* [Gymnax](https://github.com/RobertTLange/gymnax): classic environments including classic control, bsuite, MinAtar, and meta RL tasks.
* [Brax](https://github.com/google/brax): a differentiable physics engine for rigid body control tasks.
* [JaxMARL](https://github.com/FLAIROx/JaxMARL): multi-agent RL tasks.
* [Jumanji](https://github.com/instadeepai/jumanji): a diverse suite of environments, ranging from simple games to NP-hard combinatorial problems.
* [Pgx](https://github.com/sotetsuk/pgx): classic board game environments such as Chess, Go, and Shogi.
* [XLand-Minigrid](https://github.com/dunnolab/xland-minigrid): meta RL gridworld environments.
* [Craftax](https://github.com/MichaelTMatthews/Craftax): Crafter + NetHack in JAX.

**Algorithms:**

* [PureJaxRL](https://github.com/luchris429/purejaxrl): RL algorithms in JAX, inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl).
* [Evojax](https://github.com/google/evojax): neuroevolution algorithms.

---

*A little bit of history: Myriad is named after the Greek *myrias* ("ten thousand"), inspired by microfluidic "mother machines" that observe 100,000+ cells simultaneously. It brings this paradigm to computational research: providing a myriad of viewpoints from which to learn about and control complex systems — whether they are biological circuits, chemical reactors, or robotic swarms.*
