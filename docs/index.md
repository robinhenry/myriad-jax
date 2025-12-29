# Myriad

**JAX-native platform for biological systems, stochastic dynamics, and active learning at scale.**

## What makes Myriad different?

Myriad builds on the foundations of excellent libraries like Gymnasium and Brax, with a specific focus:

- **Biological & chemical systems**: Gene circuits, metabolic networks, reaction-diffusion systems
- **Stochastic dynamics**: Low-copy molecular noise, asynchronous events, multi-timescale processes
- **Active system identification**: Learning unknown parameters through intelligent experiment design
- **JAX-native parallelism**: 100,000+ environments on a single GPU for biological simulations

While Gymnasium excels at standardized benchmarks and Brax at robotics simulation, Myriad focuses on the unique challenges of biological and chemical systems where uncertainty and stochasticity are fundamental.

## The challenge

Many scientific domains require:

1. **Identifying** hidden parameters from noisy observations (System ID)
2. **Controlling** complex, uncertain dynamics (RL/Optimal Control)
3. **Planning** informative experiments (Active Learning)
4. **Scaling** to thousands of parameter variants in parallel

Myriad provides these capabilities with a focus on biological and chemical systems.

## Quickstart

**Python API:**

```{literalinclude} ../examples/07_quickstart_simple.py
:language: python
:lines: 8-18
```

**Or use the CLI:**

```bash
myriad train \
  env=cartpole_control \
  agent=dqn \
  run.num_envs=10000 \
  run.steps_per_env=100
```

## Key features

### Massively parallel
Run 100,000+ environments simultaneously on a single GPU. JAX's `vmap` and XLA compilation deliver millions of steps per second.

### Three-layer architecture
Separate physics from task logic from learning algorithms:

- **Physics layer**: Pure dynamics (`f(state, action, params) → next_state`)
- **Task layer**: Observation/reward wrappers (control vs. system ID)
- **Learner layer**: Standard RL agents (DQN, PQN, Random)

See [Core Concepts](user-guide/concepts.md) for details.

### Direct differentiable access
Unlike Gym-style environments that hide dynamics inside `step()`, Myriad exposes pure physics functions. Use them for:

- Model Predictive Control (MPC)
- Gradient-based trajectory optimization
- Neural ODEs and hybrid models

### Built for science
Domain randomization, parameter sweeps, and active experimental design are first-class features, not afterthoughts.

## Use cases

Myriad excels where **standard RL environments fall short**:

| Domain | Challenge | What Myriad Provides |
|--------|-----------|---------------------|
| **Synthetic Biology** | Design genetic circuits before lab experiments | Stochastic gene expression with 100k parameter variants |
| **Systems Biology** | Identify kinetic parameters from noisy data | Active learning for optimal perturbation experiments |
| **Biochemical Engineering** | Control bioreactors with uncertain kinetics | Simultaneous parameter learning and control |
| **Chemical Engineering** | Optimize reactors with catalyst degradation | Multi-timescale stochastic dynamics |
| **RL Research** | Benchmark on stochastic environments | 100k parallel environments with randomized physics |

### Complementary to existing tools

Myriad complements rather than replaces existing libraries:

| Strength | Gymnasium | Brax | Myriad |
|----------|-----------|------|--------|
| **Best for** | Standard RL benchmarks | Robotics simulation | Biological systems |
| **Parallelism** | VectorEnv (moderate) | vmap (high) | vmap (high) |
| **Backend** | NumPy + Optional wrappers | JAX | JAX |
| **Unique features** | Established baselines, broad ecosystem | Rigid-body physics, robotics focus | Stochastic bio systems, active SysID |
| **When to use** | Reproducing papers, standard benchmarks | Robotics, differentiable physics | Gene circuits, biochemical systems |

These tools work well together:
- **Prototype** algorithm on Gymnasium's simple environments
- **Scale** robotics simulations with Brax
- **Experiment** with biological systems using Myriad

Each library excels in its domain. Myriad fills a gap for stochastic biological and chemical systems where active parameter learning is central to the research question.

## What's implemented

**Environments:**

- `cartpole-control`: Standard stabilization task
- `cartpole-sysid`: Active parameter learning variant
- `ccas_ccar_v1`: Gene circuit with growth/division dynamics

**Agents:**

- `dqn`: Deep Q-Network (discrete actions)
- `pqn`: Parametric Q-Network (continuous actions)
- `random`: Baseline

**Infrastructure:**

- Hydra configuration system
- W&B logging integration
- Pure JAX, fully jitted training loops
- Protocol-based extensibility

## The "ImageNet" benchmark

The gene circuit environment (`ccas_ccar_v1`) serves as the stress test. If the platform can handle:

- Multi-timescale dynamics (protein expression, cell growth)
- Stochastic behavior (low-copy molecular noise)
- Asynchronous events (unpredictable cell division)
- High-dimensional parameter spaces (10+ unknowns)

...then it can handle your 50 drones, chemical reactors, or financial models.

## Where to start

**New to Myriad?**

- [Installation](getting-started/installation.md): Set up Myriad in 5 minutes
- [Quickstart](getting-started/quickstart.md): Your first training run

**Building experiments?**

- [Core Concepts](user-guide/concepts.md): Understand the three-layer architecture
- [Custom Environment](user-guide/custom_env.md): Implement your physics
- [Running Experiments](user-guide/running_experiments.md): Train agents at scale

**Modifying the engine?**

- [Contributing Setup](contributing/setup.md): Development environment
- [JAX Architecture](contributing/architecture.md): Pure functional design constraints

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting Started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: User Guide

user-guide/concepts
user-guide/custom_env
user-guide/custom_agent
user-guide/running_experiments
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API Reference

api/index
api/env
api/agent
api/spaces
api/platform
api/types
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Contributing

contributing/setup
contributing/configuration
contributing/architecture
```
