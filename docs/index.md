# Myriad

**A JAX-native platform for massively parallel system identification and control.**

## The problem

Standard RL environments (Gym, Gymnax) give you one robot and ask you to control it.

Myriad gives you **100,000 uncertain physical systems** in parallel and asks you to:

1. **Identify** their hidden parameters (System ID)
2. **Control** them to a target (RL/MPC)
3. **Plan** experiments to reduce uncertainty (Active Learning)

## Quickstart

```python
from myriad.configs.default import Config
from myriad.platform.runner import train_and_evaluate

# Create config
config = Config(
    env={"_target_": "cartpole-control"},
    agent={"_target_": "dqn"},
    run={"num_envs": 10000, "total_timesteps": 1_000_000}
)

# Train
train_and_evaluate(config)
```

Or use the CLI:

```bash
python scripts/train.py \
  env=cartpole_control \
  agent=dqn \
  run.num_envs=10000 \
  run.total_timesteps=1e6
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

| Domain | Hook | What you get |
|--------|------|--------------|
| **RL research** | Train PPO in 4 seconds on a single GPU | CartPole with 100k randomized masses/lengths |
| **Control theory** | Gradient-based MPC over stiff ODEs | Chemical reactor with drifting parameters |
| **Scientific ML** | Recover physics from 100k short trajectories | Damped oscillator parameter estimation |
| **Synthetic biology** | In-silico optimization before lab work | Gene circuit control (the "flagship" demo) |

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
