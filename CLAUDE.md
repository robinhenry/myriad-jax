# Myriad Project Context
Platform: JAX-based Digital Twin Engine for Active SysID, Model-Based, and Model-Free Control.
Goal: Massively parallel experiments (Biological/Physics systems) on GPU/TPU.

Myriad is named after the Greek 'myrias', representing the ten thousand parallel environments the engine simulates simultaneously. It provides a Myriad of viewpoints from which to observe, identify, and control complex systems.

# Core Architecture & Terminology
- **Three-Layer Pattern:** MUST maintain strict separation:
  1. **Physics:** Pure dynamics (stateless, immutable).
  2. **Task:** Rewards/Termination logic (wraps Physics).
  3. **Learner:** Agent/Optimizer.
- **Protocols:** Use Python Protocols for `Environment` and `Agent` interfaces.
- **Config:** Hydra (YAML) + Pydantic validation.

# Architecture Map
- `src/myriad/core/`: Protocols and base types (the "Three Layers").
- `src/myriad/envs/`: JAX environments (Physics implementations).
- `src/myriad/agents/`: RL/Control algorithms (PPO, SAC, etc.).
- `src/myriad/platform/`: Infrastructure (logging, W&B, heavy compute).
- `src/myriad/utils/`: Utilities (plotting, math).
- `configs/`: Hydra definitions. Structure: `env/`, `agent/`, `run/`.
- `tests/`: Mirrors src structure. `tests/conftest.py` has global JAX fixtures.

# Critical Constraints (JAX & Performance)
- **Purity:** All env/agent functions MUST be pure (no side effects).
- **Data:** Use `jax.tree_util.tree_map` for PyTree ops.
- **Control Flow:** NO Python `if/else` in jitted paths. Use `jax.lax.cond`/`select`.
- **Masking:** Use mask-aware execution (auto-reset via `jax.lax.select`).
- **JIT:** Keep static args separate (`static_argnames`).

# JAX Development Protocol
Before writing complex JAX transformations (scan, cond, pmap), you MUST:
1. Create a tiny script to verify shape propagation.
2. Verify that logic does not trigger `TracerArrayConversionError` (python control flow).

# Core Workflows
- **Test:** `python -m pytest <files>` (Prefer single test files)
- **Train:** `python scripts/train.py run.total_timesteps=1e6` (Hydra syntax)
- **Lint/Format:** `ruff check --fix src/ tests/` | `black src/ tests/`
- **Type Check:** `mypy src/myriad/`

# Documentation Protocol
- **Constraint:** Do NOT use generic AI marketing speak.
- **Instruction:** When writing docs, READ `docs/WRITING_GUIDE.md` first.

# Reference Implementations (Copy these patterns!)
- **Ideal Agent:** `src/myriad/agents/dqn.py`
- **Ideal Environment:** `src/myriad/envs/cartpole.py`
- **Ideal Test:** `tests/envs/cartpole/test_physics.py`
