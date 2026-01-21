# API Reference

Welcome to the Myriad API reference. This documentation is automatically generated from the source code docstrings.

## Core Components

The Myriad API is organized around three core concepts, reflecting the three-layer architecture:

**[Environments](02_env.md)** - Pure JAX environment dynamics and task definitions

**[Agents](03_agent.md)** - RL and control algorithms for decision-making

**[Spaces](04_spaces.md)** - Action and observation space definitions

**[Platform](05_platform.md)** - Training infrastructure and utilities

**[Core Types](06_types.md)** - Shared types and data structures

## Quick Links

```{toctree}
:maxdepth: 2

02_env
03_agent
04_spaces
05_platform
06_types
```

## Entry Points

The most commonly used functions:

- {py:func}`myriad.platform.train_and_evaluate` - Train and evaluate an agent
- {py:func}`myriad.platform.evaluate` - Evaluate a (pre-trained) agent
- {py:func}`myriad.envs.make_env` - Create an environment by ID
- {py:func}`myriad.agents.make_agent` - Create an agent by ID

## Design Philosophy

Myriad's API follows these principles:

1. **Pure functions**: All core functions are pure and JIT-compatible
2. **Explicit state**: State is passed explicitly, never hidden
3. **Static vs dynamic**: Configuration is compile-time static, parameters are runtime dynamic
4. **Protocols over inheritance**: Flexible, type-safe interfaces via Python Protocols
