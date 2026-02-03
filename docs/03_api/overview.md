# Overview

Welcome to the Myriad API reference. This documentation is automatically generated from the source code docstrings.

## Core Components

The Myriad API is organized around four core concepts:

**[Environments](envs/index.md)** - Pure JAX environment dynamics and task definitions

**[Agents](agent.md)** - RL and control algorithms for decision-making

**[Platform](platform.md)** - Training infrastructure, evaluation, and configuration

**[Core & Utilities](core.md)** - Spaces, shared types, replay buffers, and helper functions

## Entry Points

The most commonly used functions:

- {py:func}`myriad.platform.train_and_evaluate` - Train and evaluate an agent
- {py:func}`myriad.platform.evaluate` - Evaluate a (pre-trained) agent
- {py:func}`myriad.envs.make_env` - Create an environment by ID
- {py:func}`myriad.agents.make_agent` - Create an agent by ID
