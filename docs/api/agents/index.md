# Agent

```{toctree}
:maxdepth: 1
:hidden:

classical/index
rl/index
```

## Overview

```python
from myriad.agents import make_agent

agent = make_agent("dqn", action_space=env.action_space)
state = agent.init(key, sample_obs, agent.params)
action, state = agent.select_action(key, obs, state, agent.params, deterministic=False)
```

## Available Agents

| ID | Category | Agent | Description |
|----|----------|-------|-------------|
| `random` | [Classical](classical/index.md) | [Random](classical/random.md) | Uniform random action selection |
| `bangbang` | [Classical](classical/index.md) | [Bang-Bang](classical/bangbang.md) | Threshold-based bang-bang controller |
| `pid` | [Classical](classical/index.md) | [PID](classical/pid.md) | Proportional-Integral-Derivative controller |
| `dqn` | [RL](rl/index.md) | [DQN](rl/dqn.md) | Deep Q-Network (discrete actions) |
| `pqn` | [RL](rl/index.md) | [PQN](rl/pqn.md) | Parallelized Q-Network (on-policy) |

## Factory Function

```{eval-rst}
.. autofunction:: myriad.agents.make_agent
```

## Base Protocols

```{eval-rst}
.. automodule:: myriad.agents.agent
   :members:
   :undoc-members:
   :show-inheritance:
```
