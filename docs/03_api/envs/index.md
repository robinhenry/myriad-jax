# Environment

```{toctree}
:maxdepth: 1
:hidden:

classic/cartpole
bio/ccas_ccar
```

## Overview

```python
from myriad.envs import make_env

env = make_env("cartpole-control")
obs, state = env.reset(key, env.params, env.config)
obs, state, reward, done, info = env.step(key, state, action, env.params, env.config)
```

## Available Environments

| ID | Category | Description |
|----|----------|-------------|
| `cartpole-control` | [Classic](classic/cartpole.md) | Inverted pendulum balancing |
| `ccas-ccar-control` | [Bio](bio/ccas_ccar.md) | Bacterial gene circuit control |

## Factory Function

```{eval-rst}
.. autofunction:: myriad.envs.make_env
```

## Base Protocols

```{eval-rst}
.. automodule:: myriad.envs.environment
   :members:
   :undoc-members:
   :show-inheritance:
```

## Wrappers

```{eval-rst}
.. automodule:: myriad.envs.wrappers
   :members:
   :undoc-members:
   :show-inheritance:
```
