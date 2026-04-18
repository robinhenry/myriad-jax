# Environment

```{toctree}
:maxdepth: 1
:hidden:

classic/index
bio/index
wrappers
```

## Overview

```python
from myriad.envs import make_env

env = make_env("cartpole-control")
obs, state = env.reset(key, env.params, env.config)
obs, state, reward, done, info = env.step(key, state, action, env.params, env.config)
```

## Available Environments

| ID | Category | Environment | Description |
|----|----------|-------------|-------------|
| `cartpole-control` | [Classic](classic/index.md) | [CartPole](classic/cartpole.md) | Inverted pendulum balancing |
| `pendulum-control` | [Classic](classic/index.md) | [Pendulum](classic/pendulum.md) | Swing-up control |
| `opto-hill-1d-sysid` | [Biology](bio/index.md) | [Opto Hill 1D](bio/opto_hill_1d.md) | 1D optogenetic circuit, continuous light input |
| `ccasr-gfp-control` | [Biology](bio/index.md) | [CcaSR-GFP](bio/ccasr_gfp.md) | CcaS/CcaR + GFP optogenetic gene circuit |

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
