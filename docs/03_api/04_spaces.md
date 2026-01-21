# Spaces API

Action and observation spaces define the structure of environment inputs and outputs.

## Overview

Myriad uses Gym-compatible space definitions for describing action and observation spaces. Spaces define:

- The structure and shape of actions/observations
- Valid ranges for continuous values
- Number of discrete choices

## Available Spaces

### Discrete

```{eval-rst}
.. autoclass:: myriad.core.spaces.Discrete
   :members:
   :undoc-members:
   :show-inheritance:
```

**Example:**

```python
from myriad.core.spaces import Discrete

# 4 discrete actions: {0, 1, 2, 3}
action_space = Discrete(n=4)

# Sample a random action
action = action_space.sample(key)  # Returns int in [0, 3]

# Check if action is valid
is_valid = action_space.contains(2)  # True
```

### Box

```{eval-rst}
.. autoclass:: myriad.core.spaces.Box
   :members:
   :undoc-members:
   :show-inheritance:
```

**Example:**

```python
from myriad.core.spaces import Box
import jax.numpy as jnp

# 2D continuous action in [-1, 1]²
action_space = Box(
low=-1.0,
high=1.0,
shape=(2,)
)

# Sample a random action
action = action_space.sample(key)  # Returns array of shape (2,)

# With different bounds per dimension
action_space = Box(
low=jnp.array([0.0, -10.0]),
high=jnp.array([1.0, 10.0]),
shape=(2,)
)
```

## Space Protocol

```{eval-rst}
.. autoclass:: myriad.core.spaces.Space
   :members:
   :undoc-members:
   :show-inheritance:
```

All spaces implement:

- `sample(key: PRNGKey) -> Array`: Sample a random element
- `contains(x: Array) -> bool`: Check if element is valid

## Usage in Environments

Environments define their action space via the `get_action_space` function:

```python
from myriad.envs import make_env

env = make_env("cartpole-control")
action_space = env.get_action_space(env.config)
# Returns: Discrete(n=2)
```

## Usage in Agents

Agents receive the action space during creation:

```python
from myriad.agents import make_agent
from myriad.core.spaces import Discrete

agent = make_agent(
"dqn",
action_space=Discrete(2)
)
```

The agent's behavior adapts to the space type:
- **Discrete**: DQN uses discrete action selection
- **Box**: PQN uses continuous action parameterization
