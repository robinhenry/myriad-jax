# Core Types

Shared type definitions and data structures used throughout Myriad.

## Transitions

### Transition

```{eval-rst}
.. autoclass:: myriad.core.types.Transition
   :members:
   :undoc-members:
   :show-inheritance:
```

Represents a single environment transition for off-policy RL:

```python
from myriad.core.types import Transition

transition = Transition(
obs=current_obs,
action=action,
reward=reward,
next_obs=next_obs,
done=done
)
```

Used by:
- Replay buffers for experience storage
- Agent update functions for learning
- Trajectory collection

## Replay Buffer

### ReplayBuffer

```{eval-rst}
.. autoclass:: myriad.core.replay_buffer.ReplayBuffer
   :show-inheritance:
   :noindex:
```

**Example:**

```python
from myriad.core.replay_buffer import ReplayBuffer
import jax

# Create buffer
buffer = ReplayBuffer(
max_size=100_000,
obs_shape=(4,),
action_shape=(1,)
)

# Initialize state
key = jax.random.PRNGKey(0)
buffer_state = buffer.init(key)

# Add transitions
buffer_state = buffer.add(buffer_state, transition)

# Sample batch
key, sample_key = jax.random.split(key)
batch, buffer_state = buffer.sample(
buffer_state,
batch_size=64,
key=sample_key
)
```

### ReplayBufferState

```{eval-rst}
.. autoclass:: myriad.core.replay_buffer.ReplayBufferState
   :show-inheritance:
   :noindex:
```

Contains the replay buffer's internal state:
- `observations`: Stored observations
- `actions`: Stored actions
- `rewards`: Stored rewards
- `next_observations`: Stored next observations
- `dones`: Stored termination flags
- `position`: Current write position
- `size`: Current buffer size

## Base Models

### BaseModel

```{eval-rst}
.. autoclass:: myriad.core.types.BaseModel
   :members:
   :undoc-members:
   :show-inheritance:
```

Pydantic BaseModel with configuration for JAX compatibility. Used throughout the codebase for validated configuration objects.

## Utility Functions

### to_array

Converts structured observations (NamedTuples, dataclasses) to flat JAX arrays:

```python
from myriad.utils import to_array

# Works with arrays (no-op)
array_obs = jnp.array([1.0, 2.0, 3.0])
result = to_array(array_obs)  # Returns same array

# Works with NamedTuples
from collections import namedtuple
State = namedtuple("State", ["x", "v"])
state_obs = State(x=1.0, v=2.0)
result = to_array(state_obs)  # Returns jnp.array([1.0, 2.0])
```

This is useful when agents need to flatten structured environment observations.

## JAX Primitives

Myriad uses standard JAX types throughout:

- `chex.PRNGKey`: Random number generator key
- `chex.Array`: JAX array
- `chex.ArrayTree`: Pytree of arrays
- `jax.numpy`: JAX's NumPy-compatible array operations

See [JAX documentation](https://jax.readthedocs.io/) for details.

## Next Steps

- [Environment API](env.md): Environment-specific types
- [Agent API](agent.md): Agent-specific types
- [Platform API](platform.md): Training result types
