# Add a custom agent

Implement your own RL algorithm by following the Agent protocol.

## Agent protocol

```python
from typing import Protocol
import chex

class Agent(Protocol):
    def select_action(
        self,
        params: AgentParams,
        obs: chex.Array,
        key: chex.PRNGKey
    ) -> tuple[chex.Array, AgentParams]:
        """Choose action given observation."""
        ...

    def update(
        self,
        params: AgentParams,
        transition: Transition,
    ) -> tuple[AgentParams, dict]:
        """Update parameters from experience."""
        ...
```

## Implementation guide

Coming soon.

See `src/myriad/agents/dqn.py` for reference implementation.
