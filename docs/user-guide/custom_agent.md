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

See `src/aion/agents/dqn.py` for reference implementation.

## Next steps

- [DQN source](https://github.com/robinhenry/aion/blob/main/src/aion/agents/dqn.py): Reference code
- [Agents API Reference](../api-reference/agents.md): Protocol specification
- [Core Concepts](concepts.md): Understanding the three-layer pattern
