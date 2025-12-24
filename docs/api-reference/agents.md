# Agent reference

## Available agents

### DQN

**ID:** `dqn`

**Description:** Deep Q-Network with epsilon-greedy exploration.

**Action space:** `Discrete(n)`

**Key parameters:**

- `learning_rate`: Adam learning rate (default: 1e-3)
- `gamma`: Discount factor (default: 0.99)
- `epsilon_start`: Initial exploration (default: 1.0)
- `epsilon_end`: Final exploration (default: 0.05)
- `epsilon_decay_steps`: Decay steps (default: 50000)
- `target_network_frequency`: Target update interval (default: 1000)

**Config:** `configs/agent/dqn.yaml`

### PQN

**ID:** `pqn`

**Description:** Parametric Q-Network for continuous actions.

**Action space:** `Box(n)`

**Key parameters:**

- `learning_rate`: Adam learning rate (default: 3e-4)
- `gamma`: Discount factor (default: 0.99)
- `buffer_size`: Replay buffer size (default: 100000)

**Config:** `configs/agent/pqn.yaml`

### Random

**ID:** `random`

**Description:** Uniform random action selection baseline.

**Action space:** `Discrete(n)` or `Box(n)`

**Parameters:** None

## Agent protocol

```python
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
        """Update parameters from transition."""
        ...
```

## Next steps

- [Custom Agent Guide](../user-guide/custom_agent.md): Implementation guide
- [DQN source](https://github.com/robinhenry/aion/blob/main/src/aion/agents/dqn.py): Reference code
