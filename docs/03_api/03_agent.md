# Agent API

Myriad agents follow a pure functional design for RL and control algorithms.

## Factory Function

```{eval-rst}
.. autofunction:: myriad.agents.make_agent
```

## Agent Container

The `Agent` container bundles all agent functions and parameters:

```{eval-rst}
.. autoclass:: myriad.agents.agent.Agent
   :members:
   :undoc-members:
   :show-inheritance:
```

## Protocols

### AgentParams

```{eval-rst}
.. autoclass:: myriad.agents.agent.AgentParams
   :members:
   :undoc-members:
   :show-inheritance:
```

### AgentState

```{eval-rst}
.. autoclass:: myriad.agents.agent.AgentState
   :members:
   :undoc-members:
   :show-inheritance:
```

## Available Agents

### Deep Q-Network (DQN)

**ID:** `dqn`

Deep Q-Network with epsilon-greedy exploration for discrete action spaces.

**Action space:** Discrete(n)

**Key parameters:**
- `learning_rate`: Adam learning rate (default: 1e-3)
- `gamma`: Discount factor (default: 0.99)
- `epsilon_start`: Initial exploration (default: 1.0)
- `epsilon_end`: Final exploration (default: 0.05)
- `epsilon_decay_steps`: Decay steps (default: 50000)
- `target_network_frequency`: Target update interval (default: 1000)

**Configuration:** `configs/agent/dqn.yaml`

**Reference:** [src/myriad/agents/rl/dqn.py](https://github.com/robinhenry/myriad-jax/blob/main/src/myriad/agents/rl/dqn.py)

### Parametric Q-Network (PQN)

**ID:** `pqn`

Parametric Q-Network for continuous action spaces.

**Action space:** Box(n)

**Key parameters:**
- `learning_rate`: Adam learning rate (default: 3e-4)
- `gamma`: Discount factor (default: 0.99)
- `buffer_size`: Replay buffer size (default: 100000)

**Configuration:** `configs/agent/pqn.yaml`

**Reference:** [src/myriad/agents/rl/pqn.py](https://github.com/robinhenry/myriad-jax/blob/main/src/myriad/agents/rl/pqn.py)

### Random

**ID:** `random`

Uniform random action selection baseline.

**Action space:** Discrete(n) or Box(n)

**Parameters:** None

### Bang-Bang Controller

**ID:** `bangbang`

Simple threshold-based controller for control tasks.

**Action space:** Discrete(2) or Box(1)

**Parameters:**
- `threshold`: Switching threshold (default: 0.0)

### PID Controller

**ID:** `pid`

Classical PID controller for continuous control.

**Action space:** Box(1)

**Parameters:**
- `kp`: Proportional gain
- `ki`: Integral gain
- `kd`: Derivative gain

## Agent Interface

All agents implement a consistent pure functional interface:

```python
# Initialize: key, initial_obs, params → state
agent_state = agent.init(key, initial_obs, agent.params)

# Select action: key, obs, state, params, training → action, state
action, agent_state = agent.select_action(
key, obs, agent_state, agent.params, training=True
)

# Update: key, state, transition, params → state, metrics
agent_state, metrics = agent.update(
key, agent_state, transition, agent.params
)
```

This design enables:
- Shared agent state across parallel environments
- Efficient batched inference
- JIT compilation
- Functional purity
