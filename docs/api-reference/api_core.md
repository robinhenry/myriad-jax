:orphan:

# Core API reference

## Spaces

### Discrete

```python
from myriad.core.spaces import Discrete

action_space = Discrete(n=4)  # {0, 1, 2, 3}
```

### Box

```python
from myriad.core.spaces import Box

action_space = Box(
    low=-1.0,
    high=1.0,
    shape=(2,)
)
```

## Transitions

```python
from myriad.core.types import Transition

transition = Transition(
    obs=obs,
    action=action,
    reward=reward,
    next_obs=next_obs,
    done=done
)
```

## Replay buffer

```python
from myriad.core.replay_buffer import ReplayBuffer

# Create buffer
buffer = ReplayBuffer(
    max_size=100000,
    obs_shape=(4,),
    action_shape=(1,)
)

# Add transitions
buffer_state = buffer.add(buffer_state, transition)

# Sample batch
batch, buffer_state = buffer.sample(
    buffer_state,
    batch_size=64,
    key=key
)
```

## Factory functions

### make_env

```python
from myriad.envs import make_env

env = make_env("cartpole-control")
```

### make_agent

```python
from myriad.agents import make_agent
from myriad.core.spaces import Discrete

agent = make_agent(
    "dqn",
    action_space=Discrete(2)
)
```

## Training and evaluation

### train_and_evaluate

```python
from myriad.configs.default import Config
from myriad.platform import train_and_evaluate

config = Config(
    env={"_target_": "cartpole-control"},
    agent={"_target_": "dqn"},
    run={"num_envs": 10000, "steps_per_env": 100}
)

results = train_and_evaluate(config)
# Returns: TrainingResults with agent_state, metrics, config
```

### evaluate

```python
from myriad.platform import evaluate

# Evaluation-only (no training)
results = evaluate(config)
# Returns: dict with episode_return, episode_length, dones

# With episode trajectories
results = evaluate(config, return_episodes=True)
# Returns: dict with additional 'episodes' key containing full trajectories

# With pre-trained agent
results = evaluate(config, agent_state=trained_agent_state)
```

**Parameters:**

- `config` (Config): Training configuration
- `agent_state` (AgentState | None): Pre-initialized agent state. If None, agent initialized with random weights
- `return_episodes` (bool): Return full episode trajectories (observations, actions, rewards, dones)

**Returns:**

Dictionary containing:

- `episode_return`: Array of episode returns (num_eval_rollouts,)
- `episode_length`: Array of episode lengths (num_eval_rollouts,)
- `dones`: Boolean array indicating episode completion
- `episodes` (if return_episodes=True): Dictionary with:
    - `observations`: (num_eval_rollouts, max_steps, obs_dim)
    - `actions`: (num_eval_rollouts, max_steps, action_dim)
    - `rewards`: (num_eval_rollouts, max_steps)
    - `dones`: (num_eval_rollouts, max_steps)

## Next steps

- [Source code](https://github.com/robinhenry/myriad): Full implementation
- [Configuration System](../contributing/configuration.md): Hydra config reference
