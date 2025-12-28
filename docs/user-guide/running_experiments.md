# Running experiments

## Basic training

```bash
python scripts/train.py
```

Runs with default configuration from `configs/config.yaml`.

## Override config values

```bash
python scripts/train.py \
  run.num_envs=10000 \
  run.steps_per_env=100 \
  agent.learning_rate=3e-4
```

Override any parameter from the command line. Parameters use dot notation for nested values.

!!! tip "Finding Default Values"
    To find what the default value of a parameter is:

    - **Agent parameters:** Check factory function `src/myriad/agents/{agent_name}.py:make_agent()`
    - **Environment parameters:** Check factory function `src/myriad/envs/{env_name}/*.py:make_env()`
    - **Run parameters:** Check Pydantic model `src/myriad/configs/default.py:RunConfig`

    YAML files contain experiment-specific overrides, not the source of truth for defaults.

## Switch config groups

```bash
python scripts/train.py \
  env=cartpole_sysid \
  agent=pqn
```

Load different base configurations from `configs/agent/`, `configs/env/`, or `configs/run/`.

## Parameter sweeps

Coming soon: W&B sweep integration.

## Programmatic usage

```python
from myriad.configs.default import Config
from myriad.platform import train_and_evaluate

config = Config(
    env={"_target_": "cartpole-control"},
    agent={"_target_": "dqn"},
    run={"num_envs": 10000, "steps_per_env": 100}
)

train_and_evaluate(config)
```

## Evaluation-only mode

Use `evaluate()` for non-learning controllers (PID, MPC, scripted) or benchmarking pre-trained agents:

```python
from myriad.platform import evaluate

# Random policy baseline
results = evaluate(config)
print(f"Mean return: {results['episode_return'].mean()}")

# Pre-trained agent
results = evaluate(config, agent_state=trained_agent_state)
```

## Episode collection

Collect full trajectories during evaluation for analysis or visualization:

```python
# Return episode data (observations, actions, rewards, dones)
results = evaluate(config, return_episodes=True)

episodes = results['episodes']
# Shape: (num_eval_rollouts, max_steps, ...)
obs = episodes['observations']     # (N, T, obs_dim)
actions = episodes['actions']       # (N, T, action_dim)
rewards = episodes['rewards']       # (N, T)
dones = episodes['dones']           # (N, T)

# Use episode_length to trim padding
for i in range(len(results['episode_length'])):
    ep_len = int(results['episode_length'][i])
    valid_obs = obs[i, :ep_len]  # No padding
```

!!! note
    Episode collection compiles a separate code path on first use. Subsequent calls reuse cached compilation.

## Periodic episode saving

Save episodes to disk during training for qualitative monitoring:

```python
config = Config(
    run=RunConfig(
        eval_frequency=10000,                  # Eval every 10k steps
        eval_rollouts=5,                       # Run 5 episodes
        eval_episode_save_frequency=50000,     # Save every 50k steps (5x less)
        eval_episode_save_count=2,             # Save first 2 episodes (to episodes/)
    )
)

train_and_evaluate(config)
```

Episodes saved to `episodes/step_{global_step}/episode_{i}.npz`:

```python
import numpy as np

# Load saved episode
data = np.load("episodes/step_00050000/episode_0.npz")

# Trajectory data (no padding)
observations = data['observations']  # (episode_length, obs_dim)
actions = data['actions']            # (episode_length, action_dim)
rewards = data['rewards']            # (episode_length,)

# Metadata
ep_len = int(data['episode_length'])
ep_return = float(data['episode_return'])
global_step = int(data['global_step'])
```

Episodes automatically logged to W&B as artifacts when enabled.

!!! warning
    Set `eval_episode_save_frequency` >> `eval_frequency` to avoid storage bloat. Recommended: 5-10x eval frequency.

## Next steps

- [Configuration System](../contributing/configuration.md): Hydra config details (for advanced users)
- [Custom Environment Guide](custom_env.md): Implement your own environments
- [Custom Agent Guide](custom_agent.md): Implement learning algorithms
