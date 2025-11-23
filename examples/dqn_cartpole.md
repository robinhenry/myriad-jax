# DQN on CartPole Example

This example demonstrates how to train a DQN (Deep Q-Network) agent on the CartPole environment.

## Quick Start

Train a DQN agent on CartPole with default hyperparameters:

```bash
python scripts/train.py agent=dqn_agent env=cartpole
```

## Configuration

The DQN agent has the following hyperparameters (see `configs/agent/dqn_agent.yaml`):

- `learning_rate`: Learning rate for the Adam optimizer (default: 1e-3)
- `gamma`: Discount factor for future rewards (default: 0.99)
- `epsilon_start`: Initial exploration rate (default: 1.0)
- `epsilon_end`: Final exploration rate (default: 0.05)
- `epsilon_decay_steps`: Steps to decay epsilon from start to end (default: 10000)
- `target_network_frequency`: Steps between target network updates (default: 500)
- `tau`: Soft update coefficient, 1.0 = hard update (default: 1.0)

## Customizing Training

You can override any configuration parameter from the command line:

```bash
# Train for fewer timesteps with more parallel environments
python scripts/train.py \
    agent=dqn_agent \
    env=cartpole \
    run.total_timesteps=50000 \
    run.num_envs=8

# Adjust DQN hyperparameters
python scripts/train.py \
    agent=dqn_agent \
    env=cartpole \
    agent.learning_rate=5e-4 \
    agent.epsilon_decay_steps=20000

# Enable W&B logging
python scripts/train.py \
    agent=dqn_agent \
    env=cartpole \
    wandb.enabled=true \
    wandb.group=dqn_experiments
```

## Expected Performance

DQN should solve CartPole (achieve consistent high scores) within 10,000-20,000 timesteps. The environment is considered solved when the agent achieves an average reward of 195+ over 100 consecutive episodes.

## Algorithm Details

This implementation uses:
- Experience replay buffer for off-policy learning
- Epsilon-greedy exploration with linear decay
- Target network for stable Q-value updates
- Mean squared error loss on TD-errors

The Q-network is a simple MLP with two hidden layers of 64 units each with ReLU activations.
