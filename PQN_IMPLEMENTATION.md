# PQN (Parallelized Q-Network) Implementation

This document describes the PQN agent implementation added to the Aion platform.

## Overview

PQN is an on-policy value-based RL algorithm that differs from DQN in several key ways:
- **No replay buffer**: Uses fresh rollout data (on-policy)
- **No target network**: Uses LayerNorm for stability instead
- **Lambda-returns**: Uses GAE-style lambda-returns instead of 1-step TD targets
- **Multi-epoch training**: Trains multiple epochs on collected rollouts

## Files Added

1. **Agent Implementation**: `src/aion/agents/pqn.py`
   - QNetwork with LayerNorm
   - Lambda-return computation
   - Multi-epoch training with minibatch shuffling

2. **Agent Configuration**: `configs/agent/pqn_agent.yaml`
   - Network architecture (128 hidden size, 2 layers)
   - Optimizer settings (lr=2.5e-4, gradient clipping)
   - RL parameters (gamma=0.99, lambda=0.65)
   - Exploration schedule
   - Training config (4 epochs, 4 minibatches)

3. **Run Configuration**: `configs/run/pqn_cartpole.yaml`
   - 8 parallel environments
   - 32 rollout steps per update
   - Batch size of 256 (8 envs × 32 steps)

4. **Main Configuration**: `configs/pqn_cartpole.yaml`
   - Complete configuration combining all components

## Files Modified

1. **Platform Runner**: `src/aion/platform/runner.py`
   - Added rollout collection function for on-policy training
   - Made replay buffer optional
   - Added support for both step-by-step (DQN) and rollout-based (PQN) training modes
   - Updated training loop to handle both modes

2. **Config Schema**: `src/aion/configs/default.py`
   - Added `rollout_steps` parameter to RunConfig

3. **Agent Registry**: `src/aion/agents/__init__.py`
   - Registered PQN agent

## Usage

### Training with PQN on CartPole

```bash
# Using the complete PQN CartPole configuration
python scripts/run.py --config-name pqn_cartpole

# Or specify components individually
python scripts/run.py agent=pqn_agent env=cartpole run=pqn_cartpole
```

### Configuration

The key parameters for PQN are:

**Agent Parameters** (`configs/agent/pqn_agent.yaml`):
- `hidden_size`: 128 (network hidden layer size)
- `num_layers`: 2 (number of hidden layers)
- `learning_rate`: 2.5e-4
- `gamma`: 0.99 (discount factor)
- `lambda_`: 0.65 (lambda for lambda-returns, 0=1-step TD, 1=Monte Carlo)
- `num_epochs`: 4 (training epochs per rollout)
- `num_minibatches`: 4 (minibatches per epoch)

**Run Parameters** (`configs/run/pqn_cartpole.yaml`):
- `num_envs`: 8 (parallel environments)
- `rollout_steps`: 32 (steps to collect before training)
- `batch_size`: 256 (should equal num_envs × rollout_steps)

## Key Differences from DQN

| Feature | DQN | PQN |
|---------|-----|-----|
| Training Mode | Off-policy | On-policy |
| Replay Buffer | Required | Not used |
| Target Network | Required | Not used |
| Normalization | None | LayerNorm |
| Target Computation | 1-step TD | Lambda-returns |
| Update Frequency | Every step | After rollout collection |
| Training | 1 update per step | Multi-epoch on rollout |

## Architecture Details

### Lambda-Returns Computation

PQN uses lambda-returns (similar to GAE) computed backward through the trajectory:

```python
# Recursive computation:
bootstrap = reward + gamma * (1 - done) * max_Q(next_state)
delta = lambda_return - max_Q(next_state)
lambda_return = bootstrap + gamma * lambda * (1 - done) * delta
```

This mixes bootstrapping (from Q-values) with Monte Carlo returns based on the `lambda_` parameter.

### Multi-Epoch Training

For each collected rollout:
1. Collect `rollout_steps` transitions from `num_envs` environments
2. Compute lambda-returns for all transitions
3. For `num_epochs` epochs:
   - Shuffle the rollout data
   - Split into `num_minibatches` minibatches
   - Train on each minibatch

This ensures better sample efficiency from the collected data.

## Implementation Notes

### JAX Compatibility

- Uses `jax.lax.dynamic_slice` for dynamic indexing inside jitted functions
- Lambda-returns computed using `jax.lax.scan` for efficiency
- All operations are pure functional and JIT-compatible

### Platform Integration

The runner now supports two training modes:
- **Step-by-step mode** (DQN): Uses replay buffer, single-step updates
- **Rollout mode** (PQN): Collects rollouts, no replay buffer, multi-epoch training

The mode is automatically detected based on the `rollout_steps` parameter in the run config.

## Testing

The implementation has been tested with:
- CartPole environment (discrete action space)
- 5000 timesteps quick test (successful)
- All existing pytest tests pass (except 1 pre-existing failure)

## References

- [PureJaxQL Implementation](https://github.com/mttga/purejaxql)
- Original PQN paper concepts
- Lambda-returns/GAE methodology
