"""Parallelized Q-Network (PQN) with LayerNorm agent implementation using JAX and Flax.

PQN is an on-policy value-based RL algorithm designed for massively parallel training.

Features:

- Lambda-returns (GAE-style) instead of 1-step TD targets
- LayerNorm for stability (no target network needed)
- Multi-epoch training on collected rollouts
- Epsilon-greedy exploration with linear decay
- Gradient clipping for stable training

Reference: `PureJaxQL <https://github.com/mttga/purejaxql>`_
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn, struct
from flax.training.train_state import TrainState
from jax import Array

from myriad.core.spaces import Discrete, Space
from myriad.core.types import Observation, PRNGKey, Transition
from myriad.utils.observations import to_array

from ..agent import Agent


class QNetwork(nn.Module):
    """MLP Q-network with LayerNorm for discrete action spaces."""

    action_dim: int
    hidden_size: int = 128
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Forward pass to compute Q-values for all actions.

        Args:
            x: Observation array of shape (obs_dim,) or (batch_size, obs_dim)

        Returns:
            Q-values of shape (action_dim,) or (batch_size, action_dim)
        """
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


@struct.dataclass
class AgentParams:
    """Static parameters for the PQN agent.

    Attributes:
        action_space: Action space (must be Discrete).
        learning_rate: Learning rate for Adam optimizer.
        gamma: Discount factor for future rewards.
        lambda_: Lambda parameter for lambda-returns (0.0 = 1-step TD, 1.0 = Monte Carlo).
        reward_scale: Internal scaling factor applied to rewards before computing returns.
        epsilon_start: Initial exploration rate.
        epsilon_end: Final exploration rate after decay.
        epsilon_decay_steps: Number of steps to decay epsilon from start to end.
        max_grad_norm: Maximum gradient norm for clipping.
        num_epochs: Number of training epochs per rollout batch.
        num_minibatches: Number of minibatches per epoch.
        hidden_size: Hidden layer size for Q-network.
        num_layers: Number of hidden layers in Q-network.
    """

    action_space: Space
    learning_rate: float
    gamma: float
    lambda_: float
    reward_scale: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    max_grad_norm: float
    num_epochs: int
    num_minibatches: int
    hidden_size: int
    num_layers: int


@struct.dataclass
class AgentState:
    """State of the PQN agent.

    Attributes:
        train_state: Flax TrainState containing network params and optimizer state.
        global_step: Number of update steps taken (used for epsilon decay).
    """

    train_state: TrainState
    global_step: Array


def _init(
    key: PRNGKey,
    sample_obs: Observation,
    params: AgentParams,
) -> AgentState:
    """Initialize the PQN agent.

    Args:
        key: Random key for initialization
        sample_obs: Sample observation to infer network architecture
        params: Agent hyperparameters

    Returns:
        Initial agent state containing network and optimizer
    """
    # Convert observation to array if needed
    sample_obs_array = to_array(sample_obs)

    if not isinstance(params.action_space, Discrete):
        raise ValueError("PQN only supports Discrete action spaces")
    action_dim = params.action_space.n

    # Initialize Q-network
    q_network = QNetwork(
        action_dim=action_dim,
        hidden_size=params.hidden_size,
        num_layers=params.num_layers,
    )
    q_params = q_network.init(key, sample_obs_array)

    # Create optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(params.max_grad_norm),
        optax.radam(params.learning_rate),
    )

    train_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_params,
        tx=optimizer,
    )

    return AgentState(
        train_state=train_state,
        global_step=jnp.array(0, dtype=jnp.int32),
    )


def _select_action(
    key: PRNGKey,
    obs: Observation,
    state: AgentState,
    params: AgentParams,
    deterministic: bool = False,
) -> Tuple[Array, AgentState]:
    """Select action using epsilon-greedy policy.

    Args:
        key: Random key for exploration
        obs: Current observation
        state: Current agent state
        params: Agent hyperparameters
        deterministic: If True, use greedy policy (epsilon=0). Default False.

    Returns:
        Tuple of (action, unchanged agent_state)
    """
    # Convert observation to array if needed
    obs_array = to_array(obs)

    # Calculate current epsilon with linear decay (or use 0 if deterministic)
    epsilon_decayed = jnp.maximum(
        params.epsilon_end,
        params.epsilon_start
        - (params.epsilon_start - params.epsilon_end) * state.global_step / params.epsilon_decay_steps,
    )
    epsilon = jax.lax.select(deterministic, jnp.array(0.0), epsilon_decayed)

    # Get Q-values
    q_values = state.train_state.apply_fn(state.train_state.params, obs_array)

    # Epsilon-greedy action selection
    key_explore, key_action = jax.random.split(key)
    explore = jax.random.uniform(key_explore) < epsilon

    # Greedy action (argmax Q)
    greedy_action = jnp.argmax(q_values)

    # Random action
    random_action = params.action_space.sample(key_action)

    # Select based on epsilon
    action = jax.lax.select(explore, random_action, greedy_action)

    return action, state


def _compute_lambda_returns(
    rewards: Array,
    dones: Array,
    next_q_max: Array,
    gamma: float,
    lambda_: float,
) -> Array:
    """Compute lambda-returns backward through trajectory.

    Uses the GAE-style recursive formula (matching PureJaxQL):
        target_bootstrap = r_t + gamma * (1 - done_t) * V(s_{t+1})
        delta = G_{t+1} - V(s_{t+1})
        G_t = target_bootstrap + gamma * lambda * delta
        G_t = (1 - done_t) * G_t + done_t * r_t  (final masking)

    where V(s_{t+1}) = max_a Q(s_{t+1}, a) for Q-learning.

    Args:
        rewards: Rewards array of shape (num_steps,)
        dones: Done flags array of shape (num_steps,)
        next_q_max: Max Q-values for next states of shape (num_steps,)
            next_q_max[t] = V(s_{t+1})
        gamma: Discount factor
        lambda_: Lambda parameter for mixing TD targets

    Returns:
        Lambda-returns of shape (num_steps,)
    """

    def scan_fn(G_next, transition):
        reward, done, v_next = transition  # v_next = V(s_{t+1}) for this step
        target_bootstrap = reward + gamma * (1.0 - done) * v_next
        delta = G_next - v_next
        G = target_bootstrap + gamma * lambda_ * delta
        G = (1.0 - done) * G + done * reward  # mask terminal steps
        return G, G

    # Bootstrap final step correctly: G_{T-1} = r_{T-1} + gamma * (1-done) * V(s_T)
    G_last = rewards[-1] + gamma * (1.0 - dones[-1]) * next_q_max[-1]
    G_last = (1.0 - dones[-1]) * G_last + dones[-1] * rewards[-1]

    # Scan backward through trajectory (excluding last step)
    # next_q_max[t] = V(s_{t+1}), so each step reads the correct v_next
    transitions = (rewards[:-1], dones[:-1], next_q_max[:-1])
    _, Gs = jax.lax.scan(scan_fn, G_last, transitions, reverse=True)

    return jnp.concatenate([Gs, jnp.array([G_last])])


def _update(
    key: PRNGKey,
    agent_state: AgentState,
    batch: Transition,
    params: AgentParams,
) -> Tuple[AgentState, dict]:
    """Update the agent using a batch of transitions with lambda-returns.

    This function expects a batch of transitions from a rollout and performs
    multi-epoch training with minibatch shuffling.

    Args:
        key: Random key for shuffling
        agent_state: Current agent state
        batch: Batch of transitions from rollout (NOT from replay buffer)
        params: Agent hyperparameters

    Returns:
        Tuple of (new agent_state, metrics dict)
    """
    # batch has shape (T, E, ...) from make_chunked_collector.
    # Compute per-environment lambda-returns on the natural trajectory shape.
    rewards = batch.reward * params.reward_scale  # (T, E)
    dones = batch.done.astype(jnp.float32)  # (T, E)

    # next_obs shape (T, E, obs_dim) — nn.Dense broadcasts over leading dims
    next_q_values = agent_state.train_state.apply_fn(agent_state.train_state.params, batch.next_obs)
    next_q_max = jnp.max(next_q_values, axis=-1)  # (T, E)

    # _compute_lambda_returns uses purely element-wise ops and a scalar carry, so
    # passing (T, E) arrays works directly — the scan carry is (E,) at each step.
    lambda_returns = _compute_lambda_returns(rewards, dones, next_q_max, params.gamma, params.lambda_)
    lambda_returns = jax.lax.stop_gradient(lambda_returns)  # (T, E)

    # Flatten for minibatch training
    flat = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
    lambda_returns_flat = lambda_returns.reshape(-1)  # (T*E,)
    batch_size = lambda_returns_flat.shape[0]
    minibatch_size = batch_size // params.num_minibatches

    def train_epoch(carry, _):
        """Train for one epoch with shuffled minibatches."""
        train_state, epoch_key = carry

        # Shuffle indices
        perm_key, next_key = jax.random.split(epoch_key)
        perm = jax.random.permutation(perm_key, batch_size)

        def train_minibatch(train_state, minibatch_idx):
            """Train on one minibatch."""
            # Get minibatch indices using dynamic_slice
            start_idx = minibatch_idx * minibatch_size
            mb_indices = jax.lax.dynamic_slice(perm, (start_idx,), (minibatch_size,))

            # Get minibatch data using gather
            mb_obs = flat.obs[mb_indices]
            mb_actions = flat.action[mb_indices]
            mb_targets = lambda_returns_flat[mb_indices]

            def loss_fn(q_params):
                """Compute TD loss for Q-network."""
                # Current Q-values: Q(s, a)
                q_values = train_state.apply_fn(q_params, mb_obs)

                # Select Q-values for actions taken
                actions_expanded = jnp.asarray(mb_actions)[:, None]
                q_values_selected = jnp.take_along_axis(q_values, actions_expanded, axis=1).squeeze(1)

                # MSE loss against lambda-returns
                td_error = q_values_selected - mb_targets
                loss = 0.5 * jnp.mean(td_error**2)

                return loss, {
                    "td_error_mean": jnp.mean(jnp.abs(td_error)),
                    "q_value_mean": jnp.mean(q_values_selected),
                }

            # Compute gradients and update
            (loss, aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
            new_train_state = train_state.apply_gradients(grads=grads)

            return new_train_state, (loss, aux_metrics)

        # Train on all minibatches
        new_train_state, (losses, aux_metrics) = jax.lax.scan(
            train_minibatch, train_state, jnp.arange(params.num_minibatches)
        )

        # Average metrics across minibatches
        avg_loss = jnp.mean(losses)
        avg_td_error = jnp.mean(jax.vmap(lambda x: x["td_error_mean"])(aux_metrics))
        avg_q_value = jnp.mean(jax.vmap(lambda x: x["q_value_mean"])(aux_metrics))

        return (new_train_state, next_key), (avg_loss, avg_td_error, avg_q_value)

    # Train for multiple epochs
    (new_train_state, _), (epoch_losses, epoch_td_errors, epoch_q_values) = jax.lax.scan(
        train_epoch, (agent_state.train_state, key), jnp.arange(params.num_epochs)
    )

    # Create new agent state
    new_agent_state = AgentState(
        train_state=new_train_state,
        global_step=agent_state.global_step + 1,
    )

    # Return metrics (average over epochs)
    metrics = {
        "loss": jnp.mean(epoch_losses),
        "td_error": jnp.mean(epoch_td_errors),
        "q_value": jnp.mean(epoch_q_values),
        "lambda_return_mean": jnp.mean(lambda_returns_flat),
    }

    return new_agent_state, metrics


def make_agent(
    action_space: Space,
    learning_rate: float = 2.5e-4,
    reward_scale: float = 1.0,
    gamma: float = 0.99,
    lambda_: float = 0.65,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 50000,
    max_grad_norm: float = 0.5,
    num_epochs: int = 4,
    num_minibatches: int = 4,
    hidden_size: int = 128,
    num_layers: int = 2,
) -> Agent:
    """Factory function to create a PQN agent.

    Args:
        action_space: Action space (must be Discrete)
        learning_rate: Learning rate for Adam optimizer
        reward_scale: Internal scaling factor for the rewards
        gamma: Discount factor
        lambda_: Lambda parameter for lambda-returns (0.0 = 1-step TD, 1.0 = Monte Carlo)
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay_steps: Steps to decay epsilon from start to end
        max_grad_norm: Maximum gradient norm for clipping
        num_epochs: Number of training epochs per rollout
        num_minibatches: Number of minibatches per epoch
        hidden_size: Hidden layer size for Q-network
        num_layers: Number of hidden layers in Q-network

    Returns:
        Agent instance with PQN implementation
    """
    params = AgentParams(
        action_space=action_space,
        learning_rate=learning_rate,
        reward_scale=reward_scale,
        gamma=gamma,
        lambda_=lambda_,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        max_grad_norm=max_grad_norm,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    return Agent(
        params=params,
        init=_init,
        select_action=_select_action,  # type: ignore[arg-type]
        update=_update,  # type: ignore[arg-type]
    )
