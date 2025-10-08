"""A fully functional implementation of a Parallelized Q-Network (PQN) agent."""

from typing import Any, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass
from flax.training.train_state import TrainState

from .base import Agent, AgentParams, AgentState

# --- Network Definition ---


class QNetwork(nn.Module):
    """A simple MLP for Q-value approximation."""

    action_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """Forward pass through the network."""
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


# --- Agent Components ---


class PQNTrainState(TrainState):
    """Custom TrainState for PQN to include target network parameters."""

    target_params: chex.ArrayTree


@dataclass
class PQNAgentParams(AgentParams):
    """Static parameters and configuration for the PQN agent."""

    q_network: QNetwork
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    target_network_frequency: int
    tau: float


# --- Agent Pure Functions ---


def _init(key: chex.PRNGKey, sample_obs: chex.Array, params: PQNAgentParams) -> PQNTrainState:
    """Initializes the agent's training state."""
    net_params = params.q_network.init(key, sample_obs)["params"]
    tx = optax.adam(params.learning_rate)
    return PQNTrainState.create(
        apply_fn=params.q_network.apply,
        params=net_params,
        target_params=net_params,
        tx=tx,
    )


def _select_action(
    key: chex.PRNGKey,
    observation: chex.Array,
    agent_state: PQNTrainState,
    params: PQNAgentParams,
) -> Tuple[chex.Array, AgentState]:
    """Selects an action using an epsilon-greedy policy."""
    epsilon = jnp.interp(
        agent_state.step,
        jnp.array([0, params.epsilon_decay_steps]),
        jnp.array([params.epsilon_start, params.epsilon_end]),
    )

    def explore() -> chex.Array:
        """Returns a random action."""
        return jax.random.randint(key, (), 0, params.q_network.action_dim)

    def exploit() -> chex.Array:
        """Returns the action with the highest Q-value."""
        q_values = agent_state.apply_fn({"params": agent_state.params}, observation)
        return jnp.argmax(q_values, axis=-1)

    use_random = jax.random.uniform(key) < epsilon
    action = jax.lax.cond(use_random, explore, exploit)

    return action, agent_state


def _update_target_network(train_state: PQNTrainState, tau: float) -> PQNTrainState:
    """Helper function to update the target network parameters."""
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau),
        train_state.params,
        train_state.target_params,
    )
    return train_state.replace(target_params=new_target_params)


def _update(
    key: chex.PRNGKey,
    agent_state: PQNTrainState,
    transitions: Any,
    params: PQNAgentParams,
) -> Tuple[PQNTrainState, dict]:
    """Performs a Q-learning update and returns metrics."""
    _ = key  # Key is not used in this deterministic update, but is kept for API consistency.
    obs, actions, rewards, next_obs, dones = transitions

    # --- Compute Target Q-values ---
    next_q_values = agent_state.apply_fn({"params": agent_state.target_params}, next_obs)
    next_q_values = jnp.max(next_q_values, axis=-1)
    target_q_values = rewards + (1 - dones) * params.gamma * next_q_values
    target_q_values = jax.lax.stop_gradient(target_q_values)

    # --- Compute Loss and Gradients ---
    def loss_fn(net_params) -> Tuple[chex.Scalar, dict]:
        q_values = agent_state.apply_fn({"params": net_params}, obs)
        action_q_values = jnp.take_along_axis(q_values, actions[..., None], axis=-1).squeeze(-1)
        loss = optax.huber_loss(action_q_values, target_q_values).mean()
        metrics = {"loss": loss}
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(agent_state.params)

    # --- Update Train State ---
    agent_state = agent_state.apply_gradients(grads=grads)

    # --- Update Target Network ---
    agent_state = jax.lax.cond(
        agent_state.step % params.target_network_frequency == 0,
        lambda ts: _update_target_network(ts, params.tau),
        lambda ts: ts,
        agent_state,
    )

    return agent_state, metrics


# --- Factory Function ---


def make_agent(
    action_dim: int,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 10000,
    target_network_frequency: int = 100,
    tau: float = 0.005,
) -> Agent:
    """Factory function to create an instance of the PQN agent."""

    q_network = QNetwork(action_dim=action_dim)

    default_params = PQNAgentParams(
        q_network=q_network,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        target_network_frequency=target_network_frequency,
        tau=tau,
    )

    return Agent(
        init=_init,
        select_action=_select_action,
        update=_update,
        default_params=default_params,
    )
