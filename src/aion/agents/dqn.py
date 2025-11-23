"""Deep Q-Network (DQN) agent implementation using JAX and Flax.

A classic value-based RL algorithm that learns to estimate Q-values for state-action pairs
and uses epsilon-greedy exploration.
"""

from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn, struct
from flax.training.train_state import TrainState

from aion.core.spaces import Discrete, Space
from aion.core.types import Transition

from .agent import Agent


class QNetwork(nn.Module):
    """Simple MLP Q-network for discrete action spaces."""

    action_dim: int
    hidden_dims: tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """Forward pass to compute Q-values for all actions.

        Args:
            x: Observation array of shape (obs_dim,) or (batch_size, obs_dim)

        Returns:
            Q-values of shape (action_dim,) or (batch_size, action_dim)
        """
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


@struct.dataclass
class AgentParams:
    """Static parameters for the DQN agent."""

    action_space: Space
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    target_network_frequency: int
    tau: float  # For soft updates (1.0 = hard update)


@struct.dataclass
class AgentState:
    """State of the DQN agent."""

    train_state: TrainState
    target_params: Any  # Parameters of the target network
    global_step: chex.Array


def _init(
    key: chex.PRNGKey,
    sample_obs: chex.Array,
    params: AgentParams,
) -> AgentState:
    """Initialize the DQN agent.

    Args:
        key: Random key for initialization
        sample_obs: Sample observation to infer network architecture
        params: Agent hyperparameters

    Returns:
        Initial agent state containing networks and optimizer
    """
    if not isinstance(params.action_space, Discrete):
        raise ValueError("DQN only supports Discrete action spaces")

    action_dim = params.action_space.n

    # Initialize Q-network
    q_network = QNetwork(action_dim=action_dim)
    q_params = q_network.init(key, sample_obs)

    # Create training state with optimizer
    import optax

    optimizer = optax.adam(params.learning_rate)
    train_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_params,
        tx=optimizer,
    )

    # Initialize target network with same parameters
    target_params = jax.tree_util.tree_map(lambda x: x.copy(), q_params)

    return AgentState(
        train_state=train_state,
        target_params=target_params,
        global_step=jnp.array(0, dtype=jnp.int32),
    )


def _select_action(
    key: chex.PRNGKey,
    obs: chex.Array,
    agent_state: AgentState,
    params: AgentParams,
) -> Tuple[chex.Array, AgentState]:
    """Select action using epsilon-greedy policy.

    Args:
        key: Random key for exploration
        obs: Current observation
        agent_state: Current agent state
        params: Agent hyperparameters

    Returns:
        Tuple of (action, unchanged agent_state)
    """
    # Calculate current epsilon with linear decay
    epsilon = jnp.maximum(
        params.epsilon_end,
        params.epsilon_start
        - (params.epsilon_start - params.epsilon_end) * agent_state.global_step / params.epsilon_decay_steps,
    )

    # Get Q-values
    q_values = agent_state.train_state.apply_fn(agent_state.train_state.params, obs)

    # Epsilon-greedy action selection
    key_explore, key_action = jax.random.split(key)
    explore = jax.random.uniform(key_explore) < epsilon

    # Greedy action (argmax Q)
    greedy_action = jnp.argmax(q_values)

    # Random action
    random_action = params.action_space.sample(key_action)

    # Select based on epsilon
    action = jax.lax.select(explore, random_action, greedy_action)

    return action, agent_state


def _update(
    key: chex.PRNGKey,  # noqa: ARG001
    agent_state: AgentState,
    batch: Transition,
    params: AgentParams,
) -> Tuple[AgentState, dict]:
    """Update the agent using a batch of transitions.

    Args:
        key: Random key (unused in DQN)
        agent_state: Current agent state
        batch: Batch of transitions from replay buffer
        params: Agent hyperparameters

    Returns:
        Tuple of (new agent_state, metrics dict)
    """

    def loss_fn(q_params):
        """Compute TD loss for Q-network."""
        # Current Q-values: Q(s, a)
        q_values = agent_state.train_state.apply_fn(q_params, batch.obs)
        # Select Q-values for actions taken
        actions_expanded = jnp.asarray(batch.action)[:, None]
        q_values_selected = jnp.take_along_axis(q_values, actions_expanded, axis=1).squeeze(1)

        # Target Q-values: r + gamma * max_a' Q_target(s', a')
        next_q_values = agent_state.train_state.apply_fn(agent_state.target_params, batch.next_obs)
        next_q_max = jnp.max(next_q_values, axis=1)

        # TD target (no gradient through target)
        rewards = jnp.asarray(batch.reward)
        dones = jnp.asarray(batch.done, dtype=jnp.float32)
        td_target = rewards + params.gamma * next_q_max * (1.0 - dones)
        td_target = jax.lax.stop_gradient(td_target)

        # MSE loss
        td_error = q_values_selected - td_target
        loss = jnp.mean(td_error**2)

        return loss, {"td_error_mean": jnp.mean(jnp.abs(td_error)), "q_value_mean": jnp.mean(q_values_selected)}

    # Compute gradients and update
    (loss, aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(agent_state.train_state.params)
    new_train_state = agent_state.train_state.apply_gradients(grads=grads)

    # Update target network periodically
    should_update_target = (agent_state.global_step.astype(jnp.int32) % params.target_network_frequency) == 0

    if params.tau == 1.0:
        # Hard update
        new_target_params = jax.lax.cond(
            should_update_target,
            lambda: jax.tree_util.tree_map(lambda x: x.copy(), new_train_state.params),
            lambda: agent_state.target_params,
        )
    else:
        # Soft update: target = tau * online + (1 - tau) * target
        def soft_update():
            return jax.tree_util.tree_map(
                lambda online, target: params.tau * online + (1.0 - params.tau) * target,
                new_train_state.params,
                agent_state.target_params,
            )

        new_target_params = jax.lax.cond(
            should_update_target,
            soft_update,
            lambda: agent_state.target_params,
        )

    new_agent_state = AgentState(
        train_state=new_train_state,
        target_params=new_target_params,
        global_step=agent_state.global_step + 1,
    )

    metrics = {
        "loss": loss,
        "td_error": aux_metrics["td_error_mean"],
        "q_value": aux_metrics["q_value_mean"],
    }

    return new_agent_state, metrics


def make_agent(
    action_space: Space,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 10000,
    target_network_frequency: int = 500,
    tau: float = 1.0,
) -> Agent:
    """Factory function to create a DQN agent.

    Args:
        action_space: Action space (must be Discrete)
        learning_rate: Learning rate for Adam optimizer
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay_steps: Steps to decay epsilon from start to end
        target_network_frequency: Steps between target network updates
        tau: Soft update coefficient (1.0 = hard update)

    Returns:
        Agent instance with DQN implementation
    """
    params = AgentParams(
        action_space=action_space,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        target_network_frequency=target_network_frequency,
        tau=tau,
    )

    return Agent(
        params=params,
        init=_init,
        select_action=_select_action,
        update=_update,
    )
