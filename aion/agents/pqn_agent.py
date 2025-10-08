"""A Parallelized Q-Network (PQN) agent."""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from typing import Tuple
from flax.struct import dataclass, field
import chex


class QNetwork(nn.Module):
    """Optimized Q-Network for the PQN agent with better GPU utilization."""

    action_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "relu"
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        """Forward pass through the network."""
        # Use wider layers for better GPU utilization
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            if self.activation == "relu":
                x = nn.relu(x)
            elif self.activation == "swish":
                x = nn.swish(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class PQNTrainState(TrainState):
    """Custom TrainState for PQN to include target network parameters."""

    target_params: chex.ArrayTree


@dataclass
class PQNAgent:
    """A Parallelized Q-Network (PQN) agent optimized for GPU scaling."""

    q_network: QNetwork = field(pytree_node=False)
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    target_network_frequency: int
    tau: float
    action_dim: int = 2

    def init(self, key: chex.PRNGKey, sample_obs: chex.Array) -> PQNTrainState:
        """Initializes the agent's training state."""
        params = self.q_network.init(key, sample_obs)["params"]
        tx = optax.adam(self.learning_rate)
        return PQNTrainState.create(
            apply_fn=self.q_network.apply,
            params=params,
            target_params=params,
            tx=tx,
        )

    def select_action(
        self,
        key: chex.PRNGKey,
        observation: chex.Array,
        train_state: PQNTrainState,
        step: chex.Scalar,
    ) -> Tuple[chex.Array, PQNTrainState]:
        """Selects an action using an epsilon-greedy policy."""
        epsilon = jnp.interp(
            step,
            jnp.array([0, self.epsilon_decay_steps]),
            jnp.array([self.epsilon_start, self.epsilon_end]),
        )

        def explore() -> chex.Array:
            """Returns a random action."""
            return jax.random.randint(key, (), 0, self.action_dim)

        def exploit() -> chex.Array:
            """Returns the action with the highest Q-value."""
            q_values = train_state.apply_fn({"params": train_state.params}, observation)
            return jnp.argmax(q_values, axis=-1)

        use_random = jax.random.uniform(key) < epsilon
        action = jax.lax.cond(use_random, explore, exploit)

        return action, train_state

    def _update_target_network(self, train_state: PQNTrainState) -> PQNTrainState:
        """Updates the target network parameters."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.tau + tp * (1 - self.tau),
            train_state.params,
            train_state.target_params,
        )
        return train_state.replace(target_params=new_target_params)

    def update(
        self,
        train_state: PQNTrainState,
        transitions: Tuple[
            chex.Array, chex.Array, chex.Array, chex.Array, chex.Array
        ],
    ) -> Tuple[PQNTrainState, chex.Scalar]:
        """Performs a Q-learning update with gradient clipping."""
        obs, actions, rewards, next_obs, dones = transitions

        # --- Compute Target Q-values ---
        next_q_values = train_state.apply_fn(
            {"params": train_state.target_params}, next_obs
        )
        next_q_values = jnp.max(next_q_values, axis=-1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        # Stop gradients on targets for stability
        target_q_values = jax.lax.stop_gradient(target_q_values)

        # --- Compute Loss and Gradients ---
        def loss_fn(params) -> chex.Scalar:
            q_values = train_state.apply_fn({"params": params}, obs)
            action_q_values = jnp.take_along_axis(
                q_values, actions[..., None], axis=-1
            ).squeeze(-1)
            loss = optax.huber_loss(action_q_values, target_q_values).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
        
        # Clip gradients for stability
        grads = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -1.0, 1.0), grads
        )

        # --- Update Train State ---
        train_state = train_state.apply_gradients(grads=grads)

        # --- Update Target Network ---
        train_state = jax.lax.cond(
            train_state.step % self.target_network_frequency == 0,
            self._update_target_network,
            lambda ts: ts,
            train_state,
        )

        return train_state, loss
