"""Training loop for RL agents - optimized for GPU scaling."""
from functools import partial
from typing import Any, Dict, List, Tuple

import chex
import jax
import jax.numpy as jnp

from aion.agents.pqn_agent import PQNAgent, PQNTrainState
from aion.envs.toy_env_v1 import EnvParams
from aion.platform.interaction import run_episodes_parallel


@partial(jax.jit, static_argnums=(2, 4, 5, 6, 7))
def _train_step(
    key: chex.PRNGKey,
    train_state: PQNTrainState,
    agent: PQNAgent,
    env_params: EnvParams,
    num_envs: int,
    max_steps_in_episode: int,
    batch_size: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
) -> Tuple[PQNTrainState, Dict[str, chex.Scalar]]:
    """A single training step - JIT compiled for maximum performance.

    This function is optimized for GPU execution with large batch sizes
    and many parallel environments.
    """
    # --- COLLECT EXPERIENCE ---
    key, rollout_key = jax.random.split(key)

    # Pure function for action selection (JIT-compatible)
    def select_action_fn(
        key: chex.PRNGKey, observation: chex.Array, train_state: PQNTrainState, step: int
    ) -> Tuple[chex.Array, PQNTrainState]:
        epsilon = jnp.interp(
            step,
            jnp.array([0, epsilon_decay_steps]),
            jnp.array([epsilon_start, epsilon_end]),
        )

        def explore() -> chex.Array:
            return jax.random.randint(key, (), 0, agent.action_dim)

        def exploit() -> chex.Array:
            q_values = train_state.apply_fn({"params": train_state.params}, observation)
            return jnp.argmax(q_values, axis=-1)

        use_random = jax.random.uniform(key) < epsilon
        action = jax.lax.cond(use_random, explore, exploit)
        return action, train_state

    # Run parallel environments
    trajectories = run_episodes_parallel(
        rollout_key,
        select_action_fn,
        train_state,
        env_params,
        num_envs,
        max_steps_in_episode,
    )

    # Efficiently flatten and sample trajectories
    obs, actions, rewards, next_obs, dones = trajectories

    # Reshape from (max_steps, num_envs, ...) to (max_steps * num_envs, ...)
    batch_dims = obs.shape[:2]  # (max_steps, num_envs)
    total_transitions = batch_dims[0] * batch_dims[1]

    flat_obs = obs.reshape((total_transitions,) + obs.shape[2:])
    flat_actions = actions.reshape((total_transitions,) + actions.shape[2:])
    flat_rewards = rewards.reshape((total_transitions,) + rewards.shape[2:])
    flat_next_obs = next_obs.reshape((total_transitions,) + next_obs.shape[2:])
    flat_dones = dones.reshape((total_transitions,) + dones.shape[2:])

    # Efficient batch sampling
    key, sample_key = jax.random.split(key)
    # batch_size_actual = jnp.minimum(batch_size, total_transitions)
    indices = jax.random.choice(sample_key, total_transitions, (batch_size,), replace=False)

    batch = (
        flat_obs[indices],
        flat_actions[indices],
        flat_rewards[indices],
        flat_next_obs[indices],
        flat_dones[indices],
    )

    # --- UPDATE AGENT ---
    train_state, loss = agent.update(train_state, batch)

    # --- COMPUTE METRICS ---
    episode_returns = rewards.sum(axis=0)  # Sum over time steps
    metrics = {
        "loss": loss,
        "mean_episode_return": episode_returns.mean(),
        "std_episode_return": episode_returns.std(),
        "min_episode_return": episode_returns.min(),
        "max_episode_return": episode_returns.max(),
        "mean_episode_length": max_steps_in_episode,  # All episodes run to completion
        "total_transitions": total_transitions,
    }

    return train_state, metrics


def train(
    key: chex.PRNGKey,
    agent: PQNAgent,
    env_params: EnvParams,
    num_episodes: int,
    num_envs: int,
    max_steps_in_episode: int,
    batch_size: int,
    log_frequency: int = 10,
) -> Tuple[PQNTrainState, List[Dict[str, Any]]]:
    """
    Main training loop optimized for GPU scaling.

    This function supports efficient training with large numbers of parallel
    environments and can scale to thousands of environments on modern GPUs.

    Args:
        key: JAX random key
        agent: The agent to train
        env_params: Environment parameters
        num_episodes: Total number of training episodes
        num_envs: Number of parallel environments (can be very large for GPU)
        max_steps_in_episode: Maximum steps per episode
        batch_size: Batch size for agent updates
        log_frequency: How often to log metrics

    Returns:
        Tuple of (final_train_state, metrics_history)
    """
    # --- INITIALIZATION ---
    key, agent_init_key = jax.random.split(key, 2)
    sample_obs = jnp.array([5.0, 5.0])  # [x, x_target]
    train_state = agent.init(agent_init_key, sample_obs)

    # Create JIT-compiled training step
    compiled_train_step = partial(
        _train_step,
        agent=agent,
        env_params=env_params,
        num_envs=num_envs,
        max_steps_in_episode=max_steps_in_episode,
        batch_size=batch_size,
        epsilon_start=agent.epsilon_start,
        epsilon_end=agent.epsilon_end,
        epsilon_decay_steps=agent.epsilon_decay_steps,
    )

    # --- TRAINING LOOP ---
    print(f"--- Training with {num_envs} parallel environments ---")
    print(f"Total transitions per episode: {num_envs * max_steps_in_episode}")

    metrics_history = []

    for episode in range(num_episodes):
        key, train_key = jax.random.split(key)

        # Run training step
        train_state, metrics = compiled_train_step(train_key, train_state)

        metrics_history.append(metrics)

        # Log progress
        if episode % log_frequency == 0:
            loss = float(metrics["loss"])
            mean_return = float(metrics["mean_episode_return"])
            std_return = float(metrics["std_episode_return"])

            print(
                f"Episode {episode:6d} | "
                f"Loss: {loss:8.4f} | "
                f"Return: {mean_return:8.2f} ± {std_return:6.2f} | "
                f"Step: {int(train_state.step)}"
            )

    print("--- Training Complete ---")
    return train_state, metrics_history
