"""Script to train the PQN agent on the toy problem."""

import jax
import jax.numpy as jnp
import click
from pathlib import Path
from aion.agents.pqn_agent import PQNAgent, QNetwork
from aion.envs.toy_problem import (
    EnvParams, get_action_space_size, create_sine_target, create_constant_target
)
from aion.platform.training import train
from aion.utils.plotting.episodes import plot_episodes
from aion.utils.plotting.training import plot_training_metrics
from aion.platform.interaction import run_episodes_parallel


@click.command()
@click.option("--num-episodes", default=500, help="Number of episodes to train for.")
@click.option("--num-envs", default=64, help="Number of parallel environments.")
@click.option("--max-steps", default=100, help="Maximum steps per episode.")
@click.option("--batch-size", default=256, help="Batch size for training.")
@click.option("--seed", default=42, help="Random seed.")
@click.option("--target-type", default="sine", help="Target type: constant, sine, linear, step")
def main(num_episodes, num_envs, max_steps, batch_size, seed, target_type):
    """Main function to run the training."""
    # --- INITIALIZATION ---
    key = jax.random.PRNGKey(seed)
    key, train_key, eval_key = jax.random.split(key, 3)

    # Create target trajectory based on type
    if target_type == "constant":
        x_target = create_constant_target(5.0, max_steps)
    elif target_type == "sine":
        x_target = create_sine_target(amplitude=3.0, frequency=2.0, offset=10.0, length=max_steps)
    elif target_type == "linear":
        x_target = jnp.linspace(2.0, 18.0, max_steps)
    elif target_type == "step":
        steps_per_level = max_steps // 4
        x_target = jnp.concatenate([
            jnp.full((steps_per_level,), 5.0),
            jnp.full((steps_per_level,), 15.0),
            jnp.full((steps_per_level,), 8.0),
            jnp.full((max_steps - 3 * steps_per_level,), 12.0)
        ])
    else:
        raise ValueError(f"Unknown target type: {target_type}")

    # Create environment parameters
    env_params = EnvParams(
        a=1.0,
        b=1.0,
        min_x=0.0,
        max_x=20.0,
        max_steps=max_steps,
        x_target=x_target
    )

    # Create agent with optimized network architecture
    agent = PQNAgent(
        q_network=QNetwork(
            action_dim=get_action_space_size(),
            hidden_dims=(256, 256),  # Larger for better GPU utilization
            activation="relu",
            use_layer_norm=True
        ),
        learning_rate=2.5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=num_episodes * max_steps // 2,
        target_network_frequency=200,
        tau=1.0,
        action_dim=get_action_space_size(),
    )

    # --- TRAINING ---
    trained_state, metrics_history = train(
        key=train_key,
        agent=agent,
        env_params=env_params,
        num_episodes=num_episodes,
        num_envs=num_envs,
        max_steps_in_episode=max_steps,
        batch_size=batch_size,
    )

    # --- PLOT TRAINING METRICS ---
    metrics_path = Path("data/temp/pqn_training_metrics.html")
    plot_training_metrics(metrics_history, filename=metrics_path)
    print(f"Training metrics plot saved to {metrics_path}")

    # --- EVALUATION & PLOTTING ---
    print("\n--- Evaluating Trained Agent ---")
    eval_episodes = 5
    trajectories = run_episodes_parallel(
        key=eval_key,
        agent_select_action=agent.select_action,
        train_state=trained_state,
        env_params=env_params,
        num_envs=eval_episodes,
        max_steps_in_episode=max_steps,
    )

    # Create plot
    output_path = Path("data/temp/pqn_evaluation.html")
    plot_episodes(
        observations=trajectories[0],
        rewards=trajectories[2],
        filename=output_path,
        num_episodes_to_plot=eval_episodes,
    )
    print(f"Evaluation plot saved to {output_path}")


if __name__ == "__main__":
    main()
