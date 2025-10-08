"""Run a single episode with a random agent on the toy problem."""

from pathlib import Path

import click
import jax
import jax.numpy as jnp

from aion.agents.random import RandomAgent
from aion.envs.toy_env_v1 import ToyProblem
from aion.platform.interaction import run_episodes_parallel
from aion.utils.plotting.episodes import plot_episodes


@click.command()
@click.option("--num_episodes", default=10, help="Number of episodes to run in parallel.")
@click.option("--num_steps", default=100, help="Number of steps per episode.")
@click.option("--num_episodes_to_plot", default=5, help="Number of episodes to plot.")
def main(num_episodes: int, num_steps: int, num_episodes_to_plot: int):
    """Main function to run the script."""
    # Create a random key
    key = jax.random.PRNGKey(0)

    # Instantiate the environment
    env = ToyProblem()

    # Instantiate the agent
    agent = RandomAgent(env.action_space(env.default_params))

    # Run the episodes in parallel
    print(f"Running {num_episodes} episodes in parallel...")
    observations, rewards = run_episodes_parallel(key, env, agent, num_episodes, num_steps)

    # Transpose observations and rewards to have the episode dimension first
    observations = jnp.transpose(observations, (1, 0, 2))
    rewards = jnp.transpose(rewards, (1, 0))

    # Print the results for the first episode
    print(f"Episode 1 finished after {rewards.shape[1]} steps.")
    print(f"Total reward for episode 1: {jnp.sum(rewards[0])}")
    print(f"Final observation for episode 1: {observations[0, -1]}")

    # Plot the episodes
    print(f"Plotting {num_episodes_to_plot} episodes...")
    plot_filename = Path("data/temp/random_agent_parallel_episodes.html")
    plot_episodes(
        observations,
        rewards,
        filename=plot_filename,
        num_episodes_to_plot=num_episodes_to_plot,
    )
    print(f"Plot saved to {plot_filename}")


if __name__ == "__main__":
    main()
