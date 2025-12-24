"""Plotting functions for episodes."""

from pathlib import Path
from typing import Union

import jax.numpy as jnp
import plotly.colors as pcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_episodes(
    observations: jnp.ndarray,
    rewards: jnp.ndarray,
    filename: Union[str, Path] = "data/temp/episodes.html",
    num_episodes_to_plot: int = 5,
):
    """
    Plots the observations and rewards from multiple episodes on the same graph.

    Args:
        observations (jnp.ndarray): The observations from the episodes, with shape (num_episodes, num_steps, obs_dim).
        rewards (jnp.ndarray): The rewards from the episodes, with shape (num_episodes, num_steps).
        filename (Union[str, Path]): The name of the file to save the plot to.
        num_episodes_to_plot (int): The number of episodes to plot.
    """
    if isinstance(filename, str):
        filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Observations", "Rewards"),
    )

    colors = pcolors.qualitative.Plotly
    num_episodes = observations.shape[0]
    num_to_plot = min(num_episodes, num_episodes_to_plot)

    for i in range(num_to_plot):
        color = colors[i % len(colors)]
        episode_obs = observations[i]
        episode_rewards = rewards[i]

        # Plot observations
        fig.add_trace(
            go.Scatter(
                y=episode_obs[:, 0],
                name=f"State (x) - Ep {i+1}",
                legendgroup=f"ep{i+1}",
                line=dict(color=color),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Plot rewards
        fig.add_trace(
            go.Scatter(
                y=episode_rewards,
                name=f"Reward - Ep {i+1}",
                legendgroup=f"ep{i+1}",
                line=dict(color=color),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Plot target only once
    fig.add_trace(
        go.Scatter(
            y=observations[0, :, 1],
            name="Target (x_target)",
            line=dict(color="red", dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        title_text="Episode Rollouts",
        xaxis_title_text="Time Step",
    )
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Reward", row=2, col=1)

    fig.write_html(str(filename))
