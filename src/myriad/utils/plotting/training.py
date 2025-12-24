"""Plotting functions for training metrics."""

from pathlib import Path
from typing import Dict, List, Union

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_training_metrics(
    metrics_history: List[Dict[str, float]],
    filename: Union[str, Path] = "data/temp/training_metrics.html",
):
    """
    Plots training metrics like loss and total reward.

    Args:
        metrics_history (List[Dict[str, float]]): A list of dictionaries,
            where each dictionary contains metrics for a training step.
        filename (Union[str, Path]): The name of the file to save the plot to.
    """
    if isinstance(filename, str):
        filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Training Loss", "Total Reward"),
    )

    episodes = list(range(len(metrics_history)))
    losses = [float(m["loss"]) for m in metrics_history]

    # Handle both old and new metric formats
    if "total_reward" in metrics_history[0]:
        total_rewards = [float(m["total_reward"]) for m in metrics_history]
    else:
        total_rewards = [float(m["mean_episode_return"]) for m in metrics_history]

    # Plot Loss
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=losses,
            name="Loss",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    # Plot Total Reward
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=total_rewards,
            name="Total Reward",
            line=dict(color="green"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title_text="Training Metrics",
        xaxis_title_text="Episode",
    )
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Total Reward", row=2, col=1)

    fig.write_html(str(filename))
