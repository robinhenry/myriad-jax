"""Plotting utilities for visualizing training and evaluation results."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from myriad.platform.types import TrainingResults


def plot_training_curve(
    results: TrainingResults | list[TrainingResults],
    labels: str | list[str] | None = None,
    xlabel: str = "Steps per Env",  # cspell:disable-line
    ylabel: str = "Mean Return",  # cspell:disable-line
    title: str | None = None,
    figsize: tuple[float, float] = (8, 4),
    show_std: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot training curve(s) showing mean return with optional standard deviation.

    Args:
        results: Single TrainingResults or list of results to plot
        labels: Legend label(s) for the curve(s). If None, uses agent name from config
        xlabel: Label for x-axis  # cspell:disable-line
        ylabel: Label for y-axis  # cspell:disable-line
        title: Plot title. If None, auto-generates from environment name
        figsize: Figure size (width, height) in inches
        show_std: Whether to show standard deviation as shaded region
        ax: Existing axes to plot on. If None, creates new figure

    Returns:
        Tuple of (figure, axes) objects

    Example:
        >>> results = train_and_evaluate(config)
        >>> fig, ax = plot_training_curve(results)
        >>> plt.show()

        >>> # Compare multiple runs
        >>> results_list = [results_dqn, results_ppo]
        >>> fig, ax = plot_training_curve(results_list, labels=["DQN", "PPO"])
        >>> plt.show()
    """
    # Normalize inputs to lists
    results_list = [results] if not isinstance(results, list) else results

    # Handle labels
    if labels is None:
        labels_list = [r.config.agent.name.upper() for r in results_list]
    elif isinstance(labels, str):
        labels_list = [labels]
    else:
        labels_list = labels

    if len(labels_list) != len(results_list):
        raise ValueError(f"Number of labels ({len(labels_list)}) must match number of results ({len(results_list)})")

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
        if fig is None:
            raise ValueError("Provided axes must be attached to a figure")

    # Plot each result
    for result, label in zip(results_list, labels_list):
        steps = result.eval_metrics.steps_per_env
        mean = result.eval_metrics.mean_return
        std = result.eval_metrics.std_return

        # Plot mean line
        ax.plot(steps, mean, "o-", label=label)

        # Add standard deviation band
        if show_std:
            mean_arr = np.array(mean)
            std_arr = np.array(std)
            ax.fill_between(steps, mean_arr - std_arr, mean_arr + std_arr, alpha=0.2)

    # Formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Auto-generate title if not provided
    if title is None and len(results_list) == 1:
        env_name = results_list[0].config.env.name
        agent_name = results_list[0].config.agent.name.upper()
        title = f"{agent_name} Training on {env_name}"

    if title is not None:
        ax.set_title(title)

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Apply tight_layout if available (SubFigure doesn't have this method)
    if hasattr(fig, "tight_layout"):
        fig.tight_layout()  # type: ignore[union-attr]

    return fig, ax  # type: ignore[return-value]
