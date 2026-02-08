"""Pendulum-specific rendering utilities.

This module provides functions to render Pendulum states as RGB frames
for video generation and visualization.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from myriad.envs.classic.pendulum.physics import PhysicsState
from myriad.envs.classic.pendulum.tasks.control import ControlTaskConfig, ControlTaskState


def render_pendulum_frame(
    state: ControlTaskState,
    config: ControlTaskConfig,
    action: float | None = None,
    length: float = 1.0,
    figsize: tuple[float, float] = (4, 4),
    dpi: int = 100,
) -> np.ndarray:
    """Render a single Pendulum frame from task state.

    Pure rendering function that converts Pendulum state to RGB pixel array.
    Follows the standard Pendulum visualization: bob hanging from pivot point.

    The coordinate system has theta=0 at the bottom (hanging down) and
    theta increases counter-clockwise.

    Args:
        state: Pendulum task state containing physics state.
            - state.physics.theta: Angle from vertical down (rad, 0 = hanging down, pi = upright)
            - state.physics.theta_dot: Angular velocity (rad/s)
            - state.t: Current timestep
        config: Task configuration (currently unused, included for interface consistency).
        action: Current action (torque) being applied. Optional, unused in rendering.
        length: Length of the pendulum rod in meters
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch for rendering resolution

    Returns:
        RGB image array with shape (height, width, 3) and dtype uint8

    Example:
        >>> from myriad.envs.classic.pendulum.tasks.control import make_env
        >>> import jax
        >>> env = make_env()
        >>> key = jax.random.PRNGKey(0)
        >>> obs, state = env.reset(key)
        >>> frame = render_pendulum_frame(state, env.config)
        >>> frame.shape
        (400, 400, 3)
    """
    # Extract theta from physics state
    theta = float(state.physics.theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Pivot point at origin
    pivot_x, pivot_y = 0.0, 0.0

    # Bob position (theta=0 is down, so we use -cos for y, sin for x)
    bob_x = pivot_x + length * sin_theta
    bob_y = pivot_y - length * cos_theta

    # Draw rod (dark gray line from pivot to bob)
    ax.plot([pivot_x, bob_x], [pivot_y, bob_y], "k-", linewidth=4, solid_capstyle="round")

    # Draw pivot point (black circle)
    ax.plot(pivot_x, pivot_y, "o", color="black", markersize=8)

    # Draw bob (red circle)
    bob_radius = 0.1 * length
    bob = patches.Circle(
        (bob_x, bob_y),
        bob_radius,
        facecolor="crimson",
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(bob)

    # Set axis limits and styling
    limit = length * 1.3
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add subtle background
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("white")

    # Convert matplotlib figure to numpy RGB array
    fig.canvas.draw()

    # Get RGBA buffer using standard method (Matplotlib 3.x+)
    buf = fig.canvas.buffer_rgba()

    # Convert buffer to numpy array
    frame_rgba = np.asarray(buf)

    # Extract RGB channels (drop Alpha)
    frame = np.array(frame_rgba[:, :, :3], dtype=np.uint8)

    plt.close(fig)

    return frame


def render_pendulum_frame_from_obs(obs: np.ndarray) -> np.ndarray:
    """Render a pendulum frame from a flat observation array.

    Convenience wrapper that reconstructs structured state from the observation
    vector and delegates to ``render_pendulum_frame``.

    Args:
        obs: Observation array of shape (3,) with [cos_theta, sin_theta, theta_dot].

    Returns:
        RGB image array with shape (height, width, 3) and dtype uint8.
    """
    theta = np.arctan2(float(obs[1]), float(obs[0]))
    state = ControlTaskState(
        physics=PhysicsState(theta=theta, theta_dot=float(obs[2])),
        t=np.int32(0),
    )
    return render_pendulum_frame(state, ControlTaskConfig())
