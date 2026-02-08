"""CartPole-specific rendering utilities.

This module provides functions to render CartPole states as RGB frames
for video generation and visualization.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from myriad.envs.classic.cartpole.physics import PhysicsState
from myriad.envs.classic.cartpole.tasks.control import ControlTaskConfig, ControlTaskState


def render_cartpole_frame(
    state: ControlTaskState,
    config: ControlTaskConfig,
    action: int | None = None,
    pole_length: float = 1.0,
    cart_width: float = 0.5,
    cart_height: float = 0.3,
    figsize: tuple[float, float] = (6, 4),
    dpi: int = 100,
) -> np.ndarray:
    """Render a single CartPole frame from task state.

    Pure rendering function that converts CartPole state to RGB pixel array.
    Follows the standard CartPole visualization: cart on a track with pole attached.

    Args:
        state: CartPole task state containing physics state.
            - state.physics.x: Cart position (m)
            - state.physics.x_dot: Cart velocity (m/s)
            - state.physics.theta: Pole angle from vertical (rad, 0 = upright)
            - state.physics.theta_dot: Pole angular velocity (rad/s)
            - state.t: Current timestep
        config: Task configuration containing physics parameters.
            - config.task.x_threshold: Position limit for track
        action: Current action being taken (0 or 1). Optional, unused in rendering.
        pole_length: Full length of the pole in meters (default: 1.0, which is 2 * physics.pole_length)
        cart_width: Width of the cart rectangle in meters
        cart_height: Height of the cart rectangle in meters
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch for rendering resolution

    Returns:
        RGB image array with shape (height, width, 3) and dtype uint8

    Example:
        >>> from myriad.envs.classic.cartpole.tasks.control import make_env
        >>> import jax
        >>> env = make_env()
        >>> key = jax.random.PRNGKey(0)
        >>> obs, state = env.reset(key)
        >>> frame = render_cartpole_frame(state, env.config)
        >>> frame.shape
        (400, 600, 3)
    """
    # Extract state components from physics state
    x = float(state.physics.x)
    theta = float(state.physics.theta)
    x_limit = config.task.x_threshold

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Draw track (black line)
    track_y = -cart_height / 2 - 0.1
    ax.plot([-x_limit, x_limit], [track_y, track_y], "k-", linewidth=3)

    # Draw cart (blue rectangle centered at x)
    cart = patches.Rectangle(
        (x - cart_width / 2, -cart_height / 2),
        cart_width,
        cart_height,
        facecolor="royalblue",
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(cart)

    # Draw pole (red line from cart center)
    # theta = 0 is upright, positive theta is clockwise rotation
    pole_x_end = x + pole_length * np.sin(theta)
    pole_y_end = pole_length * np.cos(theta)
    ax.plot([x, pole_x_end], [0, pole_y_end], "r-", linewidth=5, solid_capstyle="round")

    # Draw pole joint (orange circle)
    ax.plot(x, 0, "o", color="darkorange", markersize=10, markeredgewidth=2, markeredgecolor="black")

    # Set axis limits and styling
    ax.set_xlim(-x_limit - 0.2, x_limit + 0.2)
    ax.set_ylim(-0.5, pole_length + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add subtle background
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("white")

    # Convert matplotlib figure to numpy RGB array
    fig.canvas.draw()

    # Get RGBA buffer using standard method (Matplotlib 3.x+)
    # This is more robust than tostring_argb which is backend-dependent
    buf = fig.canvas.buffer_rgba()

    # Convert buffer to numpy array
    # shape is (height, width, 4) for RGBA
    frame_rgba = np.asarray(buf)

    # Extract RGB channels (drop Alpha)
    # Copy to ensure contiguous array and correct type
    frame = np.array(frame_rgba[:, :, :3], dtype=np.uint8)

    plt.close(fig)

    return frame


def render_cartpole_frame_from_obs(obs: np.ndarray) -> np.ndarray:
    """Render a cartpole frame from a flat observation array.

    Convenience wrapper that reconstructs structured state from the observation
    vector and delegates to ``render_cartpole_frame``.

    Args:
        obs: Observation array of shape (4,) with [x, x_dot, theta, theta_dot].

    Returns:
        RGB image array with shape (height, width, 3) and dtype uint8.
    """
    state = ControlTaskState(
        physics=PhysicsState.from_array(obs),
        t=np.int32(0),
    )
    return render_cartpole_frame(state, ControlTaskConfig())
