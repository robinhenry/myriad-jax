"""CartPole-specific rendering utilities.

This module provides functions to render CartPole states as RGB frames
for video generation and visualization.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def render_cartpole_frame(
    state: np.ndarray,
    pole_length: float = 1.0,
    cart_width: float = 0.5,
    cart_height: float = 0.3,
    x_limit: float = 2.4,
    figsize: tuple[float, float] = (6, 4),
    dpi: int = 100,
) -> np.ndarray:
    """Render a single CartPole frame from a state.

    Pure rendering function that converts CartPole state to RGB pixel array.
    Follows the standard CartPole visualization: cart on a track with pole attached.

    Args:
        state: CartPole state array with shape (4,) containing [x, x_dot, theta, theta_dot]
        pole_length: Full length of the pole in meters (default: 1.0, which is 2 * physics.pole_length)
        cart_width: Width of the cart rectangle in meters
        cart_height: Height of the cart rectangle in meters
        x_limit: Position limits of the track (cart can move in [-x_limit, +x_limit])
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch for rendering resolution

    Returns:
        RGB image array with shape (height, width, 3) and dtype uint8

    Example:
        >>> obs = np.array([0.1, 0.0, 0.05, 0.0])  # Slight tilt
        >>> frame = render_cartpole_frame(obs)
        >>> frame.shape
        (400, 600, 3)
    """
    # Extract state components (only x and theta are needed for rendering)
    x, _x_dot, theta, _theta_dot = state

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

    # Get ARGB buffer (FigureCanvasMac on macOS)
    # Note: The actual pixel dimensions may differ from fig size due to DPI/retina scaling
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)  # type: ignore

    # Calculate actual pixel dimensions from buffer size
    # ARGB format: 4 bytes per pixel
    num_pixels = len(buf) // 4
    height_pixels = int(np.sqrt(num_pixels * figsize[1] / figsize[0]))
    width_pixels = num_pixels // height_pixels

    # Reshape and convert ARGB to RGB
    buf = buf.reshape(height_pixels, width_pixels, 4)
    frame = np.zeros((height_pixels, width_pixels, 3), dtype=np.uint8)
    frame[:, :, 0] = buf[:, :, 1]  # R
    frame[:, :, 1] = buf[:, :, 2]  # G
    frame[:, :, 2] = buf[:, :, 3]  # B

    plt.close(fig)

    return frame
