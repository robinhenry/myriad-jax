"""CcaS-CcaR gene circuit rendering utilities.

This module provides functions to render CcaS-CcaR fluorescence states as RGB frames
for video generation and visualization.

Rendering modes:
1. Single-cell trajectory: Shows one cell's fluorescence over time
2. Population heatmap: Shows many cells' fluorescence at a single timepoint
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def render_ccas_ccar_frame(
    state: np.ndarray,
    F_obs_normalizer: float = 80.0,
    trajectory_history: np.ndarray | None = None,
    action_history: np.ndarray | None = None,
    current_timestep: int = 0,
    figsize: tuple[float, float] = (10, 6),
    dpi: int = 100,
) -> np.ndarray:
    """Render a single CcaS-CcaR frame from an observation.

    Creates a visualization showing:
    - Fluorescence trajectory over time (if trajectory_history provided)
    - Current fluorescence level (F) vs target
    - Future target trajectory (if available in observation)
    - Light input status (on/off)

    Args:
        state: CcaS-CcaR observation array with shape (obs_dim,)
            Format: [F_normalized, U_obs, F_target[0], F_target[1], ..., F_target[n_horizon]]
            - F_normalized: Current GFP fluorescence (normalized by F_obs_normalizer)
            - U_obs: Light input observation (always 0.0 - not directly observable)
            - F_target: Target trajectory [current, t+1, ..., t+n_horizon]
        F_obs_normalizer: Normalization constant for F (default: 80.0)
            Used to convert normalized values back to molecule counts for display
        trajectory_history: Optional array of shape (timesteps, obs_dim) containing
            full observation history up to current timestep. If provided, shows
            trajectory over time. If None, only shows current state.
        action_history: Optional array of shape (timesteps,) containing action history
            (light on/off: 0 or 1). If provided, shows light control pattern over time.
        current_timestep: Current timestep index (used when trajectory_history is provided)
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch for rendering resolution

    Returns:
        RGB image array with shape (height, width, 3) and dtype uint8

    Example:
        >>> # Single frame without history
        >>> obs = np.array([0.3, 0.0, 0.31, 0.32])
        >>> frame = render_ccas_ccar_frame(obs)
        >>> frame.shape
        (600, 1000, 3)
        >>>
        >>> # Frame with trajectory history
        >>> history = np.array([[0.1, 0.0, 0.3, 0.3], [0.2, 0.0, 0.3, 0.3]])
        >>> frame = render_ccas_ccar_frame(obs, trajectory_history=history, current_timestep=2)
    """
    # Parse observation
    # Observation format: [F_normalized, U_obs, F_target[0], ..., F_target[n_horizon]]
    F_normalized = state[0]
    _U_obs = state[1]  # Not used (always 0.0, light not directly observable)
    F_target = state[2:]  # Target trajectory

    # Denormalize for display (convert back to molecule counts)
    F_current = F_normalized * F_obs_normalizer
    F_target_denorm = F_target * F_obs_normalizer

    # Create figure with 2 subplots: fluorescence plot (top) and light indicator (bottom)
    fig, (ax_fluor, ax_light) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, gridspec_kw={"height_ratios": [4, 1]})

    # --- Top plot: Fluorescence trajectory over time ---

    if trajectory_history is not None:
        # Extract fluorescence history from trajectory
        F_history = trajectory_history[: current_timestep + 1, 0] * F_obs_normalizer
        target_history = trajectory_history[: current_timestep + 1, 2] * F_obs_normalizer
        timesteps = np.arange(current_timestep + 1)

        # Plot actual fluorescence trajectory
        ax_fluor.plot(
            timesteps,
            F_history,
            "-",
            color="#2E86AB",
            linewidth=2.5,
            label="Actual F",
            zorder=3,
        )

        # Plot target trajectory history
        ax_fluor.plot(
            timesteps,
            target_history,
            "--",
            color="#A23B72",
            linewidth=2.0,
            label="Target F",
            alpha=0.7,
            zorder=2,
        )

        # Highlight current position with a dot
        ax_fluor.plot(
            current_timestep,
            F_current,
            "o",
            color="#2E86AB",
            markersize=12,
            markeredgewidth=2,
            markeredgecolor="white",
            zorder=5,
        )

        # Set x-axis to show full episode
        # Assume max 300 timesteps (288 is default max_steps)
        max_timesteps = max(300, current_timestep + 50)
        ax_fluor.set_xlim(-5, max_timesteps)
        ax_fluor.set_xlabel("Timestep (5 min intervals)", fontsize=10)

    else:
        # Fallback: show current state only (no history available)
        n_horizon = len(F_target_denorm)
        time_steps = np.arange(n_horizon)

        # Current value as a large dot
        ax_fluor.plot(
            0,
            F_current,
            "o",
            color="#2E86AB",
            markersize=16,
            markeredgewidth=2.5,
            markeredgecolor="white",
            label="Current F",
            zorder=5,
        )

        # Target trajectory as line and markers
        if n_horizon > 0:
            ax_fluor.plot(
                time_steps,
                F_target_denorm,
                "s-",
                color="#A23B72",
                markersize=8,
                linewidth=2.5,
                markeredgewidth=1.5,
                markeredgecolor="white",
                label="Target F",
                alpha=0.8,
            )

        ax_fluor.set_xlim(-0.5, max(n_horizon - 1, 1) + 0.5)
        ax_fluor.set_xlabel("Time Step (5 min intervals)", fontsize=10)

    # Common styling for fluorescence plot
    ax_fluor.set_ylim(0, F_obs_normalizer * 1.2)  # 0 to 120% of normalizer
    ax_fluor.set_ylabel("GFP Fluorescence (molecules)", fontsize=11, fontweight="bold")
    ax_fluor.grid(True, alpha=0.3, linestyle="--")
    ax_fluor.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax_fluor.set_facecolor("#f8f8f8")

    # Add horizontal reference lines
    ax_fluor.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

    # Add error indicator (distance from current to target)
    if len(F_target) > 0:
        error = abs(F_current - F_target_denorm[0])
        ax_fluor.text(
            0.02,
            0.98,
            f"t={current_timestep} | Error: {error:.1f}",
            transform=ax_fluor.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )

    # --- Bottom plot: Light control input ---

    if action_history is not None and trajectory_history is not None:
        # Show light on/off pattern over time
        actions = action_history[: current_timestep + 1]
        timesteps = np.arange(current_timestep + 1)

        # Plot as step function
        ax_light.step(
            timesteps,
            actions,
            where="post",
            color="#FF6B35",
            linewidth=2.5,
            label="Light Input",
        )

        # Fill under the curve to make on/off more visible
        ax_light.fill_between(timesteps, 0, actions, step="post", alpha=0.3, color="#FF6B35")

        # Set axis limits
        max_timesteps = max(300, current_timestep + 50)
        ax_light.set_xlim(-5, max_timesteps)
        ax_light.set_ylim(-0.1, 1.1)
        ax_light.set_ylabel("Light (U)", fontsize=10, fontweight="bold")
        ax_light.set_xlabel("Timestep", fontsize=9)
        ax_light.set_yticks([0, 1])
        ax_light.set_yticklabels(["OFF", "ON"])
        ax_light.grid(True, alpha=0.2, linestyle="--")
        ax_light.set_facecolor("#f8f8f8")
    else:
        # No action history available
        ax_light.text(
            0.5,
            0.5,
            "Light Control: Not Available",
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
            style="italic",
        )
        ax_light.set_xlim(0, 1)
        ax_light.set_ylim(0, 1)
        ax_light.axis("off")
        ax_light.set_facecolor("#f0f0f0")

    # Overall figure styling
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    # Convert matplotlib figure to numpy RGB array
    fig.canvas.draw()

    # Get ARGB buffer and convert to RGB
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)  # type: ignore

    # Calculate actual pixel dimensions from buffer size
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


def render_population_heatmap(
    observations: np.ndarray,
    actions: np.ndarray | None = None,
    F_obs_normalizer: float = 80.0,
    grid_shape: tuple[int, int] | None = None,
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 100,
) -> np.ndarray:
    """Render population-level heatmap of fluorescence across many cells.

    This visualization shows the fluorescence distribution across a large population
    of cells (e.g., 10K-100K+), arranged as a 2D grid like a microfluidic mother machine.

    Args:
        observations: Array of shape (n_envs, obs_dim) containing observations for all cells
            Format: each row is [F_normalized, U_obs, F_target[0], ...]
        actions: Optional array of shape (n_envs,) containing light actions (0=OFF, 1=ON)
            If provided, shows action overlay on the heatmap
        F_obs_normalizer: Normalization constant for F (default: 80.0)
        grid_shape: Optional (height, width) for reshaping. If None, auto-computes square grid
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch for rendering resolution

    Returns:
        RGB image array with shape (height, width, 3) and dtype uint8

    Example:
        >>> # Visualize 10,000 cells in a 100x100 grid
        >>> obs = np.random.rand(10000, 4) * 0.5  # Random fluorescence
        >>> actions = np.random.randint(0, 2, size=10000)  # Random actions
        >>> frame = render_population_heatmap(obs, actions)
        >>> frame.shape
        (1000, 1200, 3)
    """
    n_envs = observations.shape[0]

    # Extract fluorescence values (first column of observations)
    F_normalized = observations[:, 0]
    F_values = F_normalized * F_obs_normalizer

    # Auto-compute grid shape if not provided
    if grid_shape is None:
        # Make a square grid (or as close as possible)
        grid_width = int(np.ceil(np.sqrt(n_envs)))
        grid_height = int(np.ceil(n_envs / grid_width))
        grid_shape = (grid_height, grid_width)
    else:
        grid_height, grid_width = grid_shape

    # Pad fluorescence values to fit grid if necessary
    grid_size = grid_height * grid_width
    if n_envs < grid_size:
        # Pad with NaN (will show as white/empty in heatmap)
        F_values_padded = np.full(grid_size, np.nan)
        F_values_padded[:n_envs] = F_values
        F_values = F_values_padded

    # Reshape to 2D grid
    F_grid = F_values[: grid_height * grid_width].reshape(grid_height, grid_width)

    # Create figure
    fig, (ax_heatmap, ax_colorbar) = plt.subplots(1, 2, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": [20, 1]})

    # Plot fluorescence heatmap
    im = ax_heatmap.imshow(
        F_grid,
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
        norm=Normalize(vmin=0, vmax=F_obs_normalizer * 1.2),
    )

    # Overlay actions if provided
    if actions is not None:
        # Pad actions to match grid
        actions_padded = np.full(grid_size, np.nan)
        actions_padded[:n_envs] = actions
        actions_grid = actions_padded[: grid_height * grid_width].reshape(grid_height, grid_width)

        # Create binary mask where light is ON (action=1)
        light_on_mask = actions_grid == 1

        # Overlay light-on cells with red markers (small dots)
        y_coords, x_coords = np.where(light_on_mask)
        if len(y_coords) > 0:
            ax_heatmap.scatter(
                x_coords,
                y_coords,
                c="red",
                s=2,  # Small dots
                alpha=0.6,
                marker=".",
                label="Light ON",
            )

    # Styling for heatmap
    ax_heatmap.set_xlabel("Cell Column", fontsize=11, fontweight="bold")
    ax_heatmap.set_ylabel("Cell Row", fontsize=11, fontweight="bold")
    ax_heatmap.set_title(
        f"Population Fluorescence ({n_envs:,} cells)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    # Add legend if actions are shown
    if actions is not None:
        n_on = int(np.nansum(actions_padded == 1))
        n_off = int(np.nansum(actions_padded == 0))
        ax_heatmap.legend(
            loc="upper right",
            title=f"Light: {n_on:,} ON, {n_off:,} OFF",
            framealpha=0.9,
            fontsize=9,
        )

    # Colorbar
    cbar = plt.colorbar(im, cax=ax_colorbar)
    cbar.set_label("GFP Fluorescence (molecules)", rotation=270, labelpad=20, fontsize=11)

    # Overall figure styling
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    # Convert matplotlib figure to numpy RGB array
    fig.canvas.draw()

    # Get ARGB buffer and convert to RGB
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)  # type: ignore

    # Calculate actual pixel dimensions from buffer size
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
