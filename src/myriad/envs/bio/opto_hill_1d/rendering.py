"""opto_hill_1d rendering utilities.

Renders the 1D optogenetic circuit state as an RGB frame for video generation
and visualization.

- Top panel: X(t) trajectory over time.
- Bottom strip: U(t) continuous light intensity as a gray→yellow heatstrip
  (suppressed when the action is unknown — see ``show_action_strip``).
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .physics import PhysicsState
from .tasks.sysid import SysIdTaskConfig, SysIdTaskState


def render_frame(
    state: SysIdTaskState,
    config: SysIdTaskConfig,
    action: float | None = None,
    trajectory_history: list[SysIdTaskState] | None = None,
    action_history: list[float] | None = None,
    show_action_strip: bool = True,
    figsize: tuple[float, float] = (10.08, 6.08),
    dpi: int = 100,
) -> np.ndarray:
    """Render a single opto_hill_1d frame from task state.

    Args:
        state: Current task state (provides state.physics.X, state.t, state.U)
        config: Task configuration (provides X_obs_normalizer, max_steps)
        action: Current action being taken (continuous U ∈ [0, 1]). Optional.
        trajectory_history: Optional list of past SysIdTaskStates for plotting
            the full trajectory. If None, only the current value is shown.
        action_history: Optional list of past continuous actions (U ∈ [0, 1]).
            If provided together with trajectory_history, the light strip shows
            the control pattern over time.
        show_action_strip: If False, the bottom U(t) panel is omitted and the
            figure contains only the X(t) panel. Set False when the action is
            unknown (e.g. per-timestep rendering from a bare observation) to
            avoid silently drawing U=0.
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch

    Returns:
        RGB image array with shape (height, width, 3) and dtype uint8
    """
    X_current = float(state.physics.X)
    current_timestep = int(state.t)
    X_normalizer = config.X_obs_normalizer
    max_steps = config.max_steps

    if show_action_strip:
        fig, (ax_x, ax_light) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, gridspec_kw={"height_ratios": [4, 1]})
    else:
        fig, ax_x = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax_light = None

    # --- Top: X(t) trajectory ---
    if trajectory_history is not None and len(trajectory_history) > 0:
        history_len = min(len(trajectory_history), current_timestep + 1)
        X_history = np.array([float(s.physics.X) for s in trajectory_history[:history_len]])
        timesteps = np.arange(history_len)

        ax_x.plot(timesteps, X_history, "-", color="#2E86AB", linewidth=2.5, label="X(t)", zorder=3)
        ax_x.plot(
            current_timestep,
            X_current,
            "o",
            color="#2E86AB",
            markersize=12,
            markeredgewidth=2,
            markeredgecolor="white",
            zorder=5,
        )
    else:
        ax_x.plot(
            0,
            X_current,
            "o",
            color="#2E86AB",
            markersize=16,
            markeredgewidth=2.5,
            markeredgecolor="white",
            label="X",
            zorder=5,
        )

    x_upper = max_steps + 5
    ax_x.set_xlim(-5, x_upper)
    ax_x.set_ylim(0, X_normalizer * 1.5)
    ax_x.set_xlabel("Timestep", fontsize=10)
    ax_x.set_ylabel("X (molecules)", fontsize=11, fontweight="bold")
    ax_x.grid(True, alpha=0.3, linestyle="--")
    ax_x.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax_x.set_facecolor("#f8f8f8")
    ax_x.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

    if show_action_strip:
        U_for_label = float(action) if action is not None else float(state.U)
        label = f"t={current_timestep}  X={X_current:.1f}  U={U_for_label:.2f}"
    else:
        label = f"t={current_timestep}  X={X_current:.1f}"
    ax_x.text(
        0.02,
        0.98,
        label,
        transform=ax_x.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )

    # --- Bottom: U(t) continuous light strip ---
    if ax_light is not None:
        if action_history is not None and len(action_history) > 0:
            hist_len = min(len(action_history), current_timestep + 1)
            U_row = np.asarray(action_history[:hist_len], dtype=float).reshape(1, -1)
            ax_light.imshow(
                U_row,
                aspect="auto",
                cmap="YlOrBr",
                vmin=0.0,
                vmax=1.0,
                extent=(-0.5, hist_len - 0.5, 0, 1),
                interpolation="nearest",
            )
            ax_light.set_xlim(-5, x_upper)
            ax_light.set_ylim(0, 1)
            ax_light.set_yticks([])
            ax_light.set_xlabel("Timestep", fontsize=9)
            ax_light.set_ylabel("U (light)", fontsize=10, fontweight="bold")
        else:
            U_row = np.array([[U_for_label]])
            ax_light.imshow(
                U_row,
                aspect="auto",
                cmap="YlOrBr",
                vmin=0.0,
                vmax=1.0,
                extent=(0, 1, 0, 1),
                interpolation="nearest",
            )
            ax_light.set_xticks([])
            ax_light.set_yticks([])
            ax_light.set_xlabel(f"U = {U_for_label:.2f}", fontsize=9)
            ax_light.set_ylabel("U (light)", fontsize=10, fontweight="bold")

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    fig.canvas.draw()

    buf = fig.canvas.buffer_rgba()  # type: ignore[attr-defined]
    frame_rgba = np.asarray(buf)
    frame = np.array(frame_rgba[:, :, :3], dtype=np.uint8)

    plt.close(fig)
    return frame


def render_opto_hill_1d_frame_from_obs(obs: np.ndarray) -> np.ndarray:
    """Render a frame from a flat observation array.

    Convenience wrapper used by the platform's episode-to-video pipeline,
    which passes per-timestep observations without task state or action.
    Because U is not part of the observation, the bottom light strip is
    suppressed — drawing U=0 by default would silently misrepresent control
    history in the rendered video.

    Args:
        obs: Observation array of shape (1,) with [X_normalized].

    Returns:
        RGB image array with shape (height, width, 3) and dtype uint8.
    """
    config = SysIdTaskConfig()
    X = float(obs[0]) * config.X_obs_normalizer
    state = SysIdTaskState(
        physics=PhysicsState.create(time=jnp.array(0.0), X=jnp.array(X)),
        t=jnp.array(0),
        U=jnp.array(0.0, dtype=jnp.float32),
    )
    return render_frame(state, config, show_action_strip=False)
