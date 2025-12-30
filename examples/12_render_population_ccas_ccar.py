"""Render population-level heatmap for CcaS-CcaR gene circuit.

This example demonstrates Myriad's core value proposition: visualizing population-scale
experiments with thousands of parallel cells, like a microfluidic mother machine.

Shows:
1. Training with 10,000 parallel environments
2. Rendering fluorescence distribution as a 2D heatmap
3. Overlaying light control actions on the population
"""

import numpy as np

from myriad import create_eval_config, evaluate
from myriad.utils.rendering import frames_to_video, render_population_heatmap

# Configuration
OUTPUT_VIDEO = "videos/ccas_ccar_population.mp4"
N_CELLS = 10_000  # 10K parallel cells
FPS = 5  # 5 fps for population view

# Create evaluation config with many parallel environments
# Note: eval_rollouts controls the number of parallel environments
print(f"Running with {N_CELLS:,} parallel cells...")
config = create_eval_config(
    env="ccas-ccar-control",
    agent="bangbang",  # bangbang agent for demonstration
    **{"agent.threshold": 25 / 80, "agent.obs_field": "F_normalized", "agent.invert": True},
    eval_rollouts=N_CELLS,  # Number of parallel cells to simulate
    seed=42,
)

# Run evaluation with episode collection
print("Collecting population data...")
results = evaluate(config, return_episodes=True)

# Extract episode data
if results.episodes is not None:
    episodes = results.episodes
    # For parallel envs, episodes shape is (num_rollouts, max_steps, n_envs, obs_dim) or similar
    # Need to check actual shape and transpose if necessary
    observations = episodes["observations"]  # Check shape
    actions = episodes["actions"]  # Check shape

    print("\nRaw episode shapes:")
    print(f"  Observations: {observations.shape}")
    print(f"  Actions: {actions.shape}")

    # Determine actual episode length from the first rollout
    episode_length = int(results.episode_lengths[0])

    # Reshape if needed - evaluate returns (num_rollouts, max_steps, obs_dim)
    # but for parallel envs we want (max_steps, n_envs, obs_dim)
    # Actually, let's just use the first rollout which has all parallel envs
    if observations.ndim == 3:
        # Shape is (num_rollouts, max_steps, obs_dim) - single env per rollout
        # We need to treat this as (max_steps, n_rollouts, obs_dim)
        observations = observations.transpose(1, 0, 2)  # (max_steps, n_envs, obs_dim)
        actions = actions.transpose(1, 0)  # (max_steps, n_envs)

    print("\nEpisode collected:")
    print(f"  Environments: {observations.shape[1]:,}")
    print(f"  Timesteps: {episode_length}")
    print(f"  Total observations: {observations.shape[1] * episode_length:,}")

    # Render population heatmap for each timestep
    print(f"\nRendering {episode_length} frames...")
    frames = []
    for t in range(episode_length):
        # Get observations and actions for all environments at timestep t
        obs_t = observations[t, :, :]  # Shape: (n_envs, obs_dim)
        actions_t = actions[t, :]  # Shape: (n_envs,)

        # Render population heatmap
        frame = render_population_heatmap(
            obs_t,
            actions_t,
            grid_shape=None,  # Auto-compute square grid
        )
        frames.append(frame)

        if (t + 1) % 10 == 0:
            print(f"  Rendered {t + 1}/{episode_length} frames")

    # Convert frames to video
    print(f"\nCreating video: {OUTPUT_VIDEO}")
    video_path = frames_to_video(
        frames=np.array(frames),
        output_path=OUTPUT_VIDEO,
        fps=FPS,
        quality=8,
    )

    print(f"\nVideo saved to: {video_path}")
    print(f"Duration: {episode_length / FPS:.1f} seconds")
    print("\nVisualization shows:")
    print(f"  - Heatmap: Fluorescence intensity across {N_CELLS:,} cells")
    print(f"  - Grid: {int(np.ceil(np.sqrt(N_CELLS)))}x{int(np.ceil(np.sqrt(N_CELLS)))} cells")
    print("  - Red dots: Cells with light ON")
    print("  - Colorbar: GFP molecule count")
    print("\nThis demonstrates population-scale parallelism:")
    print(f"  {N_CELLS:,} cells simulated simultaneously on GPU")

else:
    print("Error: No episodes collected")
