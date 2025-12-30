"""Run and render a CcaS-CcaR gene circuit episode.

This example shows how to:
1. Run an episode with a random agent on the CcaS-CcaR environment
2. Collect the episode trajectory
3. Render it to a video file
"""

import numpy as np

from myriad import create_eval_config, evaluate
from myriad.utils.rendering import frames_to_video, render_ccas_ccar_frame

# Configuration
OUTPUT_VIDEO = "videos/ccas_ccar_episode.mp4"
FPS = 10  # 10 fps (each timestep is 5 minutes, so 10 fps = real-time-ish visualization)

# Create evaluation config for CcaS-CcaR control task
config = create_eval_config(
    env="ccas-ccar-control",
    agent="bangbang",
    eval_rollouts=1,  # Just one episode for visualization
    seed=42,
    **{"agent.threshold": 25 / 80, "agent.obs_field": "F_normalized", "agent.invert": True},
)

# Run evaluation and collect episode data
print("Running CcaS-CcaR episode with bangbang agent...")
results = evaluate(config, return_episodes=True)

# Extract episode data
if results.episodes is not None:
    episodes = results.episodes
    observations = episodes["observations"]  # Shape: (1, max_steps, obs_dim)
    actions = episodes["actions"]  # Shape: (1, max_steps)
    episode_length = int(results.episode_lengths[0])

    print(f"\nEpisode collected: {episode_length} timesteps")
    print(f"Episode return: {results.episode_returns[0]:.2f}")
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")

    # Render frames from observations with trajectory and action history
    print(f"\nRendering {episode_length} frames...")
    frames = []
    for t in range(episode_length):
        obs = observations[0, t]  # Get observation at timestep t
        # Pass full history up to current timestep to show trajectory
        frame = render_ccas_ccar_frame(
            obs,
            trajectory_history=observations[0],  # Full trajectory
            action_history=actions[0],  # Full action history
            current_timestep=t,
        )
        frames.append(frame)
        if (t + 1) % 50 == 0:
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

else:
    print("Error: No episodes collected")
