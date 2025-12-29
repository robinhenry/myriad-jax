"""Save episodes periodically during training.

This example shows how to configure automatic episode saving
to disk during training for qualitative monitoring.
"""

from pathlib import Path

import numpy as np

from myriad import create_config, train_and_evaluate

# Create config with periodic episode saving
config = create_config(
    env="cartpole-control",
    agent="dqn",
    num_envs=100,
    steps_per_env=100,
    eval_frequency=20,  # Eval every 20 steps
    eval_rollouts=5,  # Run 5 episodes per eval
    # Episode saving configuration
    eval_episode_save_frequency=40,  # Save every 40 steps (2x eval frequency)
    eval_episode_save_count=2,  # Save first 2 episodes only
)

print("Training with periodic episode saving...")
results = train_and_evaluate(config)

print("\nTraining complete. Episodes saved to: episodes/")

# Load and inspect a saved episode
episodes_dir = Path("episodes")
if episodes_dir.exists():
    # Find first saved episode
    saved_episodes = list(episodes_dir.glob("step_*/episode_*.npz"))
    if saved_episodes:
        episode_file = saved_episodes[0]
        print(f"\nLoading saved episode: {episode_file}")

        # Load episode data
        data = np.load(episode_file)

        # Trajectory data (no padding - already trimmed)
        observations = data["observations"]  # Shape: (episode_length, obs_dim)
        actions = data["actions"]  # Shape: (episode_length, action_dim)
        rewards = data["rewards"]  # Shape: (episode_length,)

        # Metadata
        ep_len = int(data["episode_length"])
        ep_return = float(data["episode_return"])

        print(f"Episode length: {ep_len}")
        print(f"Episode return: {ep_return:.2f}")
        print(f"Observations shape: {observations.shape}")
        print(f"Actions shape: {actions.shape}")
        print(f"Rewards shape: {rewards.shape}")
    else:
        print("No episodes were saved (training may have been too short)")
else:
    print("Episodes directory not found")
