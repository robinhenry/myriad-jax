"""Collect full episode trajectories during evaluation.

This example shows how to get detailed episode data including
observations, actions, rewards, and dones for analysis or visualization.
"""

from myriad import create_eval_config, evaluate

# Create evaluation config
config = create_eval_config(
    env="cartpole-control",
    agent="random",
    eval_rollouts=5,
)

# Run evaluation and collect episodes
print("Collecting episode trajectories...")
results = evaluate(config, return_episodes=True)

# Access episode data
if results.episodes is not None:
    episodes = results.episodes
    obs = episodes["observations"]  # Shape: (num_rollouts, max_steps, obs_dim)
    actions = episodes["actions"]  # Shape: (num_rollouts, max_steps, action_dim)
    rewards = episodes["rewards"]  # Shape: (num_rollouts, max_steps)
    dones = episodes["dones"]  # Shape: (num_rollouts, max_steps)

    print(f"\nCollected {results.num_episodes} episodes")
    print(f"Observations shape: {obs.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")

    # Example: process first episode (trim padding using episode_length)
    ep_len = int(results.episode_lengths[0])
    valid_obs = obs[0, :ep_len]  # No padding
    print(f"\nFirst episode length: {ep_len}")
    print(f"Valid observations shape: {valid_obs.shape}")
else:
    print("No episodes collected")
