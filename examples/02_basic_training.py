"""Basic training example.

This example shows the simplest way to train an agent using Myriad
programmatically.
"""

from myriad import create_config, train_and_evaluate

# Create a training config with minimal parameters
config = create_config(
    env="cartpole-control",
    agent="dqn",
    num_envs=100,
    steps_per_env=50,
    eval_frequency=10,
)

# Run training
print("Starting training...")
results = train_and_evaluate(config)

# Display results
print("\nTraining completed!")
print(results)
print("\nDetailed summary:")
for key, value in results.summary().items():
    print(f"  {key}: {value}")

# Try to save trained agent (may fail due to JAX/Flax serialization limitations)
try:
    results.save_agent("outputs/basic_training_agent.pkl")
    print("\nAgent saved to outputs/basic_training_agent.pkl")
except RuntimeError as e:
    print(f"\nNote: Could not serialize agent ({e})")
    print("Agent state is still available in results.agent_state for in-memory use")
