"""Quickstart example - simplest possible usage.

This is the code shown in the main documentation index.
"""

from myriad import create_config, train_and_evaluate

# Create config
config = create_config(
    env="cartpole-control",
    agent="dqn",
    num_envs=100,  # Reduced for quick demo
    steps_per_env=50,
    eval_frequency=10,  # Eval every 10 steps
)

# Train
print("Running quickstart training...")
results = train_and_evaluate(config)
print("\nTraining complete!")
print(results)
