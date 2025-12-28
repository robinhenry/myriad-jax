"""Advanced training example with custom configuration.

This example demonstrates how to customize training with additional
parameters and agent-specific hyperparameters.
"""

from myriad import create_config, train_and_evaluate

# Create config with agent-specific hyperparameters
config = create_config(
    env="cartpole-control",
    agent="dqn",
    num_envs=1000,
    steps_per_env=100,
    eval_frequency=20,
    eval_rollouts=20,
    seed=123,
    # Agent-specific parameters (passed through to agent)
    learning_rate=1e-3,
    batch_size=64,
    buffer_size=10000,
    # Additional overrides using dot notation
    **{
        "agent.epsilon_start": 1.0,
        "agent.epsilon_end": 0.01,
        "agent.epsilon_decay_steps": 10000,
    },
)

print("Config created:")
print(f"  Environment: {config.env.name}")
print(f"  Agent: {config.agent.name}")
print(f"  Total timesteps: {config.run.total_timesteps}")
print(f"  Parallel envs: {config.run.num_envs}")

# Run training
print("\nStarting training...")
results = train_and_evaluate(config)

# Analyze results
print("\nTraining completed!")
print(f"Final evaluation return: {results.eval_metrics.mean_return[-1]:.2f}")
print(f"Training steps: {results.training_metrics.global_steps[-1]:,}")

# Try to save agent
try:
    results.save_agent("outputs/advanced_agent.pkl")
    print("\nAgent saved to outputs/advanced_agent.pkl")
except RuntimeError as e:
    print(f"\nNote: Could not serialize agent: {e}")
    print("Agent state is still available in results.agent_state")
