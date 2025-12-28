"""Evaluate a random baseline agent.

This example shows how to evaluate non-learning controllers like
random agents without any training.
"""

from myriad import create_eval_config, evaluate

# Create evaluation config for random agent
config = create_eval_config(
    env="cartpole-control",
    agent="random",
    eval_rollouts=100,
    seed=42,
)

# Run evaluation (no training needed for random agent)
print("Evaluating random agent...")
results = evaluate(config, agent_state=None, return_episodes=False)

# Display results
print("\nEvaluation completed!")
print(results)
print(f"\nRandom baseline mean return: {results.mean_return:.2f} ± {results.std_return:.2f}")
print(f"Range: [{results.min_return:.2f}, {results.max_return:.2f}]")
