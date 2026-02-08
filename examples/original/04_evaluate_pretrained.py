"""Evaluate a pre-trained agent.

This example shows how to load a pre-trained agent and evaluate it.
Run 02_basic_training.py first to create the saved agent.
"""

from pathlib import Path

from myriad import create_eval_config, evaluate
from myriad.platform.types import TrainingResults

# Check if trained agent exists
agent_path = Path("outputs/basic_training_agent.pkl")
if not agent_path.exists():
    print(f"Note: {agent_path} not found.")
    print("This example requires a saved agent from 02_basic_training.py.")
    print("Due to JAX/Flax serialization limitations, agent saving may not work.")
    print("\nSkipping this example - use results.agent_state directly instead.")
    exit(0)

# Load the trained agent
print(f"Loading agent from {agent_path}...")
try:
    agent_state = TrainingResults.load_agent(agent_path)
except (EOFError, RuntimeError, Exception) as e:
    print(f"Failed to load agent: {e}")
    print("This is expected - JAX/Flax agent state often cannot be pickled.")
    print("\nSkipping this example - use results.agent_state directly instead.")
    exit(0)

# Create evaluation config
config = create_eval_config(
    env="cartpole-control",
    agent="dqn",
    eval_rollouts=100,
    seed=42,
)

# Run evaluation
print("Evaluating agent...")
results = evaluate(config, agent_state=agent_state, return_episodes=False)

# Display results
print("\nEvaluation completed!")
print(results)
print("\nDetailed statistics:")
for key, value in results.summary().items():
    print(f"  {key}: {value:.2f}")
