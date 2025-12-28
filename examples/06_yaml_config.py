"""Using YAML config files (for backward compatibility).

This example shows how to load configuration from YAML files,
which is useful when migrating from CLI-based workflows.
"""

from pathlib import Path

from myriad.configs.default import EvalConfig
from myriad.platform import evaluate
from myriad.utils import load_config

# Check if YAML config exists
config_path = Path(__file__).parent / "01_classical_control.yaml"
if not config_path.exists():
    print(f"YAML config not found at {config_path}")
    print("This example uses the existing 01_classical_control.yaml")
    exit(1)

# Load config from YAML file
print(f"Loading config from {config_path}...")
config = load_config(config_path, EvalConfig)

# Run evaluation
print(f"Evaluating {config.agent.name} on {config.env.name}...")
results = evaluate(config=config, return_episodes=False)

# Display results
print("\nResults summary:")
for k, v in results.summary().items():
    print(f"  {k}: {v:.2f}")
