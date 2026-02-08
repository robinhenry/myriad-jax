from pathlib import Path

from myriad.configs.default import EvalConfig
from myriad.platform import evaluate
from myriad.utils import load_config

# Load config from YAML file
config_path = Path(__file__).parent / "01_classical_control.yaml"
config = load_config(config_path, EvalConfig)

# Run evaluation
results = evaluate(config=config, return_episodes=True)

print("\nResults summary:")
for k, v in results.summary().items():
    print(f"\t{k}: {v:.2f}")
