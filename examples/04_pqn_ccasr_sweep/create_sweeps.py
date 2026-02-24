#!/usr/bin/env python3
"""Create W&B sweeps for each num_envs level.

For each level, patches run.num_envs and wandb.group in the base sweep YAML,
then calls `wandb sweep` to register it. Prints sweep IDs for manual agent launch.

Usage:
    python create_sweeps.py                      # create sweeps for all levels
    python create_sweeps.py --project my-proj    # custom W&B project
    python create_sweeps.py --levels 512 1024    # specific levels only

Workflow:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     Full Sweep Workflow                             │
    │                                                                     │
    │  1. create_sweeps.py                                                │
    │     └─ For each num_envs level (512, 1024, 16384):                 │
    │        └─ Patch pqn_ccasr_sweep.yaml → wandb sweep → sweep_id      │
    │                                                                     │
    │  2. wandb agent <sweep_id>  (one per GPU or remote job)            │
    │     └─ W&B samples (hyperparams, seed) from Bayesian posterior      │
    │     └─ Calls: python scripts/train_sweep.py <hydra overrides>       │
    │        └─ sweep_main():                                             │
    │           1. wandb.init()          → connects to sweep              │
    │           2. wandb.config          → reads sampled hyperparams      │
    │           3. inject into cfg       → overrides Hydra config         │
    │           4. train_and_evaluate()  → full training run              │
    │           5. logs eval/episode_return/mean → W&B posterior update   │
    │           6. wandb.finish()                                         │
    │                                                                     │
    │  3. W&B UI: filter by wandb.group to view per-level results        │
    │     Group by run.seed to inspect variance across seeds             │
    └─────────────────────────────────────────────────────────────────────┘
"""
import argparse
import copy
import subprocess
import tempfile
from pathlib import Path

import yaml

NUM_ENVS_LEVELS = [512, 1024, 16384]  # 2^9, 2^10, 2^14
BASE_YAML = Path(__file__).parent / "configs" / "pqn_ccasr_sweep.yaml"


def create_sweep(num_envs: int, project: str) -> str:
    with open(BASE_YAML) as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    cfg["parameters"]["run.num_envs"] = {"value": num_envs}
    cfg["parameters"]["wandb.group"] = {"value": f"pqn_ccasr_{num_envs}envs"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(cfg, tmp)
        tmp_path = tmp.name

    result = subprocess.run(
        ["wandb", "sweep", "--project", project, tmp_path],
        capture_output=True,
        text=True,
        check=True,
    )
    Path(tmp_path).unlink()

    # W&B prints "wandb agent <entity>/<project>/<sweep_id>" to stderr
    agent_lines = [line for line in result.stderr.splitlines() if "wandb agent" in line]
    if not agent_lines:
        raise RuntimeError(f"Could not parse sweep ID from wandb output:\n{result.stderr}")
    sweep_id = agent_lines[-1].split()[-1]
    return sweep_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="myriad-ccasr", help="W&B project name")
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=NUM_ENVS_LEVELS,
        help="num_envs levels to create sweeps for",
    )
    args = parser.parse_args()

    print(f"Creating {len(args.levels)} sweep(s) in project '{args.project}'")
    for num_envs in args.levels:
        sweep_id = create_sweep(num_envs, args.project)
        print(f"  num_envs={num_envs:5d}  →  wandb agent {sweep_id}")

    print("\nTo run agents (launch one per available GPU or remote job):")
    print("    wandb agent <sweep_id>")


if __name__ == "__main__":
    main()
