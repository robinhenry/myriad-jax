#!/usr/bin/env bash
# Creates W&B sweeps for all num_envs levels and launches local agents.
# Run from the repo root directory.
#
# Usage:
#   ./examples/04_pqn_ccasr_sweep/run_sweep.sh
#   WANDB_PROJECT=my-project NUM_AGENTS=2 ./examples/04_pqn_ccasr_sweep/run_sweep.sh
#
# Prerequisites:
#   wandb login             # once, to authenticate
#   export WANDB_ENTITY=your-username-or-team

set -euo pipefail

WANDB_PROJECT="${WANDB_PROJECT:-myriad-ccasr}"
NUM_AGENTS="${NUM_AGENTS:-1}"   # parallel agents per sweep level
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Creating sweeps ==="
# Capture sweep IDs output by create_sweeps.py (one per num_envs level)
mapfile -t SWEEP_IDS < <(
    python "$EXAMPLE_DIR/create_sweeps.py" --project "$WANDB_PROJECT" \
    | grep "wandb agent" | awk '{print $NF}'
)

echo ""
echo "=== Launching ${NUM_AGENTS} agent(s) per sweep ==="
for sweep_id in "${SWEEP_IDS[@]}"; do
    echo "  Launching agents for $sweep_id"
    for _ in $(seq 1 "$NUM_AGENTS"); do
        wandb agent "$sweep_id" &
    done
done

echo ""
echo "All agents running. Monitor at https://wandb.ai/${WANDB_ENTITY:-<your-entity>}/$WANDB_PROJECT"
wait
