#!/usr/bin/env bash
# Runs a two-phase hyperparameter sweep for PQN on ccasr-gfp-control.
# Run from any directory — paths are resolved relative to this script.
#
# Usage:
#   ./run_sweep.sh
#
# Prerequisites:
#   pip install myriad-jax     # installs the `myriad` CLI
#   wandb login                # once, to authenticate
#   export WANDB_ENTITY=your-username-or-team
#
# Project name is taken from the 'project:' field in the sweep YAML.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_YAML="$SCRIPT_DIR/configs/pqn_ccasr_sweep.yaml"
NUM_ENVS=1024
BASE_GROUP="pqn_ccasr"

echo "=== Phase 1: creating sweep ==="
SWEEP_ID=$(myriad sweep-create "$SWEEP_YAML" \
    --base-group "$BASE_GROUP" \
    --levels "$NUM_ENVS")

echo ""
echo "=== Phase 1: running agent ==="
MYRIAD_CONFIG_PATH="$SCRIPT_DIR/configs" wandb agent "$SWEEP_ID"

echo ""
echo "=== Phase 2: seed-eval ==="
myriad seed-eval "$SWEEP_ID" \
    --top-k 5 \
    --seeds 0,1,2 \
    --metric eval/return/best \
    --group "${BASE_GROUP}_validated"
