#!/usr/bin/env bash

# Train PQN (fig3: 1M envs, frame stacking)
# See configs/pqn.yaml for full parameter mapping from the CDC 2025 paper.
MYRIAD_CONFIG_PATH=$(pwd)/configs myriad train --config-name pqn
