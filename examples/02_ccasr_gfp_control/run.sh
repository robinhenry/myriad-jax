#!/usr/bin/env bash

# Evaluate classical baselines (fig3)
for config in random bangbang pi; do
  MYRIAD_CONFIG_PATH=$(pwd)/configs myriad evaluate --config-name $config
done

# Train PQN (fig3: 1024 envs, ~30M total timesteps, frame stacking)
# See configs/pqn.yaml for full parameter mapping from the CDC 2025 paper.
MYRIAD_CONFIG_PATH=$(pwd)/configs myriad train --config-name pqn
