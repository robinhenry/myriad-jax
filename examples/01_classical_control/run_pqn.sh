#!/usr/bin/env bash

# Evaluate PQN agent
 for config in pqn; do
    MYRIAD_CONFIG_PATH=$(pwd)/configs myriad train --auto-tune --config-name $config
  done
