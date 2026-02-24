#!/usr/bin/env bash

# Evaluate DQN agent
 for config in dqn; do
    MYRIAD_CONFIG_PATH=$(pwd)/configs myriad train --config-name $config
  done
