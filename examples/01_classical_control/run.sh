#!/usr/bin/env bash

# Evaluate random/bangbang/PI agents
 for config in dqn; do
    MYRIAD_CONFIG_PATH=$(pwd)/configs myriad train --config-name $config
  done
