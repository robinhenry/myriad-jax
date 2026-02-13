#!/usr/bin/env bash

# Evaluate random/bangbang/PI agents
 for config in random bangbang pi; do
    MYRIAD_CONFIG_PATH=$(pwd)/configs myriad evaluate --config-name $config
  done
