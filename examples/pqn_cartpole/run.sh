#!/usr/bin/env bash
MYRIAD_CONFIG_PATH=$(pwd)/configs myriad train --auto-tune --config-name pqn_cartpole
