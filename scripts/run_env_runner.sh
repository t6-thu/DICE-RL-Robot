#!/bin/bash
# Run the RL finetuning env runner on the robot PC.
# Usage: bash scripts/run_env_runner.sh

set -e

# Verify required env vars
if [ -z "$DICE_HARDWARE_CONFIG_FOLDERS" ]; then
    echo "ERROR: DICE_HARDWARE_CONFIG_FOLDERS not set"
    exit 1
fi
if [ -z "$DICE_DATASET_FOLDERS" ]; then
    echo "ERROR: DICE_DATASET_FOLDERS not set"
    exit 1
fi
if [ -z "$DICE_CHECKPOINT_FOLDERS" ]; then
    echo "ERROR: DICE_CHECKPOINT_FOLDERS not set"
    exit 1
fi

python -m dice_rl.env_runner.run_env_runner "$@"
