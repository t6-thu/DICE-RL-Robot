#!/bin/bash
# Source this file from the repo root: `. ./prepare.sh`

DICE_REPO_ROOT="$HOME/Documents/niu/DICE-RL-Robot"

# uv venv living at the repo root (see installation steps in README).
export LD_LIBRARY_PATH="$DICE_REPO_ROOT/.venv/lib:$LD_LIBRARY_PATH"

# where the collected raw data folders are
export DICE_RAW_DATASET_FOLDERS=$HOME/data/real
# where the post-processed data folders are
export DICE_DATASET_FOLDERS=$HOME/data/real_processed
# Each training session will create a folder here.
export DICE_CHECKPOINT_FOLDERS=$HOME/training_outputs
# Hardware interfaces root.
export DICE_HARDWARE_INTERFACES_ROOT="$DICE_REPO_ROOT/hardware_interfaces"
# Hardware configs.
export DICE_HARDWARE_CONFIG_FOLDERS=$DICE_HARDWARE_INTERFACES_ROOT/workcell/table_top_manip/config
# Logging folder.
export DICE_CONTROL_LOG_FOLDERS=$HOME/data/control_log

# Activate the uv venv (replaces `conda activate dice-rl-robot`).
# shellcheck disable=SC1091
source "$DICE_REPO_ROOT/.venv/bin/activate"
