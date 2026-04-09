#!/bin/bash

export LD_LIBRARY_PATH=$HOME/miniforge3/envs/dice-rl-robot/lib/:$LD_LIBRARY_PATH
# where the collected raw data folders are
export DICE_RAW_DATASET_FOLDERS=$HOME/data/real
# where the post-processed data folders are
export DICE_DATASET_FOLDERS=$HOME/data/real_processed
# Each training session will create a folder here.
export DICE_CHECKPOINT_FOLDERS=$HOME/training_outputs
# Hardware interfaces root.
export DICE_HARDWARE_INTERFACES_ROOT=$HOME/DICE-RL-Robot/hardware_interfaces
# Hardware configs.
export DICE_HARDWARE_CONFIG_FOLDERS=$DICE_HARDWARE_INTERFACES_ROOT/workcell/table_top_manip/config
# Logging folder.
export DICE_CONTROL_LOG_FOLDERS=$HOME/data/control_log
conda activate dice-rl-robot
