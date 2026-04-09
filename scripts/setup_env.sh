#!/bin/bash
# Setup script for DICE-RL-Robot
# Usage: source scripts/setup_env.sh

set -e

echo "=== DICE-RL-Robot Environment Setup ==="

# Create conda environment
echo "Creating conda environment..."
conda env create -f environment.yaml

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate dice-rl-robot"
echo "Then install: pip install -e '.[learner]'  (for GPU server)"
echo "         or:  pip install -e '.[robot]'    (for robot PC)"
echo ""
echo "Required environment variables:"
echo "  export DICE_HARDWARE_CONFIG_FOLDERS=/path/to/hardware/configs"
echo "  export DICE_DATASET_FOLDERS=/path/to/datasets"
echo "  export DICE_CHECKPOINT_FOLDERS=/path/to/checkpoints"
