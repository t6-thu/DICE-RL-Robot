#!/bin/bash
# Hanoi HiRE RL finetuning launcher.
#
# Usage in two terminals:
#   bash scripts/launch_hanoi_hire.sh learner
#   bash scripts/launch_hanoi_hire.sh envrunner --residual-scale 0.1 --max-joint-step 0.04

set -e

HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROLE="${1:-}"
shift || true

if [ -z "$ROLE" ]; then
  echo "usage: $0 {learner|envrunner} [envrunner args...]"
  exit 1
fi

export YAM_REWARD_MODE="${YAM_REWARD_MODE:-hire}"
export YAM_RUN_NAME="${YAM_RUN_NAME:-hire_stack_green_hanoi_cube_epoch0100}"
export YAM_BC_POLICY_CKPT="${YAM_BC_POLICY_CKPT:-$HOME/training_outputs/stack_green_hanoi_cube_dp/checkpoints/epoch=0100-train_loss=0.019.ckpt}"
export YAM_EXPERT_NPZ="${YAM_EXPERT_NPZ:-$HOME/data/real_processed/stack_green_hanoi_cube_224/train.npz}"
export YAM_NORM_NPZ="${YAM_NORM_NPZ:-$HOME/data/real_processed/stack_green_hanoi_cube_224/normalization.npz}"
: "${YAM_HIRE_EXPERT_CURATION_PATH:=}"
export YAM_HIRE_EXPERT_CURATION_PATH

for required in "$YAM_BC_POLICY_CKPT" "$YAM_EXPERT_NPZ" "$YAM_NORM_NPZ"; do
  if [ ! -f "$required" ]; then
    echo "[hanoi-hire] missing required file: $required"
    echo "[hanoi-hire] generate data with:"
    echo "  .venv/bin/python scripts/process_hanoi_hdf5_to_npz.py"
    exit 1
  fi
done

echo "[hanoi-hire] reward=$YAM_REWARD_MODE run=$YAM_RUN_NAME"
echo "[hanoi-hire] bc=$YAM_BC_POLICY_CKPT"
echo "[hanoi-hire] expert=$YAM_EXPERT_NPZ"
echo "[hanoi-hire] norm=$YAM_NORM_NPZ"

exec bash "$HERE/scripts/launch_isolated.sh" "$ROLE" "$@"
