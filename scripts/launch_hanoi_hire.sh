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

# This launcher is specifically for HiRE. Do not inherit a stale
# YAM_REWARD_MODE=robometer from an old terminal session.
export YAM_REWARD_MODE="hire"
export YAM_RUN_NAME="${YAM_HANOI_RUN_NAME:-hire_stack_green_hanoi_cube_centercrop_epoch0500}"
export YAM_BC_POLICY_CKPT="${YAM_HANOI_BC_POLICY_CKPT:-$HOME/training_outputs/stack_green_hanoi_cube_dp_centercrop}"
export YAM_EXPERT_NPZ="${YAM_HANOI_EXPERT_NPZ:-$HOME/data/real_processed/stack_green_hanoi_cube_224/train.npz}"
export YAM_NORM_NPZ="${YAM_HANOI_NORM_NPZ:-$HOME/data/real_processed/stack_green_hanoi_cube_224/normalization.npz}"
# Env runner counts diffusion-query chunks, not 30 Hz frames. Each chunk can
# execute fewer than 16 waypoints when inference latency causes skipped steps,
# so keep this comfortably above the demo horizon.
export YAM_MAX_EPISODE_STEPS="${YAM_MAX_EPISODE_STEPS:-80}"
: "${YAM_HIRE_EXPERT_CURATION_PATH:=}"
export YAM_HIRE_EXPERT_CURATION_PATH

for required in "$YAM_EXPERT_NPZ" "$YAM_NORM_NPZ"; do
  if [ ! -f "$required" ]; then
    echo "[hanoi-hire] missing required file: $required"
    echo "[hanoi-hire] generate data with:"
    echo "  .venv/bin/python scripts/process_hanoi_hdf5_to_npz.py"
    exit 1
  fi
done
if [ ! -f "$YAM_BC_POLICY_CKPT" ] && [ ! -f "$YAM_BC_POLICY_CKPT/checkpoints/latest.ckpt" ]; then
  echo "[hanoi-hire] missing center-crop BC checkpoint: $YAM_BC_POLICY_CKPT"
  echo "[hanoi-hire] retrain Hanoi DP with the fixed image preprocessing before RL."
  exit 1
fi

echo "[hanoi-hire] reward=$YAM_REWARD_MODE run=$YAM_RUN_NAME"
echo "[hanoi-hire] bc=$YAM_BC_POLICY_CKPT"
echo "[hanoi-hire] expert=$YAM_EXPERT_NPZ"
echo "[hanoi-hire] norm=$YAM_NORM_NPZ"
echo "[hanoi-hire] max_episode_steps=$YAM_MAX_EPISODE_STEPS"

exec bash "$HERE/scripts/launch_isolated.sh" "$ROLE" "$@"
