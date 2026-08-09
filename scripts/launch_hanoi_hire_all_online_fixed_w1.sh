#!/bin/bash
# Hanoi finetuning recipe used by the v3 run:
#   offline expert       -> sparse terminal reward
#   every online rollout -> HiRE PBRS reward (success and failure)
#   HiRE dense weight    -> fixed at w=1.0 (no success-rate decay)
#
# Use this wrapper for BOTH learner and envrunner so they resolve the same
# recovered dataset, rollout directory, and checkpoint directory.

set -e

HERE="$(cd "$(dirname "$0")/.." && pwd)"

RECOVERED_DATA_DIR="${YAM_HANOI_RECOVERED_DATA_DIR:-$HOME/文档/data/real_processed/stack_green_hanoi_cube_224_recovered_success}"
export YAM_HANOI_EXPERT_NPZ="${YAM_HANOI_RECOVERED_EXPERT_NPZ:-$RECOVERED_DATA_DIR/train.npz}"
export YAM_HANOI_NORM_NPZ="${YAM_HANOI_RECOVERED_NORM_NPZ:-$RECOVERED_DATA_DIR/normalization.npz}"

export YAM_HANOI_RUN_NAME="${YAM_HANOI_FIXED_W1_RUN_NAME:-hanoi_hire_all_online_fixed_w1_recovered_success_v3}"
export YAM_HIRE_SPARSE_ONLINE_SUCCESS="0"
export YAM_HIRE_FIXED_DENSE_WEIGHT="1.0"
# Match the deployed BC checkpoint, whose DDIM inference setting is 16.  The
# learner otherwise defaults to an 8-step shortcut while building its pool.
export YAM_LEARNER_BC_POOL_INFERENCE_STEPS="16"

echo "[hanoi-hire-fixed-w1] online_success=hire online_failure=hire fixed_dense_weight=1.0 bc_inference_steps=16 run=$YAM_HANOI_RUN_NAME"
exec bash "$HERE/scripts/launch_hanoi_hire.sh" "$@"
