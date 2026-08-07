#!/bin/bash
# Fresh Hanoi finetuning recipe:
#   offline expert       -> sparse terminal reward
#   online success       -> sparse terminal reward
#   online failure       -> HiRE-shaped reward
#
# Use this wrapper for BOTH learner and envrunner so they share one run name.

set -e

HERE="$(cd "$(dirname "$0")/.." && pwd)"

# This recovery run uses successful online rollouts as replacement expert data;
# keep both its dataset and run name explicit so it cannot be mistaken for the
# lost original teleoperation dataset.
RECOVERED_DATA_DIR="${YAM_HANOI_RECOVERED_DATA_DIR:-$HOME/文档/data/real_processed/stack_green_hanoi_cube_224_recovered_success}"
export YAM_HANOI_EXPERT_NPZ="${YAM_HANOI_RECOVERED_EXPERT_NPZ:-$RECOVERED_DATA_DIR/train.npz}"
export YAM_HANOI_NORM_NPZ="${YAM_HANOI_RECOVERED_NORM_NPZ:-$RECOVERED_DATA_DIR/normalization.npz}"

# A distinct default run name guarantees that launch_isolated.sh does not
# restore checkpoints or rollout episodes from the earlier HiRE run.
export YAM_HANOI_RUN_NAME="${YAM_HANOI_RECOVERED_RUN_NAME:-hanoi_hire_sparse_online_success_recovered_success_v1}"
export YAM_HIRE_SPARSE_ONLINE_SUCCESS="1"

echo "[hanoi-sparse-success] online_success=sparse online_failure=hire run=$YAM_HANOI_RUN_NAME"
exec bash "$HERE/scripts/launch_hanoi_hire.sh" "$@"
