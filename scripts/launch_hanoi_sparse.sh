#!/usr/bin/env bash
# Pure terminal-sparse DICE-RL baseline for the Hanoi diffusion policy.
#
# This launcher is intentionally self-contained: learner and env receive the
# exact same policy/data/run settings.  It does not reuse any prior RL actor.
#
# Usage (after activating .venv):
#   bash scripts/launch_hanoi_sparse.sh setup
#   bash scripts/launch_hanoi_sparse.sh learner
#   bash scripts/launch_hanoi_sparse.sh env

set -euo pipefail
cd "$(dirname "$0")/.."

python_bin="${YAM_PYTHON_BIN:-$PWD/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Python executable not found: $python_bin" >&2
  echo "Set YAM_PYTHON_BIN or create this repository's .venv." >&2
  exit 1
fi

export DICE_DATASET_FOLDERS="${DICE_DATASET_FOLDERS:-$HOME/文档/data/real_processed}"
export DICE_CHECKPOINT_FOLDERS="${DICE_CHECKPOINT_FOLDERS:-$HOME/training_outputs}"
export YAM_RUN_NAME="${YAM_RUN_NAME:-hanoi_sparse_terminal_v1}"

# Explicitly select terminal sparse reward: r_T=1 for a manually-labelled
# success, otherwise r=0.  No HiRE/Robometer reward source is enabled.
export YAM_REWARD_MODE=sparse

# Same Hanoi base policy and recovered expert/norm statistics as the recent
# HiRE runs.  latest.ckpt is the trained DP base policy, not an RL checkpoint.
export YAM_BC_POLICY_CKPT="${YAM_BC_POLICY_CKPT:-$HOME/training_outputs/stack_green_hanoi_cube_dp_npz_retrain/checkpoints/latest.ckpt}"
export YAM_EXPERT_NPZ="${YAM_EXPERT_NPZ:-$DICE_DATASET_FOLDERS/stack_green_hanoi_cube_224_recovered_success/train.npz}"
export YAM_NORM_NPZ="${YAM_NORM_NPZ:-$DICE_DATASET_FOLDERS/stack_green_hanoi_cube_224_recovered_success/normalization.npz}"

# Physical identity: 271309=wrist, 278369=base.  The Hanoi DP was trained
# with rgb_0=wrist and rgb_1=base, therefore this order is mandatory.
export YAM_WRIST_CAM_SERIAL="${YAM_WRIST_CAM_SERIAL:-218622271309}"
export YAM_BASE_CAM_SERIAL="${YAM_BASE_CAM_SERIAL:-218622278369}"
export YAM_POLICY_CAMERA_ORDER="${YAM_POLICY_CAMERA_ORDER:-wrist_base}"
export YAM_CAN_CHANNEL="${YAM_CAN_CHANNEL:-can_follower_l}"
export YAM_MAX_EPISODE_STEPS="${YAM_MAX_EPISODE_STEPS:-60}"
export YAM_CONTROL_HZ="${YAM_CONTROL_HZ:-30}"

_cmd="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "$_cmd" in
  setup)   exec bash scripts/setup_hanoi_sparse_run.sh ;;
  learner) exec "$python_bin" scripts/yam_rl_run_learner.py ;;
  env)     exec "$python_bin" scripts/yam_rl_run_env_runner.py ;;
  eval)    exec "$python_bin" scripts/eval_ckpt.py "$@" ;;
  help|-h|--help)
    cat <<EOF
Hanoi pure-sparse finetuning
  run name:   ${YAM_RUN_NAME}
  rollout:    ${DICE_DATASET_FOLDERS}/yam_rl_rollouts_${YAM_RUN_NAME}
  checkpoints:${DICE_CHECKPOINT_FOLDERS}/yam_rl_finetuning_${YAM_RUN_NAME}

1) bash scripts/launch_hanoi_sparse.sh setup
2) bash scripts/launch_hanoi_sparse.sh learner
   Wait for "Saved RL checkpoint: ...checkpoint_002000.pt".
3) bash scripts/launch_hanoi_sparse.sh env
   It will resume at Episode 21 and load latest_actor.pt before the first rollout.

Evaluation (interactive checkpoint picker; 15 episodes):
   bash scripts/launch_hanoi_sparse.sh eval --num-episodes 15
EOF
    ;;
  *) echo "usage: $0 {setup|learner|env|eval}" >&2; exit 2 ;;
esac
