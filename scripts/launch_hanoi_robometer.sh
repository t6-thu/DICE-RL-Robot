#!/usr/bin/env bash
# Robometer-4B progress-reward baseline for Hanoi:
#   "Move the green disk from the rightmost rod to the middle rod."
#
# Commands:
#   bash scripts/launch_hanoi_robometer.sh setup
#   bash scripts/launch_hanoi_robometer.sh server
#   bash scripts/launch_hanoi_robometer.sh learner
#   bash scripts/launch_hanoi_robometer.sh env

set -euo pipefail
cd "$(dirname "$0")/.."

python_bin="$PWD/.venv/bin/python"
[[ -x "$python_bin" ]] || { echo "Missing $python_bin" >&2; exit 1; }

export DICE_DATASET_FOLDERS="${DICE_DATASET_FOLDERS:-$HOME/文档/data/real_processed}"
export DICE_CHECKPOINT_FOLDERS="${DICE_CHECKPOINT_FOLDERS:-$HOME/training_outputs}"
export YAM_RUN_NAME="${YAM_RUN_NAME:-hanoi_robometer_progress_v1}"
export YAM_REWARD_MODE=robometer
export YAM_WARMUP_POOL="${YAM_WARMUP_POOL:-$DICE_DATASET_FOLDERS/yam_rl_rollouts_hanoi_hire_all_online_fixed_w1_causal_v4}"

export YAM_BC_POLICY_CKPT="${YAM_BC_POLICY_CKPT:-$HOME/training_outputs/stack_green_hanoi_cube_dp_npz_retrain/checkpoints/latest.ckpt}"
export YAM_EXPERT_NPZ="${YAM_EXPERT_NPZ:-$DICE_DATASET_FOLDERS/stack_green_hanoi_cube_224_recovered_success/train.npz}"
export YAM_NORM_NPZ="${YAM_NORM_NPZ:-$DICE_DATASET_FOLDERS/stack_green_hanoi_cube_224_recovered_success/normalization.npz}"

# The Hanoi policy was trained with rgb_0=wrist and rgb_1=base.  Robometer
# still receives the physical base view, selected below after channel mapping.
export YAM_WRIST_CAM_SERIAL="${YAM_WRIST_CAM_SERIAL:-218622271309}"
export YAM_BASE_CAM_SERIAL="${YAM_BASE_CAM_SERIAL:-218622278369}"
export YAM_POLICY_CAMERA_ORDER="${YAM_POLICY_CAMERA_ORDER:-wrist_base}"
export YAM_CAN_CHANNEL="${YAM_CAN_CHANNEL:-can_follower_l}"
export YAM_CONTROL_HZ="${YAM_CONTROL_HZ:-30}"
export YAM_MAX_EPISODE_STEPS="${YAM_MAX_EPISODE_STEPS:-60}"

export ROBOMETER_DIR="${ROBOMETER_DIR:-$HOME/文档/Robometer}"
export ROBOMETER_SERVER_URL="${ROBOMETER_SERVER_URL:-http://127.0.0.1:8000}"
export YAM_ROBOMETER_TASK_INSTRUCTION="${YAM_ROBOMETER_TASK_INSTRUCTION:-Move the green disk from the rightmost rod to the middle rod.}"
export YAM_ROBOMETER_CAMERA="${YAM_ROBOMETER_CAMERA:-base}"
export ROBOMETER_REWARD_WEIGHT="${ROBOMETER_REWARD_WEIGHT:-1.0}"
export YAM_ROBOMETER_QUERY_EVERY_N_CHUNKS="${YAM_ROBOMETER_QUERY_EVERY_N_CHUNKS:-4}"
export YAM_ROBOMETER_QUERY_FILL_MODE="${YAM_ROBOMETER_QUERY_FILL_MODE:-hold}"
export YAM_ROBOMETER_MAX_BATCH_SIZE="${YAM_ROBOMETER_MAX_BATCH_SIZE:-1}"
export YAM_ROBOMETER_AUTO_START_SERVER="${YAM_ROBOMETER_AUTO_START_SERVER:-1}"
export YAM_ROBOMETER_STOP_SERVER_AFTER_REWARD="${YAM_ROBOMETER_STOP_SERVER_AFTER_REWARD:-1}"
export YAM_ROBOMETER_SERVER_LAUNCH_CMD="${YAM_ROBOMETER_SERVER_LAUNCH_CMD:-ROBOMETER_DIR=$ROBOMETER_DIR bash scripts/launch_isolated.sh server}"
export YAM_ROBOMETER_SERVER_LOG_PATH="${YAM_ROBOMETER_SERVER_LOG_PATH:-$DICE_CHECKPOINT_FOLDERS/yam_rl_finetuning_$YAM_RUN_NAME/robometer_server.log}"

# Preserve the requested deployment/training diffusion count.
export YAM_LEARNER_BC_POOL_INFERENCE_STEPS="${YAM_LEARNER_BC_POOL_INFERENCE_STEPS:-16}"

cmd="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$cmd" in
  setup)   exec bash scripts/setup_hanoi_robometer_run.sh ;;
  server)  exec bash scripts/launch_isolated.sh server ;;
  learner) exec bash scripts/launch_isolated.sh learner "$@" ;;
  env)     exec bash scripts/launch_isolated.sh envrunner "$@" ;;
  eval)    exec "$python_bin" scripts/eval_ckpt.py "$@" ;;
  smoke)   exec "$python_bin" scripts/test_robometer_offline.py --episode "$YAM_WARMUP_POOL/episode_0000.npz" "$@" ;;
  health)
    exec "$python_bin" -c 'from dice_rl.reward.robometer_client import health_check; import os; u=os.environ["ROBOMETER_SERVER_URL"]; print(f"{u} healthy={health_check(u)}")'
    ;;
  help|-h|--help)
    cat <<EOF
Hanoi Robometer baseline (progress-only online reward; expert remains sparse)
  run:    $YAM_RUN_NAME
  task:   $YAM_ROBOMETER_TASK_INSTRUCTION
  policy: rgb_0=wrist, rgb_1=base; Robometer camera=physical base

First free at least 15GB, then:
  bash scripts/launch_hanoi_robometer.sh setup
  bash scripts/launch_hanoi_robometer.sh server
  bash scripts/launch_hanoi_robometer.sh health
  bash scripts/launch_hanoi_robometer.sh smoke
  # Ctrl-C stops the manual server after the health test.

Recommended single-4090 staged loop (prevents learner, VLM, and robot policy
from competing for GPU memory):
  # Scores the 20 warmups, trains 2,000 steps, writes latest_actor.pt, exits.
  bash scripts/launch_hanoi_robometer.sh learner --max-training-rounds 1

Evaluate any baseline checkpoint interactively (BC / 2000 / 3000 / …):
  bash scripts/launch_hanoi_robometer.sh eval
  # Loads latest_actor.pt, collects exactly episodes 21-30, then exits.
  bash scripts/launch_hanoi_robometer.sh env --max-episodes 10
  # Scores only uncached episodes, trains the next 1,000 steps, then exits.
  bash scripts/launch_hanoi_robometer.sh learner --max-training-rounds 1

The learner resumes checkpoint_002000.pt automatically on each later call.
It starts/stops the server only while assigning deferred rewards.
EOF
    ;;
  *) echo "usage: $0 {setup|server|health|smoke|learner|env|eval}" >&2; exit 2 ;;
esac
