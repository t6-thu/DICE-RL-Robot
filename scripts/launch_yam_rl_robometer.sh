#!/usr/bin/env bash
# YAM RL finetune with Robometer-only dense reward (HiRE-Dice_RL launch analogue).
#
# Reward recipe (online rollouts):
#   r_t = γ·Φ(s_{t+H}) − Φ(s_t),  Φ = robometer_reward_weight · progress(prefix)
#   No sparse success/failure term in online transitions.
# Expert demos in RLPD stay sparse (terminal +1 only), like
#   expert_dataset.use_env_rewards_only=true in HiRE-Dice robometer runs.
#
# Prerequisites:
#   . ./prepare.sh
#   bash scripts/robometer/start_eval_server.sh   # port 8000
#
# Usage:
#   bash scripts/launch_yam_rl_robometer.sh              # print instructions
#   bash scripts/launch_yam_rl_robometer.sh learner      # start learner only
#   bash scripts/launch_yam_rl_robometer.sh env          # start env runner only
#
# Optional env overrides:
#   ROBOMETER_SERVER_URL=http://127.0.0.1:8000
#   ROBOMETER_REWARD_WEIGHT=1.0
#   YAM_RUN_NAME=robometer_libero_w1

set -euo pipefail
cd "$(dirname "$0")/.."

export YAM_REWARD_MODE=robometer
export ROBOMETER_SERVER_URL="${ROBOMETER_SERVER_URL:-http://127.0.0.1:8000}"
export ROBOMETER_REWARD_WEIGHT="${ROBOMETER_REWARD_WEIGHT:-1.0}"
# Defaults aligned with common HiRE-Dice FINETUNE_GLOBAL_OVERRIDES
export YAM_ROBOMETER_QUERY_EVERY_N_CHUNKS="${YAM_ROBOMETER_QUERY_EVERY_N_CHUNKS:-4}"
export YAM_ROBOMETER_QUERY_FILL_MODE="${YAM_ROBOMETER_QUERY_FILL_MODE:-hold}"
export YAM_ROBOMETER_MAX_BATCH_SIZE="${YAM_ROBOMETER_MAX_BATCH_SIZE:-4}"
export YAM_ROBOMETER_CAMERA="${YAM_ROBOMETER_CAMERA:-base}"

if [[ -n "${YAM_RUN_NAME:-}" ]]; then
  export YAM_RUN_NAME
fi

_cmd="${1:-help}"

_run_learner() {
  echo "Starting YAM learner (Robometer-only reward, server=${ROBOMETER_SERVER_URL})"
  python scripts/yam_rl_run_learner.py
}

_run_env() {
  echo "Starting YAM env runner"
  python scripts/yam_rl_run_env_runner.py
}

case "${_cmd}" in
  learner) _run_learner ;;
  env)     _run_env ;;
  help|*)
    cat <<EOF
YAM Robometer RL finetune

1) Start eval server (separate terminal):
     bash scripts/robometer/start_eval_server.sh

2) Optional: fresh run with warmup pool:
     # set RUN_NAME in dice_rl/config/yam_rl_config.py or:
     YAM_RUN_NAME=robometer_libero_w1 bash scripts/setup_new_run.sh

3) Learner (GPU):
     bash scripts/launch_yam_rl_robometer.sh learner

4) Env runner (robot), after learner synced warmup episodes:
     bash scripts/launch_yam_rl_robometer.sh env

Config toggles (yam_rl_config.py TRAINING dict) are overridden by YAM_REWARD_MODE=robometer.
Tune weight: ROBOMETER_REWARD_WEIGHT=0.5 bash scripts/launch_yam_rl_robometer.sh learner
EOF
    ;;
esac
