#!/usr/bin/env bash
# Start Robometer-4B eval server for DICE-RL-Robot YAM finetune (Robometer-only reward).
#
# Usage:
#   bash scripts/robometer/start_eval_server.sh
#   PORT=8000 GPU=0 bash scripts/robometer/start_eval_server.sh
#
# Then in dice_rl/config/yam_rl_config.py (or via launch script):
#   use_hire_reward=False, use_robometer_reward=True
#   robometer_server_url=http://127.0.0.1:8000
#
# Learner:
#   YAM_REWARD_MODE=robometer python scripts/yam_rl_run_learner.py

set -euo pipefail

ROBOMETER_ROOT="${ROBOMETER_ROOT:-$HOME/hanzht/robometer}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
GPUS="${GPUS:-1}"
MODEL_PATH="${MODEL_PATH:-robometer/Robometer-4B}"

CONDA_ENV="${CONDA_ENV:-}"
if [[ -z "${CONDA_ENV}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    if conda env list | awk '{print $1}' | grep -qx "robometer"; then
      CONDA_ENV="robometer"
    elif conda env list | awk '{print $1}' | grep -qx "roboreward"; then
      CONDA_ENV="roboreward"
    fi
  fi
fi
if [[ -z "${CONDA_ENV}" ]]; then
  echo "No conda env robometer/roboreward. Set ROBOMETER_ROOT and install robometer." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PORT
export GPUS
export MODEL_PATH

cd "${ROBOMETER_ROOT}"
echo "Robometer server: port=${PORT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} GPUS=${GPUS}"
PYTHON="$(command -v python)"
exec "${PYTHON}" robometer/evals/eval_server.py \
  model_path="${MODEL_PATH}" \
  server_url=0.0.0.0 \
  server_port="${PORT}" \
  num_gpus="${GPUS}"
