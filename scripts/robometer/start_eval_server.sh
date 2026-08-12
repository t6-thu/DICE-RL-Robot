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

ROBOMETER_ROOT="${ROBOMETER_ROOT:-$HOME/文档/Robometer}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
GPUS="${GPUS:-1}"
MODEL_PATH="${MODEL_PATH:-robometer/Robometer-4B}"

PYTHON="${ROBOMETER_PYTHON:-$ROBOMETER_ROOT/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Robometer Python not found: ${PYTHON}" >&2
  echo "Set ROBOMETER_ROOT or ROBOMETER_PYTHON." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PORT
export GPUS
export MODEL_PATH

cd "${ROBOMETER_ROOT}"
echo "Robometer server: port=${PORT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} GPUS=${GPUS}"
exec "${PYTHON}" robometer/evals/eval_server.py \
  model_path="${MODEL_PATH}" \
  server_url=0.0.0.0 \
  server_port="${PORT}" \
  num_gpus="${GPUS}"
