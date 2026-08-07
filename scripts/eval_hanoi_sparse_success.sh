#!/bin/bash
# Safely evaluate checkpoints from the recovered-success Hanoi finetuning run.

set -e

HERE="$(cd "$(dirname "$0")/.." && pwd)"

# The learner holds most of the 4090 memory and the training env runner owns
# the CAN robot. Evaluation must have both resources to itself.
learner_pids="$(pgrep -f '[y]am_rl_run_learner\.py' || true)"
runner_pids="$(pgrep -f '[y]am_rl_run_env_runner\.py' || true)"
if [ -n "$learner_pids" ] || [ -n "$runner_pids" ]; then
  echo "[hanoi-eval] refusing to start while training processes are alive"
  [ -n "$learner_pids" ] && echo "[hanoi-eval] learner pid(s): $learner_pids"
  [ -n "$runner_pids" ] && echo "[hanoi-eval] envrunner pid(s): $runner_pids"
  echo "[hanoi-eval] wait for the current checkpoint save, then stop them cleanly"
  exit 1
fi

if ! ip -brief link show can_follower_l 2>/dev/null | grep -q 'UP'; then
  echo "[hanoi-eval] can_follower_l is not UP"
  echo "[hanoi-eval] run: sudo ip link set can_follower_l up type can bitrate 1000000"
  exit 1
fi

eval_run="${YAM_HANOI_RECOVERED_RUN_NAME:-hanoi_hire_sparse_online_success_recovered_success_v1}"
echo "[hanoi-eval] run=$eval_run"
echo "[hanoi-eval] default protocol: 15 saved episodes/checkpoint, 60 chunks/episode"

exec bash "$HERE/scripts/launch_hanoi_sparse_success.sh" eval \
  --num-episodes 15 \
  --max-episode-steps 60 \
  --residual-scale 1.0 \
  --max-joint-step 0.04 \
  "$@"
