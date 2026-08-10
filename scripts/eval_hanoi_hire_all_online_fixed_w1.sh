#!/bin/bash
# Safely evaluate checkpoints from the all-online-HiRE, fixed-w=1 Hanoi run.

set -e

HERE="$(cd "$(dirname "$0")/.." && pwd)"

# The learner holds most of the GPU and the training env runner owns the CAN
# robot. Checkpoint evaluation must have both resources to itself.
learner_pids="$(pgrep -f '[y]am_rl_run_learner\.py' || true)"
runner_pids="$(pgrep -f '[y]am_rl_run_env_runner\.py' || true)"
if [ -n "$learner_pids" ] || [ -n "$runner_pids" ]; then
  echo "[hanoi-eval-fixed-w1] refusing to start while training processes are alive"
  [ -n "$learner_pids" ] && echo "[hanoi-eval-fixed-w1] learner pid(s): $learner_pids"
  [ -n "$runner_pids" ] && echo "[hanoi-eval-fixed-w1] envrunner pid(s): $runner_pids"
  echo "[hanoi-eval-fixed-w1] stop them cleanly before evaluation"
  exit 1
fi

if ! ip -brief link show can_follower_l 2>/dev/null | grep -q 'UP'; then
  echo "[hanoi-eval-fixed-w1] can_follower_l is not UP"
  echo "[hanoi-eval-fixed-w1] run: sudo ip link set can_follower_l up type can bitrate 1000000"
  exit 1
fi

eval_run="${YAM_HANOI_FIXED_W1_RUN_NAME:-hanoi_hire_all_online_fixed_w1_recovered_success_v3}"
echo "[hanoi-eval-fixed-w1] run=$eval_run"
echo "[hanoi-eval-fixed-w1] protocol: raw BC+actor policy, 15 saved episodes/checkpoint, 60 chunks/episode"

exec bash "$HERE/scripts/launch_hanoi_hire_all_online_fixed_w1.sh" eval \
  --num-episodes 15 \
  --max-episode-steps 60 \
  --raw-policy \
  "$@"
