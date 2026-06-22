#!/usr/bin/env bash
# Staged Robometer workflow for a single workstation:
#   1. Collect episodes with envrunner only.
#   2. Stop envrunner.
#   3. Start Robometer server + one-shot learner for reward/training.
#   4. Stop Robometer server.
#   5. Restart envrunner with the updated actor.
#
# Run from a real terminal or tmux, not from VSCode's integrated terminal.

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

CMD="${1:-run}"
shift || true

COLLECT_EPISODES="${1:-${COLLECT_EPISODES:-1}}"
TRAIN_ROUNDS="${TRAIN_ROUNDS:-1}"
COLLECT_RESIDUAL_SCALE="${COLLECT_RESIDUAL_SCALE:-0.0}"
RESUME_RESIDUAL_SCALE="${RESUME_RESIDUAL_SCALE:-0.3}"
MAX_JOINT_STEP="${MAX_JOINT_STEP:-0.05}"
SERVER_START_TIMEOUT_S="${SERVER_START_TIMEOUT_S:-240}"
STOP_SERVER_AFTER_TRAIN="${STOP_SERVER_AFTER_TRAIN:-1}"

# Conservative defaults for same-workstation Robometer runs.
export YAM_REWARD_MODE="${YAM_REWARD_MODE:-robometer}"
export ROBOMETER_BATCH_SIZE="${ROBOMETER_BATCH_SIZE:-1}"
export ROBOMETER_THREADS="${ROBOMETER_THREADS:-4}"
export LEARNER_THREADS="${LEARNER_THREADS:-4}"
export YAM_LEARNER_POOL_SIZE_LIMIT="${YAM_LEARNER_POOL_SIZE_LIMIT:-2000}"
export YAM_LEARNER_BATCH_SIZE="${YAM_LEARNER_BATCH_SIZE:-64}"
export YAM_LEARNER_K_ACTOR="${YAM_LEARNER_K_ACTOR:-1}"
export YAM_LEARNER_K_CRITIC="${YAM_LEARNER_K_CRITIC:-1}"
export YAM_LEARNER_ENCODE_BATCH_SIZE="${YAM_LEARNER_ENCODE_BATCH_SIZE:-16}"
export YAM_LEARNER_BC_POOL_INFERENCE_STEPS="${YAM_LEARNER_BC_POOL_INFERENCE_STEPS:-2}"

# In staged mode we intentionally train after each collected block. Override
# these before launching if you want the config schedule (20 warmup / 10 update).
export YAM_LEARNER_NUM_EPISODES_BEFORE_FIRST_TRAINING="${YAM_LEARNER_NUM_EPISODES_BEFORE_FIRST_TRAINING:-1}"
export YAM_LEARNER_UPDATE_EVERY_X_EPISODE="${YAM_LEARNER_UPDATE_EVERY_X_EPISODE:-1}"

SERVER_PID=""
SERVER_STARTED=0
SERVER_LOG="${SERVER_LOG:-$HOME/training_outputs/robometer_server_staged.log}"

log() {
  printf '[staged] %s\n' "$*"
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/robometer_staged_workflow.sh run [episodes]
  bash scripts/robometer_staged_workflow.sh collect [episodes]
  bash scripts/robometer_staged_workflow.sh train
  bash scripts/robometer_staged_workflow.sh resume
  bash scripts/robometer_staged_workflow.sh stop-heavy

Useful env vars:
  COLLECT_EPISODES=2
  TRAIN_ROUNDS=1
  COLLECT_RESIDUAL_SCALE=0.0
  RESUME_RESIDUAL_SCALE=0.3
  MAX_JOINT_STEP=0.05
  ROBOMETER_BATCH_SIZE=1
  YAM_LEARNER_BATCH_SIZE=64
  YAM_LEARNER_POOL_SIZE_LIMIT=2000
EOF
}

confirm() {
  local msg="$1"
  if [[ "${AUTO_YES:-0}" == "1" ]]; then
    log "$msg"
    return 0
  fi
  printf '\n%s\nPress Enter to continue, or Ctrl-C to stop. ' "$msg"
  read -r _
}

print_resources() {
  log "Current memory:"
  free -h || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "Current GPU:"
    nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu \
      --format=csv,noheader || true
  fi
}

heavy_processes() {
  pgrep -af 'yam_rl_run_learner.py|robometer/evals/eval_server.py' || true
}

stop_heavy() {
  log "Stopping existing learner / Robometer server processes if present..."
  pkill -TERM -f 'yam_rl_run_learner.py' 2>/dev/null || true
  pkill -TERM -f 'robometer/evals/eval_server.py' 2>/dev/null || true
  sleep 3
  pkill -KILL -f 'yam_rl_run_learner.py' 2>/dev/null || true
  pkill -KILL -f 'robometer/evals/eval_server.py' 2>/dev/null || true
}

ensure_no_heavy_for_collect() {
  local procs
  procs="$(heavy_processes)"
  if [[ -z "$procs" ]]; then
    return 0
  fi
  log "Existing heavy process(es) detected:"
  printf '%s\n' "$procs"
  if [[ "${STOP_EXISTING_HEAVY:-0}" == "1" ]]; then
    stop_heavy
    return 0
  fi
  log "Refusing to collect while learner/server are running. Set STOP_EXISTING_HEAVY=1 to stop them."
  exit 1
}

collect() {
  local episodes="${1:-$COLLECT_EPISODES}"
  ensure_no_heavy_for_collect
  print_resources
  confirm "Stage 1: envrunner will collect ${episodes} episode(s) with learner/server stopped. Keep the robot workspace clear."
  bash scripts/launch_isolated.sh envrunner \
    --residual-scale "$COLLECT_RESIDUAL_SCALE" \
    --max-joint-step "$MAX_JOINT_STEP" \
    --max-episodes "$episodes"
}

server_healthy() {
  curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1
}

start_server_if_needed() {
  if server_healthy; then
    log "Robometer server is already healthy on port 8000; reusing it."
    SERVER_STARTED=0
    return 0
  fi

  mkdir -p "$(dirname "$SERVER_LOG")"
  log "Starting Robometer server. Log: $SERVER_LOG"
  setsid bash scripts/launch_isolated.sh server >"$SERVER_LOG" 2>&1 &
  SERVER_PID="$!"
  SERVER_STARTED=1

  local deadline=$((SECONDS + SERVER_START_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if server_healthy; then
      log "Robometer server is healthy."
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "Robometer server exited during startup. Last log lines:"
      tail -80 "$SERVER_LOG" || true
      exit 1
    fi
    sleep 2
  done

  log "Timed out waiting for Robometer server. Last log lines:"
  tail -80 "$SERVER_LOG" || true
  exit 1
}

stop_started_server() {
  if [[ "$SERVER_STARTED" != "1" || -z "$SERVER_PID" ]]; then
    return 0
  fi
  if [[ "$STOP_SERVER_AFTER_TRAIN" != "1" ]]; then
    log "Leaving Robometer server running because STOP_SERVER_AFTER_TRAIN=0."
    return 0
  fi
  log "Stopping Robometer server pid=$SERVER_PID"
  kill -TERM "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
  sleep 5
  kill -KILL "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
}

cleanup() {
  stop_started_server
}
trap cleanup EXIT

train_once() {
  print_resources
  confirm "Stage 2: Robometer server + learner will run reward assignment and ${TRAIN_ROUNDS} training round(s)."
  start_server_if_needed
  log "Starting one-shot learner with conservative resource knobs."
  bash scripts/launch_isolated.sh learner \
    --max-training-rounds "$TRAIN_ROUNDS" \
    --exit-when-idle
  stop_started_server
  SERVER_STARTED=0
  SERVER_PID=""
}

resume_envrunner() {
  print_resources
  confirm "Stage 3: envrunner will restart with latest_actor.pt if available. Keep the robot workspace clear."
  bash scripts/launch_isolated.sh envrunner \
    --residual-scale "$RESUME_RESIDUAL_SCALE" \
    --max-joint-step "$MAX_JOINT_STEP"
}

case "$CMD" in
  run)
    collect "$COLLECT_EPISODES"
    train_once
    resume_envrunner
    ;;
  collect)
    collect "$COLLECT_EPISODES"
    ;;
  train)
    train_once
    ;;
  resume)
    resume_envrunner
    ;;
  stop-heavy)
    stop_heavy
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
