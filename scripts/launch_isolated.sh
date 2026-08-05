#!/bin/bash
# CPU-isolated launcher for robometer-mode RL finetuning.
#
# WHY: the robometer reward server is a 4B model. Its inference spikes (31
# threads, all cores) + the learner's training threads oversubscribe the 24
# cores and starve the env_runner's 30Hz real-time control loop. That latency
# makes CAN frames miss their deadline -> motor "loss communication", and also
# freezes the desktop (gnome/vscode) during the spike.
#
# FIX: pin each process to disjoint cores and de-prioritize the heavy ones, so
# the real-time control loop always has dedicated cores.
#
#   Core allocation (24 cores):
#     0-5    env_runner        real-time control + cameras  (isolated, normal prio)
#     6-23   robometer + learner   heavy compute            (niced down)
#
# Plus thread caps so torch/openmp don't each spawn 24 threads.
#
# Usage (three separate terminals):
#   bash scripts/launch_isolated.sh server      # robometer eval server
#   bash scripts/launch_isolated.sh learner     # DICE-RL learner
#   bash scripts/launch_isolated.sh envrunner   # robot env runner
#   bash scripts/launch_isolated.sh eval        # interactive checkpoint eval
#
# If the robometer server is ALREADY running, you don't need to restart it —
# just rebind it live (this script's `rebind <PID>` does that):
#   bash scripts/launch_isolated.sh rebind 8043

set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROBOMETER_DIR="$HOME/Documents/niu/Robometer"

ENV_CORES="0-5"
HEAVY_CORES="6-23"

ROLE="$1"
shift || true
case "$ROLE" in
  server)
    echo "[isolated] robometer server → cores $HEAVY_CORES, nice +10, 8 threads"
    cd "$ROBOMETER_DIR"
    exec env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
      taskset -c "$HEAVY_CORES" nice -n 10 \
      uv run python robometer/evals/eval_server.py \
        model_path=robometer/Robometer-4B \
        server_url=0.0.0.0 server_port=8000 num_gpus=1 batch_size=4
    ;;

  learner)
    echo "[isolated] learner → cores $HEAVY_CORES, nice +5, 8 threads"
    cd "$HERE"
    source ./prepare.sh
    RUN_NAME="$(python -c 'from dice_rl.config.yam_rl_config import RUN_NAME; print(RUN_NAME)')"
    RUN_DIR="$HOME/training_outputs/yam_rl_finetuning_${RUN_NAME}"
    mkdir -p "$RUN_DIR"
    exec env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
      MALLOC_ARENA_MAX=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      taskset -c "$HEAVY_CORES" nice -n 5 \
      python scripts/yam_rl_run_learner.py 2>&1 \
        | tee -a "$RUN_DIR/learner.log"
    ;;

  envrunner)
    echo "[isolated] env_runner → cores $ENV_CORES (dedicated, real-time control protected)"
    cd "$HERE"
    source ./prepare.sh
    exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      OPENCV_FOR_THREADS_NUM=1 PYTHONUNBUFFERED=1 \
      taskset -c "$ENV_CORES" python scripts/yam_rl_run_env_runner.py "$@"
    ;;

  eval)
    echo "[isolated] checkpoint eval → cores $ENV_CORES (dedicated robot control)"
    cd "$HERE"
    source ./prepare.sh
    exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      OPENCV_FOR_THREADS_NUM=1 PYTHONUNBUFFERED=1 \
      taskset -c "$ENV_CORES" python scripts/eval_ckpt.py "$@"
    ;;

  rebind)
    PID="$2"
    [ -z "$PID" ] && { echo "usage: $0 rebind <PID>"; exit 1; }
    echo "[isolated] rebinding running PID $PID → cores $HEAVY_CORES, nice +10"
    taskset -acp "$HEAVY_CORES" "$PID"
    renice -n 10 -p "$PID" || true
    taskset -cp "$PID"
    ;;

  *)
    echo "usage: $0 {server|learner|envrunner|eval|rebind <PID>}"
    exit 1
    ;;
esac
