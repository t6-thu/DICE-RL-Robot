#!/usr/bin/env bash
# Initialise a clean Robometer Hanoi run from exactly the prior 20 warmups.
# Uses hard links, so it does not duplicate several GiB of images.

set -euo pipefail

data_root="${DICE_DATASET_FOLDERS:-$HOME/文档/data/real_processed}"
ckpt_root="${DICE_CHECKPOINT_FOLDERS:-$HOME/training_outputs}"
run_name="${YAM_RUN_NAME:-hanoi_robometer_progress_v1}"
source_dir="${YAM_WARMUP_POOL:-$data_root/yam_rl_rollouts_hanoi_hire_all_online_fixed_w1_causal_v4}"
data_dir="$data_root/yam_rl_rollouts_$run_name"
ckpt_dir="$ckpt_root/yam_rl_finetuning_$run_name"
min_free_gb="${YAM_MIN_FREE_GB:-15}"

available_gb=$(df -Pk "$data_root" | awk 'NR==2 {printf "%d", $4/1024/1024}')
if (( available_gb < min_free_gb )); then
  echo "Only ${available_gb}GB free; require at least ${min_free_gb}GB before a real-robot run." >&2
  echo "No directories were created. Free disk space first, then run setup again." >&2
  exit 1
fi

for path in "$data_dir" "$ckpt_dir"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite existing path: $path" >&2
    exit 1
  fi
done
for i in $(seq 0 19); do
  source_file=$(printf '%s/episode_%04d.npz' "$source_dir" "$i")
  [[ -f "$source_file" ]] || { echo "Missing warmup episode: $source_file" >&2; exit 1; }
done

mkdir -p "$data_dir" "$ckpt_dir"
for i in $(seq 0 19); do
  ln "$(printf '%s/episode_%04d.npz' "$source_dir" "$i")" "$data_dir/"
done

python - "$data_dir" <<'PY'
from pathlib import Path
import sys
import numpy as np
p = Path(sys.argv[1])
files = [p / f"episode_{i:04d}.npz" for i in range(20)]
success, orders = 0, set()
for f in files:
    with np.load(f, allow_pickle=False) as d:
        success += bool(d["rewards"][-1] > 0.5)
        orders.add(str(d["policy_camera_order"].item()))
print(f"Hard-linked {len(files)} warmups: {success}/{len(files)} success; order={sorted(orders)}")
PY
echo "Fresh Robometer run initialised; no RL actor checkpoint was copied."
