#!/usr/bin/env bash
# Create a fresh Hanoi sparse-reward run from exactly episodes 0000..0019.
# It deliberately refuses to overwrite anything.

set -euo pipefail

data_root="${DICE_DATASET_FOLDERS:-$HOME/文档/data/real_processed}"
ckpt_root="${DICE_CHECKPOINT_FOLDERS:-$HOME/training_outputs}"
run_name="${YAM_RUN_NAME:-hanoi_sparse_terminal_v1}"
warmup_pool="${YAM_WARMUP_POOL:-$data_root/yam_rl_rollouts_hanoi_hire_all_online_fixed_w1_causal_v4}"
data_dir="$data_root/yam_rl_rollouts_$run_name"
ckpt_dir="$ckpt_root/yam_rl_finetuning_$run_name"

for path in "$data_dir" "$ckpt_dir"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite existing path: $path" >&2
    echo "Choose a new YAM_RUN_NAME; this setup script never deletes old runs." >&2
    exit 1
  fi
done

for i in $(seq 0 19); do
  source_file=$(printf '%s/episode_%04d.npz' "$warmup_pool" "$i")
  [[ -f "$source_file" ]] || { echo "Missing warmup episode: $source_file" >&2; exit 1; }
done

mkdir -p "$data_dir" "$ckpt_dir"
for i in $(seq 0 19); do
  cp "$(printf '%s/episode_%04d.npz' "$warmup_pool" "$i")" "$data_dir/"
done

python - "$data_dir" <<'PY'
from pathlib import Path
import sys
import numpy as np

p = Path(sys.argv[1])
files = [p / f"episode_{i:04d}.npz" for i in range(20)]
labels = []
orders = set()
for f in files:
    with np.load(f, allow_pickle=False) as d:
        labels.append(bool(d["rewards"][-1] > 0.5))
        if "policy_camera_order" in d:
            orders.add(str(d["policy_camera_order"].item()))
print(f"Copied {len(files)} warmup episodes: {sum(labels)}/{len(labels)} success")
print(f"policy_camera_order in copied data: {sorted(orders) or ['(not recorded)']}")
PY

echo "Fresh sparse run initialized. No previous RL checkpoint was copied."
echo "Next: bash scripts/launch_hanoi_sparse.sh learner"
