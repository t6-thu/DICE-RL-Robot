#!/bin/bash
# Prepare a new RL finetuning run with the standard 20-episode warmup pool.
#
# Reads RUN_NAME from dice_rl/config/yam_rl_config.py, then:
#   1. wipes existing yam_rl_rollouts_<RUN_NAME>/ and yam_rl_finetuning_<RUN_NAME>/
#   2. recreates both dirs
#   3. copies the canonical 20 warmup episodes (episode_0000..0019) from the
#      warmup-pool source dir
#
# After this, simply start `python scripts/yam_rl_run_learner.py …` and then
# `python scripts/yam_rl_run_env_runner.py`.  The learner will auto-load the
# 20 episodes and immediately trigger Round 1 training; the env_runner will
# start recording at episode_0020 (since dir already has 20 files).
#
# Usage:
#   . ./prepare.sh
#   bash scripts/setup_new_run.sh
#
# Override warmup pool source via env var:
#   YAM_WARMUP_POOL=~/data/real_processed/yam_rl_rollouts_pos_only_no_neg \
#       bash scripts/setup_new_run.sh

set -e

# Resolve RUN_NAME from config (Python so we don't have to parse).
RUN_NAME=$(python3 -c "import sys, os
sys.path.insert(0, '$(dirname "$0")/..')
from dice_rl.config.yam_rl_config import RUN_NAME
print(RUN_NAME)")

WARMUP_POOL=${YAM_WARMUP_POOL:-$HOME/data/real_processed/yam_rl_rollouts_pos_only_no_neg}
DATA_DIR=$HOME/data/real_processed/yam_rl_rollouts_${RUN_NAME}
CKPT_DIR=$HOME/training_outputs/yam_rl_finetuning_${RUN_NAME}

echo "=========================================="
echo " RUN_NAME      = $RUN_NAME"
echo " WARMUP_POOL   = $WARMUP_POOL"
echo " DATA_DIR      = $DATA_DIR"
echo " CKPT_DIR      = $CKPT_DIR"
echo "=========================================="

# Sanity check warmup pool
n_pool=$(ls "$WARMUP_POOL"/episode_*.npz 2>/dev/null | wc -l)
if [ "$n_pool" -lt 20 ]; then
    echo "❌ Warmup pool $WARMUP_POOL has only $n_pool episodes (need ≥ 20)"
    exit 1
fi

read -p "About to WIPE $DATA_DIR and $CKPT_DIR.  Proceed? [y/N] " ans
if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then
    echo "Aborted."
    exit 1
fi

rm -rf "$DATA_DIR" "$CKPT_DIR"
mkdir -p "$DATA_DIR" "$CKPT_DIR"

cp "$WARMUP_POOL"/episode_000[0-9].npz "$WARMUP_POOL"/episode_001[0-9].npz "$DATA_DIR"/

n=$(ls "$DATA_DIR"/episode_*.npz | wc -l)
echo "✓ Copied $n warmup episodes → $DATA_DIR"

# Success rate of the pool
python3 -c "
import numpy as np, glob, os
paths = sorted(glob.glob('$DATA_DIR/episode_*.npz'))
labels = ['S' if np.load(p)['rewards'][-1] > 0.5 else 'F' for p in paths]
ns = labels.count('S')
print(f'  warmup BC baseline: {ns}/{len(paths)} = {100*ns/len(paths):.0f}%  ({\" \".join(labels)})')
"

echo ""
echo "Next steps:"
echo "  terminal 1:  python scripts/yam_rl_run_learner.py 2>&1 \\"
echo "                 | tee -a $CKPT_DIR/learner.log"
echo "  terminal 2:  python scripts/yam_rl_run_env_runner.py"
echo ""
echo "Wait for learner to print 'Synced total_episodes=20 from disk replay buffer'"
echo "BEFORE starting env_runner — otherwise env_runner caches ep=0 and"
echo "overwrites your warmup pool!"
