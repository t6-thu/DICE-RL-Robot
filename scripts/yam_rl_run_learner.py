#!/usr/bin/env python3
"""Launch the DICE-RL learner for YAM.

    . ./prepare.sh
    python scripts/yam_rl_run_learner.py
"""
import logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(name)s %(levelname)s] %(message)s")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dice_rl.config.yam_rl_config import (
    BC_POLICY_CKPT, EXPERT_NPZ, ONLINE_DATA_DIR, NORM_NPZ, RL_CKPT_DIR,
    HIRE_INIT_DIR,
    TRAINING, NETWORK, COMM,
)
from dice_rl.learner.yam_rl_learner import YAMRLLearner

learner = YAMRLLearner(
    pretrained_policy_ckpt = BC_POLICY_CKPT,
    expert_npz_path        = EXPERT_NPZ,
    online_data_dir        = ONLINE_DATA_DIR,
    rl_checkpoint_dir      = RL_CKPT_DIR,
    hire_init_dir          = HIRE_INIT_DIR,
    **{**TRAINING, **NETWORK, **COMM},
)
learner.run()
