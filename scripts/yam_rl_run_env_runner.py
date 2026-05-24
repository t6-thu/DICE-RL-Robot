#!/usr/bin/env python3
"""Launch the DICE-RL env runner for YAM.

    sudo ip link set can_follower_l up type can bitrate 1000000
    . ./prepare.sh
    python scripts/yam_rl_run_env_runner.py
"""
import logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(name)s %(levelname)s] %(message)s")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dice_rl.config.yam_rl_config import (
    BC_POLICY_CKPT, NORM_NPZ, ONLINE_DATA_DIR, TRAINING, HARDWARE, NETWORK, COMM,
)
from dice_rl.env_runner.yam_rl_env_runner import YAMRLEnvRunner

runner = YAMRLEnvRunner(
    pretrained_policy_ckpt = BC_POLICY_CKPT,
    norm_npz_path          = NORM_NPZ,
    online_data_dir        = ONLINE_DATA_DIR,
    actor_hidden_dims      = NETWORK["actor_hidden_dims"],
    obs_horizon            = TRAINING["obs_horizon"],
    action_horizon         = TRAINING["action_horizon"],
    action_dim             = TRAINING["action_dim"],
    **HARDWARE,
    **COMM,
)
runner.run()
