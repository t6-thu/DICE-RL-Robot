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

import copy

from dice_rl.config import yam_rl_config as cfg
from dice_rl.config.yam_env_overrides import (
    apply_learner_env_overrides,
    apply_robometer_env_overrides,
)
from dice_rl.config.yam_rl_config import (
    BC_POLICY_CKPT, EXPERT_NPZ, ONLINE_DATA_DIR, NORM_NPZ, RL_CKPT_DIR,
    HIRE_INIT_DIR, HIRE_EXPERT_CURATION_PATH,
    TRAINING, NETWORK, COMM,
)
from dice_rl.learner.yam_rl_learner import YAMRLLearner

# Robometer / reward toggles via env (see scripts/robometer/README.md).
_training = apply_learner_env_overrides(
    apply_robometer_env_overrides(copy.deepcopy(TRAINING))
)
_reward_mode = os.environ.get("YAM_REWARD_MODE", "").strip().lower()
if _reward_mode in ("hire", ""):
    pass
elif _reward_mode == "robometer":
    pass  # apply_robometer_env_overrides already enabled it
else:
    raise ValueError(f"Unknown YAM_REWARD_MODE={_reward_mode!r} (use 'hire' or 'robometer')")

# Optional RUN_NAME override without editing yam_rl_config.py
if os.environ.get("YAM_RUN_NAME"):
    cfg.RUN_NAME = os.environ["YAM_RUN_NAME"]
    cfg.ONLINE_DATA_DIR = os.path.join(
        os.environ.get("DICE_DATASET_FOLDERS", os.path.expanduser("~/data/real_processed")),
        f"yam_rl_rollouts_{cfg.RUN_NAME}",
    )
    cfg.RL_CKPT_DIR = os.path.join(
        os.environ.get("DICE_CHECKPOINT_FOLDERS", os.path.expanduser("~/training_outputs")),
        f"yam_rl_finetuning_{cfg.RUN_NAME}",
    )

learner = YAMRLLearner(
    pretrained_policy_ckpt    = getattr(cfg, "BC_POLICY_CKPT", BC_POLICY_CKPT),
    expert_npz_path           = getattr(cfg, "EXPERT_NPZ", EXPERT_NPZ),
    online_data_dir           = getattr(cfg, "ONLINE_DATA_DIR", ONLINE_DATA_DIR),
    rl_checkpoint_dir         = getattr(cfg, "RL_CKPT_DIR", RL_CKPT_DIR),
    hire_init_dir             = getattr(cfg, "HIRE_INIT_DIR", HIRE_INIT_DIR),
    hire_expert_curation_path = getattr(cfg, "HIRE_EXPERT_CURATION_PATH", HIRE_EXPERT_CURATION_PATH),
    expected_policy_camera_order = os.environ.get("YAM_POLICY_CAMERA_ORDER") or None,
    **{**_training, **NETWORK, **COMM},
)
learner.run()
