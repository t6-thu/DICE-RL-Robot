#!/usr/bin/env python3
"""Launch the DICE-RL env runner for YAM.

    sudo ip link set can_follower_l up type can bitrate 1000000
    . ./prepare.sh
    python scripts/yam_rl_run_env_runner.py
"""
import logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(name)s %(levelname)s] %(message)s")

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dice_rl.config.yam_env_overrides import apply_hardware_env_overrides
from dice_rl.config.yam_rl_config import (
    BC_POLICY_CKPT, NORM_NPZ, ONLINE_DATA_DIR, RL_CKPT_DIR,
    TRAINING, HARDWARE, NETWORK, COMM,
)
from dice_rl.env_runner.yam_rl_env_runner import YAMRLEnvRunner

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--residual-scale", type=float, default=1.0,
                help="Scale on the RL residual (0=pure BC, 1=full RL). "
                     "Use 0.0 to A/B-test pure BC, or 0.3 for a softer RL effect.")
_p.add_argument("--max-joint-step", type=float, default=0.08,
                help="Max absolute change per 30 Hz command per joint/gripper. "
                     "Use smaller values (0.04-0.06) when debugging motor loss.")
_p.add_argument("--max-episodes", type=int, default=None,
                help="Exit after saving this many non-discarded episodes. "
                     "Useful for staged collect-then-train Robometer workflows.")
_p.add_argument("--no-wait-learner", action="store_true",
                help="Do not wait for learner_status.json before constructing "
                     "the env runner. Mainly for debugging.")
_args, _ = _p.parse_known_args()


def _pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_for_learner_before_init() -> None:
    """Avoid loading envrunner GPU/robot resources while learner is heavy."""
    if _args.no_wait_learner or not RL_CKPT_DIR:
        return
    path = os.path.join(RL_CKPT_DIR, "learner_status.json")
    last_log = 0.0
    while True:
        try:
            with open(path) as f:
                status = json.load(f)
        except FileNotFoundError:
            return
        except Exception:
            return
        phase = str(status.get("phase", "idle"))
        if phase == "idle":
            return
        pid = status.get("pid")
        if not _pid_alive(pid):
            return
        now = time.monotonic()
        if now - last_log > 15.0:
            logging.info(
                "Learner is %s (%s); waiting before envrunner initializes",
                phase, status.get("message", ""))
            last_log = now
        time.sleep(2.0)


_wait_for_learner_before_init()

runner = YAMRLEnvRunner(
    pretrained_policy_ckpt = BC_POLICY_CKPT,
    norm_npz_path          = NORM_NPZ,
    online_data_dir        = ONLINE_DATA_DIR,
    rl_checkpoint_dir      = RL_CKPT_DIR,
    actor_hidden_dims      = NETWORK["actor_hidden_dims"],
    residual_scale         = _args.residual_scale,
    max_joint_step         = _args.max_joint_step,
    max_episodes           = _args.max_episodes,
    obs_horizon            = TRAINING["obs_horizon"],
    action_horizon         = TRAINING["action_horizon"],
    action_dim             = TRAINING["action_dim"],
    **apply_hardware_env_overrides(HARDWARE),
    **COMM,
)
runner.run()
