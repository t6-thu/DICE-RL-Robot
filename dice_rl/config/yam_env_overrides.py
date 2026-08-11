"""Apply YAM_FINETUNE_OVERRIDES-style env vars to config dicts.

HiRE-Dice uses Hydra, e.g.::

    FINETUNE_GLOBAL_OVERRIDES="env.wrappers.robomimic_image.robometer_query_every_n_chunks=4 ..."

On real YAM, set the same knobs via env vars (see scripts/robometer/README.md) or edit
``dice_rl/config/yam_rl_config.py``.
"""

from __future__ import annotations

import os
from typing import Any, Dict


def _float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def _int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return str(v).strip() if v not in (None, "") else default


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def apply_robometer_env_overrides(training: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay Robometer fields from YAM_ROBOMETER_* / legacy ROBOMETER_* env vars."""
    out = dict(training)
    if os.environ.get("YAM_REWARD_MODE", "").strip().lower() == "robometer":
        out["use_hire_reward"] = False
        out["use_robometer_reward"] = True

    mapping = {
        "robometer_server_url": ("ROBOMETER_SERVER_URL", "YAM_ROBOMETER_SERVER_URL"),
        "robometer_task_instruction": ("YAM_ROBOMETER_TASK_INSTRUCTION",),
        "robometer_reward_weight": ("ROBOMETER_REWARD_WEIGHT", "YAM_ROBOMETER_REWARD_WEIGHT"),
        "robometer_camera": ("YAM_ROBOMETER_CAMERA", "ROBOMETER_CAMERA_KEY"),
        "robometer_query_fill_mode": ("YAM_ROBOMETER_QUERY_FILL_MODE",),
        "robometer_max_batch_size": ("YAM_ROBOMETER_MAX_BATCH_SIZE",),
        "robometer_query_every_n_chunks": ("YAM_ROBOMETER_QUERY_EVERY_N_CHUNKS",),
        "robometer_max_frames": ("YAM_ROBOMETER_MAX_FRAMES",),
        "robometer_request_timeout_s": ("YAM_ROBOMETER_REQUEST_TIMEOUT_S",),
    }
    for key, env_names in mapping.items():
        for en in env_names:
            if os.environ.get(en) not in (None, ""):
                val = os.environ[en]
                if key in (
                    "robometer_reward_weight",
                    "robometer_gamma_pbrs",
                    "robometer_request_timeout_s",
                ):
                    out[key] = float(val)
                elif key in (
                    "robometer_query_every_n_chunks",
                    "robometer_max_batch_size",
                    "robometer_max_frames",
                ):
                    out[key] = int(val)
                else:
                    out[key] = val.strip() if isinstance(val, str) else val
                break

    if os.environ.get("YAM_ROBOMETER_BGR_TO_RGB") not in (None, ""):
        out["robometer_bgr_to_rgb"] = _bool("YAM_ROBOMETER_BGR_TO_RGB", False)
    if os.environ.get("YAM_ROBOMETER_USE_FRAME_STEPS") not in (None, ""):
        out["robometer_use_frame_steps"] = _bool("YAM_ROBOMETER_USE_FRAME_STEPS", False)
    if os.environ.get("YAM_ROBOMETER_USE_RELATIVE_REWARDS") not in (None, ""):
        out["robometer_use_relative_rewards"] = _bool(
            "YAM_ROBOMETER_USE_RELATIVE_REWARDS", True
        )

    return out


def apply_hardware_env_overrides(hardware: Dict[str, Any]) -> Dict[str, Any]:
    """Apply robot-side launch overrides without mutating shared config."""
    out = dict(hardware)
    if os.environ.get("YAM_BASE_CAM_SERIAL"):
        out["base_cam_serial"] = _str("YAM_BASE_CAM_SERIAL", out["base_cam_serial"])
    if os.environ.get("YAM_WRIST_CAM_SERIAL"):
        out["wrist_cam_serial"] = _str("YAM_WRIST_CAM_SERIAL", out["wrist_cam_serial"])
    if os.environ.get("YAM_CAN_CHANNEL"):
        out["can_channel"] = _str("YAM_CAN_CHANNEL", out["can_channel"])
    if os.environ.get("YAM_GRIPPER_TYPE"):
        out["gripper_type"] = _str("YAM_GRIPPER_TYPE", out["gripper_type"])
    if os.environ.get("YAM_CONTROL_HZ"):
        out["control_hz"] = _float("YAM_CONTROL_HZ", out["control_hz"])
    if os.environ.get("YAM_MAX_EPISODE_STEPS"):
        out["max_episode_steps"] = _int(
            "YAM_MAX_EPISODE_STEPS", out["max_episode_steps"]
        )
    return out
