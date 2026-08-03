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
    reward_mode = os.environ.get("YAM_REWARD_MODE", "").strip().lower()
    if reward_mode == "robometer":
        out["use_hire_reward"] = False
        out["use_robometer_reward"] = True
    elif reward_mode == "hire":
        out["use_hire_reward"] = True
        out["use_robometer_reward"] = False

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


def apply_learner_env_overrides(training: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay learner resource knobs from env vars.

    These are intentionally separate from reward selection. They let a single
    workstation run Robometer + learner without building a huge pre-encoded
    training pool that can trigger desktop-wide OOM.
    """
    out = dict(training)
    mapping = {
        "gradient_steps": ("YAM_LEARNER_GRADIENT_STEPS",),
        "batch_size": ("YAM_LEARNER_BATCH_SIZE",),
        "num_next_noise_samples": ("YAM_LEARNER_K_CRITIC",),
        "num_multi_z_for_actor_loss": ("YAM_LEARNER_K_ACTOR",),
        "training_pool_size_limit": ("YAM_LEARNER_POOL_SIZE_LIMIT",),
        "training_encode_batch_size": ("YAM_LEARNER_ENCODE_BATCH_SIZE",),
        "bc_pool_inference_steps": ("YAM_LEARNER_BC_POOL_INFERENCE_STEPS",),
    }
    for key, env_names in mapping.items():
        for en in env_names:
            if os.environ.get(en) not in (None, ""):
                out[key] = int(os.environ[en])
                break
    return out


def apply_hardware_env_overrides(hardware: Dict[str, Any]) -> Dict[str, Any]:
    """RealSense serials and capture size from env (robot PC)."""
    out = dict(hardware)
    if os.environ.get("YAM_BASE_CAM_SERIAL"):
        out["base_cam_serial"] = _str("YAM_BASE_CAM_SERIAL", out["base_cam_serial"])
    if os.environ.get("YAM_WRIST_CAM_SERIAL"):
        out["wrist_cam_serial"] = _str("YAM_WRIST_CAM_SERIAL", out["wrist_cam_serial"])
    if os.environ.get("YAM_POLICY_CAMERA_ORDER"):
        out["policy_camera_order"] = _str(
            "YAM_POLICY_CAMERA_ORDER", out.get("policy_camera_order", "base_wrist")
        )
    if os.environ.get("YAM_MAX_EPISODE_STEPS"):
        out["max_episode_steps"] = _int("YAM_MAX_EPISODE_STEPS", out["max_episode_steps"])
    if os.environ.get("YAM_MAX_CAMERA_AGE"):
        out["max_camera_age"] = _float(
            "YAM_MAX_CAMERA_AGE", out.get("max_camera_age", 0.5)
        )
    return out
