#!/usr/bin/env python3
"""Evaluate a pre-trained flow-matching policy on a real YAM robot arm.

Designed to mirror the alignment choices used in RLinf's pi0.5 yam_deploy
branch (rlinf/envs/realworld/yam/yam_env.py), namely:
  * `linear_4310` active gripper (auto-calibrates at startup)
  * 30 Hz action frequency (matches LeRobot YAM dataset fps)
  * Explicit home pose (6 arm joints) via i2rt's `move_joints` slow ramp
  * Two RealSense D405 cameras (alphabetical order: base, then wrist)

Run with the **dice-rl venv**:

    /home/bike/Documents/niu/dice-rl/.venv/bin/python \
        /home/bike/Documents/niu/DICE-RL-Robot/scripts/eval_flow_matching_yam.py \
        --config /home/bike/Documents/niu/DICE-RL-Robot/checkpoints/yam_picknplace_paperplate_arizonabottle_pre_flow_matching_unet_img_ta16_td10/2026-05-16_01-07-54_42/.hydra/config.yaml \
        --ckpt   /home/bike/Documents/niu/DICE-RL-Robot/checkpoints/yam_picknplace_paperplate_arizonabottle_pre_flow_matching_unet_img_ta16_td10/2026-05-16_01-07-54_42/checkpoint/state_700.pt \
        --norm   /home/bike/Documents/niu/DICE-RL-Robot/checkpoints/yam_picknplace_paperplate_arizonabottle_pre_flow_matching_unet_img_ta16_td10/data_meta/normalization.npz \
        --can_channel can_follower_r \
        --base_serial 218622278369 \
        --wrist_serial 218622271309 \
        --home_joint_pos -0.006,0.835,0.835,-0.596,-0.007,-0.025 \
        --dry_run

Policy I/O contract (deduced from dice-rl source):
  * obs.state  : (1, cond_steps=2, 7)  state history, most recent at end
  * obs.rgb    : (1, img_cond_steps=2, 6, 128, 128)  base+wrist concatenated on channel
  * action     : (1, horizon=16, 7) in normalized [-1,1] space  → denormalize via
                 (a+1)/2 * (a_max - a_min) + a_min  yields absolute joint targets
                 (action_min == obs_min in the dataset).
  * exec_steps : 8 actions per chunk (cfg.act_steps).

Safety knobs:
  * --dry_run            : runs perception+inference but does NOT command the robot.
  * --max_joint_delta    : per-tick joint motion clamp (rad).
  * --command_gripper    : default OFF when teaching handle is mounted; turn ON for
                            an active gripper (e.g. linear_4310).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

# Make dice-rl's `model/` package importable. The flow matching code lives
# in /home/bike/Documents/niu/dice-rl and is not a pip-installed package.
_DICE_RL_REPO = os.environ.get("DICE_RL_REPO", str(Path.home() / "Documents/niu/dice-rl"))
if _DICE_RL_REPO not in sys.path:
    sys.path.insert(0, _DICE_RL_REPO)

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

log = logging.getLogger("eval_yam")
logging.basicConfig(level=logging.INFO, format="[%(name)s %(levelname)s] %(message)s")


# Joint limits from i2rt.robots.get_robot.get_yam_robot (incl. its +/-0.15 rad
# buffer). The 7th channel is the gripper.
YAM_JOINT_LIMITS_LOW = np.array([-2.767, -0.15, -0.15, -1.72, -1.72, -2.24, 0.0], dtype=np.float32)
YAM_JOINT_LIMITS_HIGH = np.array([3.28, 3.80, 3.28, 1.72, 1.72, 2.24, 1.5], dtype=np.float32)


# ---------------------------------------------------------------------------
# Min-max normalization (matches dice-rl/script/dataset/process_yam_dataset.py)
# ---------------------------------------------------------------------------


class MinMaxNorm:
    def __init__(self, lo: np.ndarray, hi: np.ndarray) -> None:
        self.lo = lo.astype(np.float32)
        self.hi = hi.astype(np.float32)
        self.range = (hi - lo + 1e-6).astype(np.float32)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (2.0 * (x - self.lo) / self.range - 1.0).astype(np.float32)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return ((x + 1.0) / 2.0 * self.range + self.lo).astype(np.float32)


# ---------------------------------------------------------------------------
# RealSense streamer (one per camera). rgb8 directly (matches RLinf YamEnv).
# ---------------------------------------------------------------------------


class _CameraThread(threading.Thread):
    def __init__(self, serial: str, width: int, height: int, fps: int, name: str) -> None:
        super().__init__(daemon=True, name=f"cam-{name}")
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None  # RGB uint8 HxWx3
        self._t_latest: float = 0.0
        self._pipeline = None

    def _open(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        pipeline.start(config)
        for _ in range(5):
            pipeline.wait_for_frames()
        self._pipeline = pipeline

    def run(self) -> None:
        try:
            self._open()
        except Exception as e:
            log.exception("[%s] open failed: %s", self.name, e)
            return
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                rgb = np.asanyarray(color.get_data())  # already RGB
                with self._lock:
                    self._latest = rgb
                    self._t_latest = time.monotonic()
            except Exception as e:
                log.warning("[%s] frame error: %s", self.name, e)
                time.sleep(0.05)

    def get(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            if self._latest is None:
                return None, 0.0
            return self._latest.copy(), self._t_latest

    def stop(self) -> None:
        self._stop.set()
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Policy load: instantiate FlowMatchingModel from hydra config, load weights
# ---------------------------------------------------------------------------


def load_flow_policy(config_path: str, ckpt_path: str, device: torch.device, use_ema: bool):
    # Hydra config (.hydra/config.yaml) contains ${now:...} interpolations; register a
    # stub resolver so OmegaConf can load it outside a hydra runtime.
    OmegaConf.register_new_resolver("now", lambda fmt: "n/a", replace=True)
    cfg = OmegaConf.load(config_path)
    import hydra

    model = hydra.utils.instantiate(cfg.model).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and use_ema and "ema" in state:
        sd = state["ema"]
        which = "ema"
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
        which = "model"
    else:
        sd = state
        which = "raw"
    log.info("using %r weights from checkpoint (epoch=%s)", which, state.get("epoch") if isinstance(state, dict) else "n/a")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log.info("loaded weights: %d missing, %d unexpected", len(missing), len(unexpected))
    if missing:
        log.warning("missing[:10]: %s", missing[:10])
    if unexpected:
        log.warning("unexpected[:10]: %s", unexpected[:10])
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Image preprocessing: bilinear resize to 128x128, uint8 RGB
# (dice-rl/script/dataset/process_yam_dataset.py uses cv2.resize, no center crop)
# ---------------------------------------------------------------------------


def _resize_to_128(rgb: np.ndarray) -> np.ndarray:
    """RGB HxWx3 uint8 -> 3x128x128 uint8."""
    rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    return np.transpose(rgb, (2, 0, 1)).astype(np.uint8)  # (3, 128, 128)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_home(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            f"--home_joint_pos needs 6 comma-separated floats (arm only), got {len(parts)}: {s!r}"
        )
    return np.array([float(p) for p in parts], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to .hydra/config.yaml")
    parser.add_argument("--ckpt", required=True, help="Path to state_*.pt")
    parser.add_argument("--norm", required=True, help="Path to normalization.npz")
    parser.add_argument("--can_channel", default="can_follower_r")
    parser.add_argument(
        "--gripper_type",
        default="linear_4310",
        choices=["linear_4310", "linear_3507", "crank_4310", "yam_teaching_handle", "no_gripper"],
        help="i2rt GripperType. linear_* auto-calibrates at startup (gripper actuates once).",
    )
    parser.add_argument("--base_serial", required=True)
    parser.add_argument("--wrist_serial", required=True)
    parser.add_argument(
        "--home_joint_pos",
        type=_parse_home,
        required=True,
        help="6 comma-separated arm joint positions (rad), e.g. -0.006,0.835,0.835,-0.596,-0.007,-0.025. "
        "Must match the start pose of your training data.",
    )
    parser.add_argument(
        "--home_gripper_pos",
        type=float,
        default=0.5,
        help="Normalized gripper position at home (0=closed, 1=open).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--control_hz", type=float, default=30.0,
        help="Action frequency (Hz). Default matches LeRobot YAM dataset fps.",
    )
    parser.add_argument("--ramp_seconds", type=float, default=3.0)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument(
        "--max_joint_delta", type=float, default=0.08,
        help="Per-tick max joint delta in rad (arm only). At 30 Hz, 0.08 rad ≈ 2.4 rad/s.",
    )
    parser.add_argument(
        "--command_gripper",
        action="store_true",
        help="Send commands to the 7th channel (gripper). Default OFF; turn ON only when an "
        "active gripper is mounted and you're sure the policy's gripper output is sane.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run perception + inference but do NOT actually command the robot.",
    )
    parser.add_argument(
        "--print_actions",
        action="store_true",
        help="Print the full denormalized action chunk each step.",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Load checkpoint['model'] instead of checkpoint['ema']. The dice-rl repo "
        "defaults to EMA for SL-trained policies; only override if you know better.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    log.info("device=%s  dry_run=%s  command_gripper=%s", device, args.dry_run, args.command_gripper)

    # --- policy + normalization ---
    model, cfg = load_flow_policy(args.config, args.ckpt, device, use_ema=not args.no_ema)
    norm_data = np.load(args.norm)
    state_norm = MinMaxNorm(norm_data["obs_min"], norm_data["obs_max"])
    action_norm = MinMaxNorm(norm_data["action_min"], norm_data["action_max"])
    cond_steps = int(cfg.cond_steps)
    img_cond_steps = int(cfg.img_cond_steps)
    horizon_steps = int(cfg.horizon_steps)
    act_steps = int(cfg.act_steps)
    log.info(
        "policy cfg: cond_steps=%d img_cond_steps=%d horizon=%d act_steps=%d flow_steps=%d",
        cond_steps, img_cond_steps, horizon_steps, act_steps, int(cfg.flow_steps),
    )
    log.info("state range: lo=%s\n             hi=%s", state_norm.lo, state_norm.hi)

    # --- connect hardware ---
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    gripper_type = GripperType.from_string_name(args.gripper_type)
    log.info("opening YAM on %s with gripper=%s", args.can_channel, gripper_type)
    if "linear" in args.gripper_type:
        log.warning("linear gripper auto-calibrates at startup; the gripper will actuate once.")
    robot = get_yam_robot(
        channel=args.can_channel,
        gripper_type=gripper_type,
        zero_gravity_mode=True,
    )

    base_cam = _CameraThread(args.base_serial, 640, 480, 30, "base")
    wrist_cam = _CameraThread(args.wrist_serial, 640, 480, 30, "wrist")
    base_cam.start()
    wrist_cam.start()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if base_cam.get()[0] is not None and wrist_cam.get()[0] is not None:
            break
        time.sleep(0.05)
    if base_cam.get()[0] is None or wrist_cam.get()[0] is None:
        log.error("cameras did not produce frames within 5 s")
        sys.exit(1)
    log.info("both cameras streaming.")

    def safe_shutdown(*_):
        log.info("shutting down: holding current pose, stopping threads")
        try:
            robot.command_joint_pos(robot.get_joint_pos())
        except Exception:
            pass
        base_cam.stop()
        wrist_cam.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, safe_shutdown)
    signal.signal(signal.SIGTERM, safe_shutdown)

    # --- per-episode loop ---
    state_hist: deque = deque(maxlen=cond_steps)
    img_hist: deque = deque(maxlen=img_cond_steps)
    period = 1.0 / args.control_hz

    for ep in range(args.num_episodes):
        log.info("=== episode %d ===", ep + 1)

        # Move to home (arm + gripper) - matches data-collection start pose.
        home_full = np.concatenate([args.home_joint_pos, [args.home_gripper_pos]]).astype(np.float32)
        log.info("moving to home: arm=%s gripper=%.3f (ramp=%.1fs)",
                 args.home_joint_pos.tolist(), args.home_gripper_pos, args.ramp_seconds)
        if not args.dry_run:
            robot.move_joints(home_full, time_interval_s=args.ramp_seconds)
            time.sleep(0.5)

        # Read full state after home so history starts from a sane pose.
        def _read_state() -> np.ndarray:
            obs = robot.get_observations()
            joint = np.asarray(obs["joint_pos"], dtype=np.float32)        # (6,)
            grip = np.asarray(obs.get("gripper_pos", [args.home_gripper_pos]),
                              dtype=np.float32).reshape(-1)               # (1,)
            return np.concatenate([joint[:6], grip[:1]])                  # (7,)

        # Pre-fill history.
        q = _read_state()
        base_rgb, _ = base_cam.get()
        wrist_rgb, _ = wrist_cam.get()
        base128 = _resize_to_128(base_rgb)
        wrist128 = _resize_to_128(wrist_rgb)
        rgb_concat = np.concatenate([base128, wrist128], axis=0)  # (6, 128, 128)
        for _ in range(cond_steps):
            state_hist.append(q.copy())
        for _ in range(img_cond_steps):
            img_hist.append(rgb_concat.copy())

        # Distribution check on home state.
        in_range = np.all((q >= state_norm.lo - 0.05) & (q <= state_norm.hi + 0.05))
        if not in_range:
            log.warning(
                "current state %s is OUTSIDE the BC training range.\n"
                "  lo=%s\n  hi=%s\n"
                "The policy may behave unexpectedly. Re-check --home_joint_pos.",
                q, state_norm.lo, state_norm.hi,
            )
        else:
            log.info("home state is inside BC training range.")

        input("Press Enter to start the eval loop (Ctrl-C to abort).")

        for step in range(args.max_steps):
            tick = time.monotonic()

            # Sample the latest sensor data.
            q_now = _read_state()
            base_rgb, _ = base_cam.get()
            wrist_rgb, _ = wrist_cam.get()
            base128 = _resize_to_128(base_rgb)
            wrist128 = _resize_to_128(wrist_rgb)
            rgb_concat = np.concatenate([base128, wrist128], axis=0)
            state_hist.append(q_now)
            img_hist.append(rgb_concat)

            # Build cond tensors. Most-recent at end (matches sequence.py).
            state_arr = np.stack(list(state_hist), axis=0)            # (To, 7)
            img_arr = np.stack(list(img_hist), axis=0)                # (To, 6, 128, 128)
            state_n = state_norm.normalize(state_arr)
            state_t = torch.from_numpy(state_n)[None].to(device)      # (1, To, 7)
            img_t = torch.from_numpy(img_arr)[None].to(device).float()  # (1, To, 6, 128, 128)
            cond = {"state": state_t, "rgb": img_t}

            t0 = time.monotonic()
            with torch.no_grad():
                sample = model(cond=cond, deterministic=True)
            actions_n = sample.trajectories[0].cpu().numpy()           # (16, 7) in [-1, 1]
            actions = action_norm.denormalize(actions_n)               # absolute joint targets
            infer_ms = (time.monotonic() - t0) * 1000.0

            if args.print_actions:
                log.info("[ep %d step %d] infer=%.1fms  q_now=%s  pred_chunk[0]=%s ... [act-1]=%s",
                         ep+1, step, infer_ms,
                         np.round(q_now, 3).tolist(),
                         np.round(actions[0], 3).tolist(),
                         np.round(actions[act_steps - 1], 3).tolist())
            else:
                log.info("[ep %d step %d] infer=%.1fms  q_now=%s",
                         ep+1, step, infer_ms, np.round(q_now, 3).tolist())

            # Execute first act_steps actions at control_hz.
            chunk = actions[:act_steps]
            for i, q_target in enumerate(chunk):
                target_tick = tick + i * period

                q_cur = _read_state()
                delta = q_target - q_cur
                # Clamp per-tick delta on arm joints to limit motion speed.
                delta[:6] = np.clip(delta[:6], -args.max_joint_delta, args.max_joint_delta)
                q_cmd = q_cur + delta
                # Joint range clamp.
                np.clip(q_cmd, YAM_JOINT_LIMITS_LOW, YAM_JOINT_LIMITS_HIGH, out=q_cmd)
                # Hold gripper at current value unless explicitly requested.
                if not args.command_gripper:
                    q_cmd[6] = q_cur[6]

                if not args.dry_run:
                    robot.command_joint_pos(q_cmd)

                # Maintain control rate.
                sleep = target_tick + period - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)

        log.info("episode %d finished (max_steps=%d)", ep + 1, args.max_steps)

    safe_shutdown()


if __name__ == "__main__":
    main()
