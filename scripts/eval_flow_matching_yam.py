#!/usr/bin/env python3
"""Evaluate a pre-trained flow-matching policy on a real YAM arm.

Run this with the **dice-rl venv** (the one that has the original training
codebase importable):

    /home/bike/Documents/niu/dice-rl/.venv/bin/python \
        /home/bike/Documents/niu/DICE-RL-Robot/scripts/eval_flow_matching_yam.py \
        --config /home/bike/Documents/niu/DICE-RL-Robot/checkpoints/yam_picknplace_paperplate_arizonabottle_pre_flow_matching_unet_img_ta16_td10/2026-05-16_01-07-54_42/.hydra/config.yaml \
        --ckpt   /home/bike/Documents/niu/DICE-RL-Robot/checkpoints/yam_picknplace_paperplate_arizonabottle_pre_flow_matching_unet_img_ta16_td10/2026-05-16_01-07-54_42/checkpoint/state_700.pt \
        --norm   /home/bike/Documents/niu/DICE-RL-Robot/checkpoints/yam_picknplace_paperplate_arizonabottle_pre_flow_matching_unet_img_ta16_td10/data_meta/normalization.npz \
        --can_channel can_follower_r \
        --base_serial 218622278369 \
        --wrist_serial 218622271309

Design (intentionally minimal, NOT going through YamServerEnv):
  * dataset config says action_dim=action_min ranges match obs (joint pos),
    so we treat policy output as **absolute joint targets** (despite the
    ``abs_action: false`` flag in the hydra config, which is just a training
    knob about whether to subtract the previous state on the dataset side).
  * we open YAM directly with i2rt, and the two RealSense cameras directly
    with pyrealsense2, in their own threads.
  * each inference returns 16 actions (horizon_steps), we execute the first
    8 (act_steps) at a fixed step interval before re-querying.
  * before the first chunk, we slow-move (over 3 s) to the first predicted
    joint target so we don't jerk the arm.
  * we clamp every commanded q to the i2rt YAM joint limits.
  * Ctrl-C cleanly hands the arm back to its current pose (`command_joint_pos
    (current_q)`) and shuts down threads.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

log = logging.getLogger("eval_yam")
logging.basicConfig(level=logging.INFO, format="[%(name)s %(levelname)s] %(message)s")


# Joint limits from i2rt.robots.get_robot.get_yam_robot (incl. its +/-0.15 rad
# buffer). Index 6 is the gripper; we use the YAM_TEACHING_HANDLE convention
# which exposes the handle motor in [0, 1.5] roughly.
YAM_JOINT_LIMITS_LOW = np.array([-2.767, -0.15, -0.15, -1.72, -1.72, -2.24, 0.0], dtype=np.float32)
YAM_JOINT_LIMITS_HIGH = np.array([3.28, 3.80, 3.28, 1.72, 1.72, 2.24, 2.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Normalization
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
# RealSense streamer (one per camera). Pushes BGR frames + uint8 [0,255].
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
        self._latest: np.ndarray | None = None  # BGR uint8 HxWx3
        self._t_latest: float = 0.0
        self._pipeline = None

    def _open(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
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
                bgr = np.asanyarray(color.get_data())
                with self._lock:
                    self._latest = bgr
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
# Policy loading: build the FlowMatchingModel from hydra config + load weights
# ---------------------------------------------------------------------------


def load_flow_policy(config_path: str, ckpt_path: str, device: torch.device):
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    # `cfg.model` is `FlowMatchingModel` with nested `network`.
    import hydra

    model = hydra.utils.instantiate(cfg.model).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    if hasattr(model, "load_state_dict"):
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log.info("loaded weights: %d missing, %d unexpected", len(missing), len(unexpected))
        if missing:
            log.warning("missing[:10]: %s", missing[:10])
        if unexpected:
            log.warning("unexpected[:10]: %s", unexpected[:10])
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Image preprocessing: each cam -> resize to 128x128 -> uint8 (B, 3, 128, 128)
# Two cams concatenated along channel dim -> (B, 6, 128, 128). The training
# data ordering is alphabetical: base_camera, hand_camera -> [base, wrist].
# ---------------------------------------------------------------------------


def _resize_to_128(bgr: np.ndarray) -> np.ndarray:
    """BGR HxWx3 uint8 -> RGB 3x128x128 uint8 (no scaling; encoder /255 internally)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    return np.transpose(rgb, (2, 0, 1)).astype(np.uint8)  # (3, 128, 128)


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--norm", required=True)
    parser.add_argument("--can_channel", default="can_follower_r")
    parser.add_argument("--base_serial", required=True, help="base RealSense serial")
    parser.add_argument("--wrist_serial", required=True, help="wrist RealSense serial")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--control_hz", type=float, default=10.0,
                        help="Joint command rate. Conservative default; raise after sanity check.")
    parser.add_argument("--ramp_seconds", type=float, default=3.0,
                        help="Slow-ramp duration to the first predicted joint target.")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Per-episode safety cap (steps).")
    parser.add_argument("--max_joint_delta", type=float, default=0.15,
                        help="Per-tick max joint delta in rad. Prevents jerks if a "
                        "predicted target is far from current state.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run the full perception + inference loop but do NOT "
                        "command joints. Useful to verify everything before motion.")
    parser.add_argument("--command_gripper", action="store_true",
                        help="By default we DO NOT send commands to the 7th motor "
                        "(teaching handle is passive; sending positions can fight "
                        "the operator's hand). Pass this flag if you switched to "
                        "an active gripper.")
    args = parser.parse_args()

    device = torch.device(args.device)
    log.info("device=%s", device)

    # --- Load policy + config + norm stats ---
    model, cfg = load_flow_policy(args.config, args.ckpt, device)
    norm_data = np.load(args.norm)
    state_norm = MinMaxNorm(norm_data["obs_min"], norm_data["obs_max"])
    action_norm = MinMaxNorm(norm_data["action_min"], norm_data["action_max"])
    cond_steps = int(cfg.cond_steps)
    img_cond_steps = int(cfg.img_cond_steps)
    horizon_steps = int(cfg.horizon_steps)
    act_steps = int(cfg.act_steps)
    log.info(
        "cond_steps=%d img_cond_steps=%d horizon=%d act_steps=%d flow_steps=%d",
        cond_steps, img_cond_steps, horizon_steps, act_steps, int(cfg.flow_steps),
    )

    # --- Connect hardware ---
    from i2rt.robots.get_robot import GripperType, get_yam_robot

    log.info("opening YAM on %s", args.can_channel)
    robot = get_yam_robot(channel=args.can_channel, gripper_type=GripperType.YAM_TEACHING_HANDLE)
    robot.start_server()

    base_cam = _CameraThread(args.base_serial, 640, 480, 30, "base")
    wrist_cam = _CameraThread(args.wrist_serial, 640, 480, 30, "wrist")
    base_cam.start()
    wrist_cam.start()

    # Wait for cameras and motors to all produce a sample.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if base_cam.get()[0] is not None and wrist_cam.get()[0] is not None:
            break
        time.sleep(0.05)
    if base_cam.get()[0] is None or wrist_cam.get()[0] is None:
        log.error("cameras did not produce frames within 5 s")
        sys.exit(1)

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

    # --- Episode loop ---
    state_hist: deque = deque(maxlen=cond_steps)
    img_hist: deque = deque(maxlen=img_cond_steps)
    period = 1.0 / args.control_hz

    for ep in range(args.num_episodes):
        log.info("=== episode %d ===", ep + 1)
        # Pre-fill history with the current observation.
        q = robot.get_joint_pos().astype(np.float32)  # (7,)
        base_bgr, _ = base_cam.get()
        wrist_bgr, _ = wrist_cam.get()
        base_rgb = _resize_to_128(base_bgr)
        wrist_rgb = _resize_to_128(wrist_bgr)
        rgb_concat = np.concatenate([base_rgb, wrist_rgb], axis=0)  # (6, 128, 128)
        for _ in range(cond_steps):
            state_hist.append(q.copy())
        for _ in range(img_cond_steps):
            img_hist.append(rgb_concat.copy())

        # Check whether starting pose is in the BC training distribution.
        in_range = np.all((q >= state_norm.lo - 0.05) & (q <= state_norm.hi + 0.05))
        if not in_range:
            log.warning("current q=%s is OUTSIDE the training state range\n"
                        "  lo=%s\n  hi=%s\n"
                        "policy may behave unexpectedly. Move arm closer before continuing.",
                        q, state_norm.lo, state_norm.hi)
        input("Press Enter to start, Ctrl-C to abort.")

        ramped = False
        for step in range(args.max_steps):
            tick = time.monotonic()
            # Update history with latest sample.
            q_now = robot.get_joint_pos().astype(np.float32)
            base_bgr, _ = base_cam.get()
            wrist_bgr, _ = wrist_cam.get()
            base_rgb = _resize_to_128(base_bgr)
            wrist_rgb = _resize_to_128(wrist_bgr)
            rgb_concat = np.concatenate([base_rgb, wrist_rgb], axis=0)
            state_hist.append(q_now)
            img_hist.append(rgb_concat)

            # Build cond tensors. Most recent at end (matches sequence.py).
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
            actions = action_norm.denormalize(actions_n)               # (16, 7) joint targets
            infer_ms = (time.monotonic() - t0) * 1000.0

            # Execute first act_steps actions, one per period.
            chunk = actions[: act_steps]
            for i, q_target in enumerate(chunk):
                # Per-tick max delta clamp + joint-limit clamp.
                q_cur = robot.get_joint_pos().astype(np.float32)
                delta = q_target - q_cur
                np.clip(delta, -args.max_joint_delta, args.max_joint_delta, out=delta)
                q_cmd = q_cur + delta
                np.clip(q_cmd, YAM_JOINT_LIMITS_LOW, YAM_JOINT_LIMITS_HIGH, out=q_cmd)
                # Optionally hold the 7th motor at its current passive position.
                if not args.command_gripper:
                    q_cmd[6] = q_cur[6]

                if not ramped:
                    # First chunk's first action: slow-ramp from q_cur to q_cmd.
                    N = max(int(args.ramp_seconds * args.control_hz), 1)
                    for j in range(1, N + 1):
                        alpha = j / N
                        q_step = (1 - alpha) * q_cur + alpha * q_cmd
                        if not args.command_gripper:
                            q_step[6] = q_cur[6]
                        if not args.dry_run:
                            robot.command_joint_pos(q_step)
                        time.sleep(period)
                    ramped = True
                else:
                    if not args.dry_run:
                        robot.command_joint_pos(q_cmd)
                    # Sleep to maintain control rate.
                    elapsed = time.monotonic() - tick - i * period
                    sleep = period - elapsed
                    if sleep > 0:
                        time.sleep(sleep)

            log.info("[ep %d step %d] infer=%5.1f ms  q_now=%s",
                     ep + 1, step, infer_ms, np.round(q_now, 3).tolist())

        log.info("episode %d finished max_steps=%d", ep + 1, args.max_steps)

    safe_shutdown()


if __name__ == "__main__":
    main()
