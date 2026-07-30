#!/usr/bin/env python3
"""Evaluate a pre-trained flow-matching policy on a real YAM robot arm.

Designed to mirror the alignment choices used in RLinf's pi0.5 yam_deploy
branch (rlinf/envs/realworld/yam/yam_env.py), namely:
  * `linear_4310` active gripper (auto-calibrates at startup)
  * 30 Hz action frequency (matches LeRobot YAM dataset fps)
  * Explicit home pose (6 arm joints) via i2rt's `move_joints` slow ramp
  * Two RealSense D405 cameras (alphabetical order: base, then wrist)

Run with the **DICE-RL-Robot venv** (no separate dice-rl venv needed — we
just add the dice-rl repo to sys.path so `model.flow_matching.*` imports work):

    source /home/bike/Documents/niu/DICE-RL-Robot/.venv/bin/activate
    python /home/bike/Documents/niu/DICE-RL-Robot/scripts/eval_flow_matching_yam.py \
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

Safety knobs (intentionally MINIMAL — we send the policy's output verbatim
otherwise, since per-tick delta clamps were found to staircase-clip smooth
chunks and disrupt the BC trajectory):
  * --dry_run            : runs perception + inference but does NOT command the robot.
  * --command_gripper    : default ON; disable only when a passive teaching
                            handle is mounted (use --no-command_gripper).
  * joint range clamp    : hardware-mandated (built into the script, not a flag).
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


class _SyncCamera:
    """Synchronous on-demand RealSense reader. NO background thread.

    This is intentional — RLinf's yam_deploy reads frames synchronously inside
    `_get_obs` for the same reason (see infer_pi05_realrobot.py comment about
    GIL starvation of the CAN control thread). Running camera readers in
    background Python threads starves i2rt's CAN control thread of GIL time
    and triggers "loss communication" errors.
    """

    def __init__(self, serial: str, width: int, height: int, fps: int, name: str) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        self._pipeline = None
        self._latest: np.ndarray | None = None
        self._t_latest: float = 0.0

    def start(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        pipeline.start(config)
        # One blocking wait to bootstrap the cache. After this, get() polls
        # non-blocking so the 30Hz inner action loop stays on its 33ms budget.
        try:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            if color:
                self._latest = np.asanyarray(color.get_data())
                self._t_latest = time.monotonic()
        except Exception as e:
            log.warning("[%s] bootstrap frame error: %s", self.name, e)
        self._pipeline = pipeline

    def get(self) -> tuple[np.ndarray | None, float]:
        """Non-blocking poll. Always returns IMMEDIATELY with the freshest
        cached frame (up to ~33ms stale, matching training's 30Hz sampling).
        This is the key trick to keep the inner action loop on a true 30Hz
        budget: blocking wait_for_frames would otherwise add ~33ms per camera
        per tick, halving the effective control rate.
        """
        if self._pipeline is None:
            return None, 0.0
        try:
            new_frames = self._pipeline.poll_for_frames()
            if new_frames:
                color = new_frames.get_color_frame()
                if color:
                    self._latest = np.asanyarray(color.get_data())
                    self._t_latest = time.monotonic()
        except Exception as e:
            log.warning("[%s] poll error: %s", self.name, e)
        if self._latest is None:
            return None, 0.0
        return self._latest, self._t_latest

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


# ---------------------------------------------------------------------------
# Policy load: instantiate FlowMatchingModel from hydra config, load weights
# ---------------------------------------------------------------------------


def load_flow_policy(config_path: str, ckpt_path: str, device: torch.device, use_ema: bool):
    # Hydra config (.hydra/config.yaml) contains ${now:...} interpolations; register a
    # stub resolver so OmegaConf can load it outside a hydra runtime.
    OmegaConf.register_new_resolver("now", lambda _fmt: "n/a", replace=True)
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
# Image preprocessing: must EXACTLY match the training-side pipeline.
#
# Training videos at /mnt/.../picknplace_paperplate_arizonabottle were produced
# by two stacked steps:
#
#   (1) convert_yam_to_lerobot_rlinf.py:resize_short_side_and_center_crop
#       camera_native (e.g. 640x480) ->
#       scale so shorter side reaches 256 (preserves aspect) ->
#       center-crop to 256x256 (drops the excess on the long side) ->
#       saved into LeRobot as 256x256 RGB videos.
#
#   (2) dice-rl/script/dataset/process_yam_dataset.py:_resize_rgb
#       256x256 ->
#       torch.nn.functional.interpolate(size=(128,128), mode="bilinear",
#                                       align_corners=False) ->
#       stored in train.npz as 128x128 uint8.
#
# An earlier eval pipeline went straight from 640x480 to 128x128, which (a)
# squashed the 4:3 aspect into 1:1, distorting the scene, and (b) used a
# different downsampler. Either alone is enough to push the visual feature
# off the BC training distribution and disable the grasp transition.
# ---------------------------------------------------------------------------


def _resize_short_side_and_center_crop(rgb: np.ndarray, target: int = 256) -> np.ndarray:
    """Mirror of resize_short_side_and_center_crop from convert_yam_to_lerobot_rlinf.py.

    rgb: (H, W, 3) uint8 RGB -> (target, target, 3) uint8 RGB.
    """
    h, w = rgb.shape[:2]
    scale = max(target / w, target / h)
    new_w = max(target, int(np.ceil(w * scale)))
    new_h = max(target, int(np.ceil(h * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (new_w - target) // 2
    y0 = (new_h - target) // 2
    return resized[y0:y0 + target, x0:x0 + target]


def _resize_to_128(rgb: np.ndarray) -> np.ndarray:
    """Full two-step training preprocessing.

    Input:  RGB (H, W, 3) uint8 at the camera's native resolution (typ. 640x480).
    Output: (3, 128, 128) uint8 RGB, ready to concat with another camera on
            the channel axis and feed to the policy.
    """
    # Step 1: short-side resize + center-crop -> 256x256 (matches LeRobot storage).
    rgb_256 = _resize_short_side_and_center_crop(rgb, target=256)
    # Step 2: torch bilinear -> 128x128 (matches process_yam_dataset.py:_resize_rgb).
    rgb_t = torch.from_numpy(rgb_256).permute(2, 0, 1).unsqueeze(0).float()
    rgb_t = torch.nn.functional.interpolate(
        rgb_t, size=(128, 128), mode="bilinear", align_corners=False
    )
    return rgb_t.squeeze(0).clamp(0, 255).to(torch.uint8).numpy()      # (3,128,128)


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
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--camera_fps", type=int, default=30,
                        help="Camera capture rate. Drop to 15 or 5 if the wrist camera is "
                        "sharing a USB 2.0 controller with the CAN dongles -- the lower "
                        "bandwidth keeps the CAN packets flowing without packet starvation.")
    parser.add_argument("--no_cameras", action="store_true",
                        help="DIAGNOSTIC: skip camera open entirely; feed zero-valued images "
                        "to the policy. The policy output will be wrong (no visual info) but "
                        "the CAN bus / control thread should behave exactly like the i2rt "
                        "baseline. Use this to confirm whether cameras are the trigger.")
    parser.add_argument("--gripper_limits", default=None,
                        help="For linear_*: 'closed,open' (raw motor angles in rad) so i2rt "
                        "skips the auto-calibration that destabilizes the CAN bus. We learned "
                        "the left-arm linear_4310 limits from a prior auto-cal run: "
                        "1.077,6.316. Pass that string to use linear_4310 + skip calibration.")
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
        default=1.0,
        help="Normalized gripper command at home (i2rt command space). Verified "
        "convention on this machine via test_gripper_direction.py: command=1 -> "
        "OPEN, command=0 -> CLOSED. Default 1.0 = fully open at home, matching "
        "the training-data start where state[6] ~= 0.998 (gripper fully open "
        "before approach). Setting this to 0.5 would have the gripper close "
        "halfway during the home ramp, which is what the policy interprets as "
        "'mid-grasp' and likely confuses the visual-state input.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--control_hz", type=float, default=30.0,
        help="Action frequency (Hz). Default matches LeRobot YAM dataset fps.",
    )
    parser.add_argument("--ramp_seconds", type=float, default=8.0,
                        help="Time (s) for the auto-home move. Longer = gentler current draw "
                        "= less bus stress. Default 8s is conservative.")
    parser.add_argument("--skip_home", action="store_true",
                        help="Skip the automatic move_joints(home, ...) at episode start. "
                        "Use this when you've already manually placed the arm at home pose "
                        "(e.g. by back-driving it under gravity_comp mode). Avoids the "
                        "current spike that can destabilize CAN if home is far from current pose.")
    parser.add_argument("--max_home_distance", type=float, default=0.5,
                        help="If any single joint must move farther than this (rad) to reach "
                        "home, refuse to auto-home (suggest manual placement first). "
                        "Override with a larger value if you trust your bus.")
    parser.add_argument(
        "--num_episodes", type=int, default=999,
        help="Cap on the number of episodes per launch. Between episodes the "
        "arm ramps back to home and (without --no_prompt) the script prompts "
        "you to press Enter for the next try or 'q' to quit. Default 999 = "
        "effectively unlimited; use 1 if you want a one-shot run.",
    )
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument(
        "--act_steps", type=int, default=None,
        help="Override cfg.act_steps (= how many actions of the 16-step chunk to execute "
        "before re-querying). Training default is 8 (264ms open-loop). Smaller values "
        "(e.g. 4 or 2) close the loop tighter and reduce BC compounding error at the cost "
        "of more inference per second.",
    )
    parser.add_argument(
        "--noise_seed", type=int, default=None,
        help="If set, use a FIXED init noise for flow-matching inference (deterministic). "
        "Without this, every call uses fresh torch.randn -> run-to-run variance even on "
        "the same observation. Set to any int (e.g. 0) for reproducible behavior.",
    )
    parser.add_argument(
        "--reset_on_exit",
        action="store_true",
        default=True,
        help="On episode end or Ctrl-C, slowly ramp the arm back to --home_joint_pos so "
        "the next run starts from a clean state. Uses i2rt's move_joints (50-step "
        "interpolated). Disable with --no-reset_on_exit.",
    )
    parser.add_argument(
        "--no-reset_on_exit", dest="reset_on_exit", action="store_false",
    )
    parser.add_argument(
        "--reset_seconds", type=float, default=4.0,
        help="Duration of the reset-on-exit ramp.",
    )
    parser.add_argument(
        "--command_gripper",
        action="store_true",
        default=True,
        help="Send the policy's gripper output (action[6]) to the 7th motor (linear_4310 etc). "
        "Default ON. The offline replay confirmed gripper predictions match training data "
        "with MAE ~0.10. Without this, the arm approaches the object but never closes. "
        "Disable only if you mount a passive teaching handle.",
    )
    parser.add_argument(
        "--no-command_gripper", dest="command_gripper", action="store_false",
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
    parser.add_argument(
        "--check_cameras_only",
        action="store_true",
        help="Just open both cameras, save labeled snapshots to /tmp, and exit. "
        "Use this to visually verify base vs wrist serial assignment before running eval.",
    )
    parser.add_argument(
        "--no_prompt",
        action="store_true",
        help="Skip the 'Press Enter to start' interactive prompt. Useful for non-interactive runs.",
    )
    parser.add_argument(
        "--dump_obs_dir",
        default=None,
        help="Optional directory. When set, every inference step saves the exact "
        "128x128 RGB tensor fed into the policy as JPG so you can visually verify "
        "the camera view matches training. Files: <dir>/step_NNNN_base.jpg and "
        "<dir>/step_NNNN_wrist.jpg.",
    )
    args = parser.parse_args()

    # ---- Camera-only sanity check ----
    if args.check_cameras_only:
        base_cam = _SyncCamera(args.base_serial, args.camera_width, args.camera_height, args.camera_fps, "base")
        wrist_cam = _SyncCamera(args.wrist_serial, args.camera_width, args.camera_height, args.camera_fps, "wrist")
        base_cam.start()
        wrist_cam.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if base_cam.get()[0] is not None and wrist_cam.get()[0] is not None:
                break
            time.sleep(0.05)
        base_rgb, _ = base_cam.get()
        wrist_rgb, _ = wrist_cam.get()
        if base_rgb is None or wrist_rgb is None:
            log.error("camera open failed")
            base_cam.stop(); wrist_cam.stop()
            sys.exit(1)
        # Convert RGB -> BGR for cv2 imwrite, and add a label.
        def _annotate(rgb, label, serial):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(bgr, f"serial {serial}",
                        (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            return bgr
        base_anno = _annotate(base_rgb, "BASE", args.base_serial)
        wrist_anno = _annotate(wrist_rgb, "WRIST", args.wrist_serial)
        # Side-by-side, padded to equal height.
        h = max(base_anno.shape[0], wrist_anno.shape[0])
        def _pad(img):
            if img.shape[0] < h:
                pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
                return np.vstack([img, pad])
            return img
        side = np.hstack([_pad(base_anno), _pad(wrist_anno)])
        out_combined = "/tmp/yam_camera_check.jpg"
        out_base = "/tmp/yam_camera_base.jpg"
        out_wrist = "/tmp/yam_camera_wrist.jpg"
        cv2.imwrite(out_combined, side)
        cv2.imwrite(out_base, base_anno)
        cv2.imwrite(out_wrist, wrist_anno)
        log.info("camera check images saved:")
        log.info("  combined side-by-side : %s", out_combined)
        log.info("  base only             : %s", out_base)
        log.info("  wrist only            : %s", out_wrist)
        log.info("Open with: xdg-open %s   (or eog/feh/your viewer)", out_combined)
        base_cam.stop()
        wrist_cam.stop()
        sys.exit(0)
    # ---- end camera-only mode ----

    device = torch.device(args.device)
    log.info("device=%s  dry_run=%s  command_gripper=%s", device, args.dry_run, args.command_gripper)

    if args.dump_obs_dir is not None:
        os.makedirs(args.dump_obs_dir, exist_ok=True)
        log.info("will dump per-step 128x128 obs JPEGs to %s", args.dump_obs_dir)

    # --- policy + normalization ---
    model, cfg = load_flow_policy(args.config, args.ckpt, device, use_ema=not args.no_ema)
    norm_data = np.load(args.norm)
    state_norm = MinMaxNorm(norm_data["obs_min"], norm_data["obs_max"])
    action_norm = MinMaxNorm(norm_data["action_min"], norm_data["action_max"])
    cond_steps = int(cfg.cond_steps)
    img_cond_steps = int(cfg.img_cond_steps)
    horizon_steps = int(cfg.horizon_steps)
    # Allow CLI override of act_steps so we can close the loop tighter at eval
    # time than the training config used. Smaller act_steps = more frequent
    # re-querying = less compounding error from open-loop drift, at the cost
    # of more inference per second.
    act_steps = args.act_steps if args.act_steps is not None else int(cfg.act_steps)
    log.info(
        "policy cfg: cond_steps=%d img_cond_steps=%d horizon=%d act_steps=%d "
        "(from cfg=%d, %s) flow_steps=%d",
        cond_steps, img_cond_steps, horizon_steps, act_steps, int(cfg.act_steps),
        "CLI override" if args.act_steps is not None else "no override",
        int(cfg.flow_steps),
    )
    log.info("state range: lo=%s\n             hi=%s", state_norm.lo, state_norm.hi)

    # --- model warmup BEFORE touching the robot ---
    # Mirrors RLinf's `infer_pi05_realrobot.py` which loads + warms the model
    # before opening the robot. Without this, the FIRST inference inside the
    # eval loop takes ~300ms (kernel compile) and holds the GIL the whole time,
    # starving i2rt's CAN control thread of GIL access and triggering
    # "motor X loss communication". After warmup, inferences run ~30ms.
    log.info("warming up policy on GPU (this is what RLinf does to avoid GIL "
             "starvation of the CAN control thread later)...")
    _warm_t0 = time.monotonic()
    with torch.no_grad():
        dummy_state = torch.zeros(1, cond_steps, int(cfg.obs_dim), device=device)
        dummy_rgb = torch.zeros(1, img_cond_steps, 6, 128, 128, dtype=torch.float32, device=device)
        # Two passes — first compiles kernels, second is the true steady-state time.
        for i in range(2):
            _ = model(cond={"state": dummy_state, "rgb": dummy_rgb}, deterministic=True)
            torch.cuda.synchronize() if device.type == "cuda" else None
    log.info("warmup done in %.0fms (first inference now ~30ms)",
             (time.monotonic() - _warm_t0) * 1000)

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
    # NOTE: MotorChainRobot.__init__ already spins up `_server_thread` that
    # drives the control loop, so we MUST NOT call start_server() again here;
    # a second loop fights the first one on the bus and triggers
    # "Motor error detected: ... loss communication".

    # Fail-fast guard: gripper auto-calibration can crash the control thread
    # (motor X "loss communication") on a flaky CAN bus. If that thread is
    # dead, every command_joint_pos call below silently no-ops, producing a
    # misleadingly clean log. We poll for 10 s and abort if it dies.
    server_thread = getattr(robot, "_server_thread", None)

    def _assert_control_alive(context: str) -> None:
        if server_thread is not None and not server_thread.is_alive():
            log.error(
                "[%s] i2rt robot_server thread is NOT alive. The control loop crashed "
                "(typically CAN comm loss during gripper auto-calibration). Commands "
                "would silently no-op. Recommended fixes:\n"
                "  1) sudo ip link set %s down && sudo ip link set %s up type can bitrate 1000000  (CAN reset)\n"
                "  2) Try --gripper_type yam_teaching_handle to skip auto-calibration\n"
                "  3) Try the other follower CAN channel",
                context, args.can_channel, args.can_channel,
            )
            try:
                base_cam.stop(); wrist_cam.stop()
                robot.close()
            except Exception:
                pass
            sys.exit(2)

    # Match RLinf yam_env.py exactly: open cameras IMMEDIATELY after robot
    # init. NO command, NO sleep, NO fail-fast wait. i2rt's default state is
    # zero-torque + gravity-comp; the arm just hangs there. Issuing any
    # command (including a "hold current pose") flips kp from 0 to 80 in one
    # step, which on a stale state read can produce a torque spike strong
    # enough to delay a CAN tick and trip the motor watchdog.
    _assert_control_alive("post-init")

    if args.no_cameras:
        log.warning("--no_cameras: feeding zero images to policy; CAN traffic is the only "
                    "USB load. Policy output will be wrong but CAN should match i2rt baseline.")
        # Use a sentinel object that mimics _SyncCamera's `get()` API.
        class _ZeroCam:
            def __init__(self, h, w):
                self._frame = np.zeros((h, w, 3), dtype=np.uint8)
            def start(self): pass
            def stop(self): pass
            def get(self): return self._frame.copy(), time.monotonic()
        base_cam = _ZeroCam(args.camera_height, args.camera_width)
        wrist_cam = _ZeroCam(args.camera_height, args.camera_width)
    else:
        base_cam = _SyncCamera(args.base_serial, args.camera_width, args.camera_height, args.camera_fps, "base")
        wrist_cam = _SyncCamera(args.wrist_serial, args.camera_width, args.camera_height, args.camera_fps, "wrist")
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

    # SIGINT semantics:
    #   1st Ctrl-C while an episode is running -> stop current episode, reset
    #     to home, then prompt for the next episode (NOT exit the script).
    #   2nd Ctrl-C (while idle at the prompt, or another Ctrl-C inside the
    #     current cleanup) -> hard quit.
    _abort_episode = {"flag": False}
    _force_quit = {"flag": False}

    def _ramp_to_home() -> None:
        if args.dry_run or not args.reset_on_exit:
            return
        try:
            home_arm = np.asarray(args.home_joint_pos, dtype=np.float64)
            current = robot.get_joint_pos().astype(np.float64)
            home_full = np.concatenate([home_arm, [current[6]]])
            log.info("resetting to home pose over %.1fs ...", args.reset_seconds)
            robot.move_joints(home_full, time_interval_s=args.reset_seconds)
            log.info("at home.")
        except Exception as e:
            log.warning("reset_to_home failed: %s", e)

    def _final_shutdown() -> None:
        log.info("shutting down camera pipelines...")
        try:
            base_cam.stop()
            wrist_cam.stop()
        except Exception:
            pass

    def handle_sigint(*_):
        if _abort_episode["flag"]:
            # Already aborting; this is a 2nd Ctrl-C -> hard quit.
            log.warning("force quit (Ctrl-C twice)")
            _force_quit["flag"] = True
            try:
                _final_shutdown()
            except Exception:
                pass
            os._exit(130)
        log.info("Ctrl-C received: ending current episode, will reset to home.")
        _abort_episode["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    # --- per-episode loop ---
    state_hist: deque = deque(maxlen=cond_steps)
    img_hist: deque = deque(maxlen=img_cond_steps)
    period = 1.0 / args.control_hz

    # Define state-reader once, before any per-episode use.
    def _read_state() -> np.ndarray:
        obs = robot.get_observations()
        joint = np.asarray(obs["joint_pos"], dtype=np.float32)        # (6,)
        grip = np.asarray(obs.get("gripper_pos", [args.home_gripper_pos]),
                          dtype=np.float32).reshape(-1)               # (1,)
        return np.concatenate([joint[:6], grip[:1]])                  # (7,)

    for ep in range(args.num_episodes):
        log.info("=== episode %d ===", ep + 1)

        # Move to home (arm + gripper) - matches data-collection start pose.
        home_full = np.concatenate([args.home_joint_pos, [args.home_gripper_pos]]).astype(np.float32)

        if args.skip_home:
            log.info("--skip_home set: leaving arm wherever you put it.")
            q_now_arm = _read_state()[:6]
            far = np.max(np.abs(q_now_arm - args.home_joint_pos))
            log.info("current arm vs target home: max |delta|=%.3f rad", far)
        else:
            q_now_arm = _read_state()[:6]
            far = np.max(np.abs(q_now_arm - args.home_joint_pos))
            log.info("moving to home: arm=%s gripper=%.3f (ramp=%.1fs, max |delta|=%.3f rad)",
                     args.home_joint_pos.tolist(), args.home_gripper_pos, args.ramp_seconds, far)
            if far > args.max_home_distance:
                log.error(
                    "Refusing to auto-home: max joint delta %.3f rad > --max_home_distance %.3f. "
                    "Manually back-drive the arm closer to the home pose (use gravity_comp mode "
                    "via `python /home/bike/Documents/niu/i2rt/i2rt/robots/motor_chain_robot.py "
                    "--channel %s --gripper_type crank_4310 --operation_mode gravity_comp`), "
                    "or re-run with --skip_home if you've already done so.",
                    far, args.max_home_distance, args.can_channel,
                )
                base_cam.stop(); wrist_cam.stop()
                try:
                    robot.close()
                except Exception:
                    pass
                sys.exit(3)
            _assert_control_alive("pre-home-move")
            if not args.dry_run:
                robot.move_joints(home_full, time_interval_s=args.ramp_seconds)
                time.sleep(0.5)
            _assert_control_alive("post-home-move")

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

        if args.no_prompt:
            log.info("--no_prompt set; starting eval loop immediately.")
        else:
            input("Press Enter to start the eval loop (Ctrl-C to abort).")

        _abort_episode["flag"] = False  # reset for this episode

        for step in range(args.max_steps):
            if _abort_episode["flag"]:
                break
            # NOTE: do NOT capture `tick` here. Camera reads (~66ms) + inference
            # (~30ms) take ~100ms, and if we paced the inner action loop from
            # this early tick, the first 3 actions would fire instantly trying
            # to "catch up", producing a visible snap before the chunk settles
            # into 33ms cadence. We capture chunk_start_t AFTER inference so
            # all 8 actions in the chunk are spaced uniformly at 1/control_hz.

            # Do NOT re-sample state/images here. The inner action loop below
            # samples state + cameras at 30Hz into state_hist / img_hist, so
            # after the previous chunk finished the deques already contain the
            # most-recent two 30Hz frames (separated by ~33ms, matching how
            # training samples its cond pair). Re-sampling here would inject a
            # third near-duplicate "now" frame and collapse the history's
            # 33ms gap to ~1ms, which hides the velocity cue the policy uses
            # to decide when to close the gripper.
            #
            # For the very FIRST iteration, the deques were pre-filled before
            # entering the episode loop with the at-home state duplicated,
            # which mirrors how dataset/sequence.py pads at episode start.
            q_now = state_hist[-1]
            rgb_concat = img_hist[-1]
            if args.dump_obs_dir is not None:
                # rgb_concat is (6, 128, 128) RGB CHW. Slice the two cams back out.
                base_hwc = np.transpose(rgb_concat[:3], (1, 2, 0))   # (128,128,3) RGB
                wrist_hwc = np.transpose(rgb_concat[3:], (1, 2, 0))
                cv2.imwrite(
                    os.path.join(args.dump_obs_dir, f"step_{step:04d}_base.jpg"),
                    cv2.cvtColor(base_hwc, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(args.dump_obs_dir, f"step_{step:04d}_wrist.jpg"),
                    cv2.cvtColor(wrist_hwc, cv2.COLOR_RGB2BGR),
                )

            # Build cond tensors. Most-recent at end (matches sequence.py).
            state_arr = np.stack(list(state_hist), axis=0)            # (To, 7)
            img_arr = np.stack(list(img_hist), axis=0)                # (To, 6, 128, 128)
            state_n = state_norm.normalize(state_arr)
            state_t = torch.from_numpy(state_n)[None].to(device)      # (1, To, 7)
            img_t = torch.from_numpy(img_arr)[None].to(device).float()  # (1, To, 6, 128, 128)
            cond = {"state": state_t, "rgb": img_t}

            # Flow matching's `deterministic=True` flag is a no-op in the model
            # code (the init noise is still torch.randn each call). For eval
            # reproducibility, optionally pass a FIXED init_noise tensor so we
            # get the same trajectory whenever we see the same observation.
            if args.noise_seed is not None:
                g = torch.Generator(device=device)
                g.manual_seed(int(args.noise_seed))
                init_noise = torch.randn(1, horizon_steps, 7, device=device, generator=g)
            else:
                init_noise = None

            t0 = time.monotonic()
            with torch.no_grad():
                sample = model(cond=cond, init_noise=init_noise, deterministic=True)
            actions_n = sample.trajectories[0].cpu().numpy()           # (16, 7) in [-1, 1]
            actions = action_norm.denormalize(actions_n)               # absolute joint targets
            infer_ms = (time.monotonic() - t0) * 1000.0

            # Focused gripper diagnostics: highlight the 7th dimension across
            # the predicted chunk. Convention: 1 = OPEN, 0 = CLOSED.
            chunk_grip = actions[:act_steps, 6]
            grip_now = q_now[6]
            grip_min = float(chunk_grip.min())
            grip_max = float(chunk_grip.max())
            grip_summary = (
                f"grip now={grip_now:.3f}  pred_chunk[6]=["
                + ",".join(f"{v:.2f}" for v in chunk_grip)
                + f"]  min={grip_min:.3f} max={grip_max:.3f}"
            )
            if args.print_actions:
                log.info("[ep %d step %d] infer=%.1fms  q_now=%s  pred_chunk[0]=%s ... [act-1]=%s",
                         ep+1, step, infer_ms,
                         np.round(q_now, 3).tolist(),
                         np.round(actions[0], 3).tolist(),
                         np.round(actions[act_steps - 1], 3).tolist())
            else:
                log.info("[ep %d step %d] infer=%.1fms  q_now=%s",
                         ep+1, step, infer_ms, np.round(q_now, 3).tolist())
            log.info("    %s", grip_summary)

            # Execute first act_steps actions at control_hz, paced from RIGHT
            # NOW (after inference) so the chunk doesn't open with a snap.
            chunk = actions[:act_steps]
            chunk_start_t = time.monotonic()
            for i, q_target in enumerate(chunk):
                if _abort_episode["flag"]:
                    break
                target_send_t = chunk_start_t + i * period
                # Wait until this command's intended send time. If we're early,
                # sleep; if we're behind, send immediately (no catch-up rush).
                now = time.monotonic()
                if now < target_send_t:
                    time.sleep(target_send_t - now)

                # Send the policy's prediction VERBATIM. Two safety nets only:
                #   * Joint range clamp so we never command outside the YAM's
                #     mechanical limits (hardware-mandated; cannot be removed).
                #   * Optional --no-command_gripper for the rare case of running
                #     with a passive teaching handle. Default ON so the policy's
                #     full action vector reaches the motors unchanged.
                # NO per-tick delta clamp. Earlier versions had one but it
                # staircase-clipped the smooth trajectory the policy was trained
                # to produce, creating visible shake and almost certainly
                # contributing to BC's compounding error.
                q_cmd = np.asarray(q_target, dtype=np.float64).copy()
                np.clip(q_cmd, YAM_JOINT_LIMITS_LOW, YAM_JOINT_LIMITS_HIGH, out=q_cmd)
                q_cur = _read_state()
                if not args.command_gripper:
                    q_cmd[6] = q_cur[6]

                if not args.dry_run:
                    robot.command_joint_pos(q_cmd)

                # Sample the next state + images into the cond history AT 30Hz
                # (the action rate), matching how training data was sampled.
                # The dataset's `cond["state"]` is [state_{t-1}, state_t] with
                # the two frames 33ms apart (consecutive samples at 30Hz). If
                # we only refreshed the history once per chunk (every 264ms),
                # the model sees an 8x apparent velocity in state and likely
                # loses the deceleration cue that triggers gripper close.
                next_base_rgb, _ = base_cam.get()
                next_wrist_rgb, _ = wrist_cam.get()
                next_base128 = _resize_to_128(next_base_rgb)
                next_wrist128 = _resize_to_128(next_wrist_rgb)
                next_rgb_concat = np.concatenate([next_base128, next_wrist128], axis=0)
                state_hist.append(q_cur)
                img_hist.append(next_rgb_concat)
                # Pacing for the NEXT iteration is enforced at the TOP via
                # target_send_t.

        # ---- end of episode ----
        if _abort_episode["flag"]:
            log.info("episode %d aborted by user (Ctrl-C).", ep + 1)
        else:
            log.info("episode %d finished (max_steps=%d).", ep + 1, args.max_steps)

        # Ramp back to home so the next episode starts from a clean pose.
        _ramp_to_home()

        # Decide whether to prompt before the next episode.
        #   - After Ctrl-C (`_abort_episode` true): ALWAYS prompt, even with
        #     --no_prompt. Rationale: Ctrl-C is a deliberate human intervention;
        #     the operator wants to reposition / inspect / decide before retry.
        #   - After natural max_steps end: respect --no_prompt.
        is_last = (ep + 1 >= args.num_episodes)
        aborted = _abort_episode["flag"]
        should_prompt = not is_last and (aborted or not args.no_prompt)
        if should_prompt:
            tag = "ABORTED by Ctrl-C" if aborted else "done"
            try:
                ans = input(
                    f"Episode {ep + 1}/{args.num_episodes} {tag}. "
                    "Press Enter to start next trial, q+Enter to quit: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "q"
            if ans == "q":
                log.info("user requested quit.")
                break

    _final_shutdown()


if __name__ == "__main__":
    main()
