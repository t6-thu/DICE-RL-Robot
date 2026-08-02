#!/usr/bin/env python3
"""Deploy a DICE-RL-Robot diffusion policy checkpoint on the real YAM arm.

This script is the diffusion-policy counterpart of eval_flow_matching_yam.py.
It reuses all the hardware / camera / normalisation infrastructure but swaps
the model to DiffusionUnetTimmMod1Policy loaded from a DICE-RL-Robot .ckpt.

Usage:
    . /home/bike/Documents/niu/DICE-RL-Robot/prepare.sh

    python /home/bike/Documents/niu/DICE-RL-Robot/scripts/eval_dp_yam.py \
        --ckpt  ~/training_outputs/2026.05.19/<run>/checkpoints/epoch=0499-*.ckpt \
        --norm  ~/data/real_processed/yam_picknplace_arizonabottle_224/normalization.npz \
        --can_channel can_follower_l \
        --gripper_type linear_4310 \
        --base_serial  218622278369 \
        --wrist_serial 218622271309 \
        --policy_camera_order wrist_base \
        --weights ema \
        --home_joint_pos=-0.010,0.833,0.903,-0.598,-0.028,-0.029 \
        --skip_home --no_prompt --max_steps 100 --print_actions
"""

from __future__ import annotations
import argparse, logging, os, signal, sys, time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

log = logging.getLogger("eval_dp_yam")
logging.basicConfig(level=logging.INFO, format="[%(name)s %(levelname)s] %(message)s")

YAM_JOINT_LIMITS_LOW  = np.array([-2.767,-0.15,-0.15,-1.72,-1.72,-2.24, 0.0], dtype=np.float32)
YAM_JOINT_LIMITS_HIGH = np.array([ 3.28, 3.80, 3.28, 1.72, 1.72, 2.24, 1.5], dtype=np.float32)


# ---- normaliser (same as eval_flow_matching_yam.py) ----------------------

class MinMaxNorm:
    def __init__(self, lo, hi):
        self.lo = lo.astype(np.float32)
        self.hi = hi.astype(np.float32)
        self.range = (hi - lo + 1e-6).astype(np.float32)
    def normalize(self, x):
        return (2.0*(x - self.lo)/self.range - 1.0).astype(np.float32)
    def denormalize(self, x):
        return ((x + 1.0)/2.0*self.range + self.lo).astype(np.float32)


# ---- camera (same non-blocking pattern) ----------------------------------

class _SyncCamera:
    """Non-blocking RealSense reader (see eval_flow_matching_yam.py)."""
    def __init__(self, serial, w, h, fps, name):
        self.serial, self.width, self.height, self.fps, self.name = serial, w, h, fps, name
        self._pipeline = None
        self._latest = None
        self._t = 0.0
    def start(self):
        import pyrealsense2 as rs
        pipe = rs.pipeline()
        cfg  = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        pipe.start(cfg)
        try:
            frames = pipe.wait_for_frames(timeout_ms=2000)
            self._latest = np.asanyarray(frames.get_color_frame().get_data())
            self._t = time.monotonic()
        except Exception as e:
            log.warning("[%s] bootstrap: %s", self.name, e)
        self._pipeline = pipe
    def get(self):
        if self._pipeline is None:
            return None, 0.0
        try:
            f = self._pipeline.poll_for_frames()
            if f:
                c = f.get_color_frame()
                if c:
                    self._latest = np.asanyarray(c.get_data())
                    self._t = time.monotonic()
        except Exception as e:
            log.warning("[%s] poll: %s", self.name, e)
        if self._latest is None:
            return None, 0.0
        return self._latest, self._t
    def stop(self):
        if self._pipeline:
            try: self._pipeline.stop()
            except: pass
            self._pipeline = None


# ---- image preprocessing (matches training pipeline) ---------------------

def _resize_short_side_and_center_crop(rgb: np.ndarray, target: int = 256) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = max(target / w, target / h)
    nw = max(target, int(np.ceil(w * scale)))
    nh = max(target, int(np.ceil(h * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (nw - target) // 2, (nh - target) // 2
    return resized[y0:y0+target, x0:x0+target]


def _preprocess_for_policy(rgb: np.ndarray) -> np.ndarray:
    """640×480 RGB uint8 -> (3, 224, 224) float32 in [0, 1]."""
    mode = os.environ.get("YAM_IMAGE_PREPROCESS", "center_crop").strip().lower()
    if mode in ("direct", "direct_resize", "resize"):
        rgb224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        return np.transpose(rgb224, (2, 0, 1)).astype(np.float32) / 255.0
    if mode not in ("center_crop", "centercrop", "crop"):
        raise ValueError(
            f"Unknown YAM_IMAGE_PREPROCESS={mode!r}; use center_crop or direct_resize"
        )
    rgb_256 = _resize_short_side_and_center_crop(rgb, target=256)
    rgb_t = torch.from_numpy(rgb_256).permute(2, 0, 1).unsqueeze(0).float()
    rgb_t = torch.nn.functional.interpolate(rgb_t, size=(224, 224), mode="bilinear", align_corners=False)
    # The Hanoi NPZ converter quantizes the resized frame to uint8 before the
    # dataset divides by 255. Mirror that detail exactly at deployment.
    rgb_u8 = rgb_t.squeeze(0).clamp(0, 255).to(torch.uint8).numpy()
    return rgb_u8.astype(np.float32) / 255.0  # (3, 224, 224) float [0,1]


def _pack_policy_images(
    base: np.ndarray,
    wrist: np.ndarray,
    order: str,
) -> np.ndarray:
    """Pack physical camera frames into the checkpoint's rgb_0/rgb_1 order."""
    if order == "base_wrist":
        return np.concatenate([base, wrist], axis=0)
    if order == "wrist_base":
        return np.concatenate([wrist, base], axis=0)
    raise ValueError(f"unknown policy camera order: {order!r}")


def _split_physical_images(
    packed: np.ndarray,
    order: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (physical_base, physical_wrist) from a packed policy frame."""
    if order == "base_wrist":
        return packed[:3], packed[3:]
    if order == "wrist_base":
        return packed[3:], packed[:3]
    raise ValueError(f"unknown policy camera order: {order!r}")


def _camera_ages(
    base_timestamp: float,
    wrist_timestamp: float,
    now: float | None = None,
) -> tuple[float, float]:
    now = time.monotonic() if now is None else float(now)
    base_age = float("inf") if base_timestamp <= 0 else now - base_timestamp
    wrist_age = float("inf") if wrist_timestamp <= 0 else now - wrist_timestamp
    return base_age, wrist_age


# ---- model loading -------------------------------------------------------

def _select_policy_state_dict(ckpt: dict, weights: str):
    if weights not in {"ema", "model"}:
        raise ValueError(f"weights must be 'ema' or 'model', got {weights!r}")
    key = "ema_model" if weights == "ema" else "model"
    state_dicts = ckpt.get("state_dicts", {})
    if key not in state_dicts:
        available = sorted(state_dicts)
        raise KeyError(
            f"checkpoint has no {key!r} weights; available state_dicts={available}"
        )
    return key, state_dicts[key]


def load_dp_policy(ckpt_path: str, device: torch.device, weights: str = "ema"):
    # Checkpoints also contain the other policy weights and optimizer state.
    # Load the payload on CPU so those unused tensors do not consume GPU RAM.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt['cfg']
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    policy = hydra.utils.instantiate(cfg.policy)
    weights_key, state_dict = _select_policy_state_dict(ckpt, weights)
    policy.load_state_dict(state_dict)
    policy.to(device).eval()
    log.info("loaded diffusion policy weights=%s", weights_key)
    return policy, cfg


# ---- CLI -----------------------------------------------------------------

def _parse_home(s):
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(f"Need 6 floats, got: {s!r}")
    return np.array([float(p) for p in parts], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt',         required=True)
    parser.add_argument('--norm',         required=True,
                        help='normalization.npz from process_yam_dataset.py at 224×224')
    parser.add_argument('--can_channel',  default='can_follower_l')
    parser.add_argument('--gripper_type', default='linear_4310',
                        choices=['linear_4310','linear_3507','crank_4310',
                                 'yam_teaching_handle','no_gripper'])
    parser.add_argument('--base_serial',  required=True)
    parser.add_argument('--wrist_serial', required=True)
    parser.add_argument(
        '--weights',
        choices=['ema', 'model'],
        default='ema',
        help='Checkpoint weights to deploy. EMA is the training-time eval policy.',
    )
    parser.add_argument(
        '--policy_camera_order',
        choices=['base_wrist', 'wrist_base'],
        required=True,
        help=(
            'Mapping from physical cameras to policy tensor keys. '
            'base_wrist means rgb_0=base/rgb_1=wrist; '
            'wrist_base means rgb_0=wrist/rgb_1=base.'
        ),
    )
    parser.add_argument('--home_joint_pos', type=_parse_home, required=True)
    parser.add_argument('--home_gripper_pos', type=float, default=1.0)
    parser.add_argument('--device',       default='cuda')
    parser.add_argument('--control_hz',   type=float, default=30.0)
    parser.add_argument('--max_steps',    type=int, default=600)
    parser.add_argument('--num_episodes', type=int, default=999)
    parser.add_argument('--ramp_seconds', type=float, default=8.0)
    parser.add_argument('--reset_seconds',type=float, default=4.0)
    parser.add_argument('--max_home_distance', type=float, default=0.5)
    parser.add_argument(
        '--max_camera_age',
        type=float,
        default=0.5,
        help='Abort the episode if either policy camera frame is older than this many seconds.',
    )
    parser.add_argument('--reset_on_exit', action='store_true', default=True)
    parser.add_argument('--no-reset_on_exit', dest='reset_on_exit', action='store_false')
    parser.add_argument('--skip_home',    action='store_true')
    parser.add_argument('--no_prompt',    action='store_true')
    parser.add_argument('--command_gripper', action='store_true', default=True)
    parser.add_argument('--no-command_gripper', dest='command_gripper', action='store_false')
    parser.add_argument('--print_actions', action='store_true')
    parser.add_argument('--dry_run',      action='store_true')
    parser.add_argument(
        '--home_only',
        action='store_true',
        help='Move to the configured home pose and exit without policy execution.',
    )
    parser.add_argument('--dump_obs_dir', default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    log.info("device=%s  dry_run=%s  command_gripper=%s", device, args.dry_run, args.command_gripper)

    if args.dump_obs_dir:
        os.makedirs(args.dump_obs_dir, exist_ok=True)

    # ---- normaliser (dice-rl space ↔ i2rt raw) --------------------------
    n    = np.load(args.norm)
    sn   = MinMaxNorm(n['obs_min'],    n['obs_max'])     # state  normalise
    an   = MinMaxNorm(n['action_min'], n['action_max'])  # action denormalise

    # ---- load diffusion policy ------------------------------------------
    log.info("loading policy from %s …", args.ckpt)
    policy, cfg = load_dp_policy(args.ckpt, device, weights=args.weights)
    obs_horizon    = cfg.task.obs_horizon     # 2
    action_horizon = cfg.task.action_horizon  # 16
    act_steps      = action_horizon           # execute the full chunk before re-querying

    # GPU warm-up (prevents GIL starvation of CAN control thread on first forward)
    log.info("warming up policy (prevents GIL starvation on first inference)…")
    _w_rgb   = torch.zeros(1, obs_horizon, 3, 224, 224, device=device)
    _w_joint = torch.zeros(1, obs_horizon, 7, device=device)
    with torch.no_grad():
        for _ in range(2):
            policy.predict_action({"sparse": {"rgb_0": _w_rgb, "rgb_1": _w_rgb, "joint_pos": _w_joint}})
    torch.cuda.synchronize()
    log.info("warmup done.")

    # Start and validate cameras before opening the robot. RealSense startup can
    # block for seconds; doing it after the YAM control thread starts can starve
    # the motor watchdog and latch a "loss communication" error.
    base_cam  = _SyncCamera(args.base_serial,  640, 480, 30, "base")
    wrist_cam = _SyncCamera(args.wrist_serial, 640, 480, 30, "wrist")
    try:
        base_cam.start()
        wrist_cam.start()
    except Exception:
        base_cam.stop()
        wrist_cam.stop()
        raise
    deadline = time.monotonic() + 8.0
    base_age = wrist_age = float("inf")
    while time.monotonic() < deadline:
        base_frame, base_t = base_cam.get()
        wrist_frame, wrist_t = wrist_cam.get()
        base_age, wrist_age = _camera_ages(base_t, wrist_t)
        if (
            base_frame is not None
            and wrist_frame is not None
            and base_age <= args.max_camera_age
            and wrist_age <= args.max_camera_age
        ):
            break
        time.sleep(0.05)
    if base_age > args.max_camera_age or wrist_age > args.max_camera_age:
        base_cam.stop()
        wrist_cam.stop()
        raise RuntimeError(
            "cameras failed freshness check: "
            f"base_age={base_age:.3f}s wrist_age={wrist_age:.3f}s "
            f"limit={args.max_camera_age:.3f}s"
        )
    log.info(
        "both cameras streaming; base_age=%.3fs wrist_age=%.3fs",
        base_age,
        wrist_age,
    )

    # ---- hardware --------------------------------------------------------
    from i2rt.robots.get_robot import get_yam_robot, GripperType
    gripper_type = GripperType.from_string_name(args.gripper_type)
    log.info("opening YAM on %s with gripper=%s", args.can_channel, gripper_type)
    try:
        robot = get_yam_robot(
            channel=args.can_channel,
            gripper_type=gripper_type,
            zero_gravity_mode=True,
        )
    except Exception:
        base_cam.stop()
        wrist_cam.stop()
        raise

    # ---- state reader ----------------------------------------------------
    def _read_state() -> np.ndarray:
        obs = robot.get_observations()
        j   = np.asarray(obs['joint_pos'], dtype=np.float32)  # 6-D arm
        g   = np.asarray(obs.get('gripper_pos', [args.home_gripper_pos]), dtype=np.float32).reshape(-1)
        return np.concatenate([j[:6], g[:1]])  # 7-D raw i2rt

    # ---- shutdown / ramp helpers ----------------------------------------
    _abort = {"flag": False}
    def _ramp_to_home():
        if args.dry_run or not args.reset_on_exit: return
        try:
            home = np.concatenate([args.home_joint_pos, [args.home_gripper_pos]]).astype(np.float64)
            cur  = robot.get_joint_pos().astype(np.float64)
            home[6] = cur[6]  # keep gripper at current
            log.info("ramp to home (%.1fs)…", args.reset_seconds)
            robot.move_joints(home, time_interval_s=args.reset_seconds)
            log.info("at home.")
        except Exception as e:
            log.warning("ramp failed: %s", e)

    def _shutdown(*_):
        if _abort["flag"]:
            log.warning("force quit"); base_cam.stop(); wrist_cam.stop(); os._exit(130)
        log.info("Ctrl-C: ending episode…"); _abort["flag"] = True

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    def _close_hardware():
        try:
            base_cam.stop()
            wrist_cam.stop()
        except Exception:
            pass
        try:
            robot.close()
        except Exception as e:
            log.warning("robot close failed: %s", e)

    if args.home_only:
        q_now = _read_state()
        far = float(np.abs(q_now[:6] - args.home_joint_pos).max())
        home = np.concatenate(
            [args.home_joint_pos, [q_now[6]]],
        ).astype(np.float64)
        log.info(
            "home-only: current=%s target=%s max|delta|=%.3f rad",
            np.round(q_now, 3).tolist(),
            np.round(home, 3).tolist(),
            far,
        )
        if far > args.max_home_distance:
            _close_hardware()
            raise RuntimeError(
                f"Refusing home-only move: max|delta|={far:.3f} rad exceeds "
                f"--max_home_distance={args.max_home_distance:.3f}. "
                "Inspect the robot and increase the threshold explicitly only "
                "if the path is clear."
            )
        if not args.no_prompt:
            answer = input(
                f"Type MOVE to command a {args.ramp_seconds:.1f}s ramp to home: "
            ).strip()
            if answer != "MOVE":
                log.info("home-only cancelled; no motion commanded.")
                _close_hardware()
                return
        if args.dry_run:
            log.info("home-only dry-run: no motion commanded.")
        else:
            robot.move_joints(home, time_interval_s=args.ramp_seconds)
            log.info("home-only complete. q=%s", np.round(_read_state(), 3).tolist())
        _close_hardware()
        return

    period = 1.0 / args.control_hz

    # ---- episode loop ----------------------------------------------------
    state_hist = deque(maxlen=obs_horizon)
    img_hist   = deque(maxlen=obs_horizon)

    for ep in range(args.num_episodes):
        _abort["flag"] = False
        log.info("=== episode %d ===", ep + 1)

        # auto-home or skip
        home_full = np.concatenate([args.home_joint_pos, [args.home_gripper_pos]]).astype(np.float32)
        if args.skip_home:
            q_arm = _read_state()[:6]
            far   = float(np.abs(q_arm - args.home_joint_pos).max())
            log.info("skip_home: max|delta|=%.3f rad", far)
        else:
            q_now = _read_state()[:6]
            far   = float(np.abs(q_now - args.home_joint_pos).max())
            log.info("auto-home (far=%.3f rad, ramp=%.1fs)…", far, args.ramp_seconds)
            if far > args.max_home_distance:
                log.error("far=%.3f > --max_home_distance %.3f; refuse. Use --skip_home or increase threshold.", far, args.max_home_distance)
                continue
            if not args.dry_run:
                robot.move_joints(home_full, time_interval_s=args.ramp_seconds)
                time.sleep(0.3)

        # pre-fill history
        q0 = _read_state()
        b0, bt0 = base_cam.get(); w0, wt0 = wrist_cam.get()
        base_age, wrist_age = _camera_ages(bt0, wt0)
        if (
            b0 is None
            or w0 is None
            or base_age > args.max_camera_age
            or wrist_age > args.max_camera_age
        ):
            _close_hardware()
            raise RuntimeError(
                "camera frame stale before episode: "
                f"base_age={base_age:.3f}s wrist_age={wrist_age:.3f}s "
                f"limit={args.max_camera_age:.3f}s"
            )
        b0 = _preprocess_for_policy(b0);  w0 = _preprocess_for_policy(w0)
        img0 = _pack_policy_images(b0, w0, args.policy_camera_order)
        log.info("policy_camera_order=%s", args.policy_camera_order)
        for _ in range(obs_horizon):
            state_hist.append(q0.copy()); img_hist.append(img0.copy())

        in_range = np.all((q0 >= sn.lo - 0.05) & (q0 <= sn.hi + 0.05))
        if not in_range:
            log.warning("start state outside training range — policy may behave oddly")
        else:
            log.info("start state is inside training range.")

        if args.no_prompt:
            log.info("starting eval loop immediately.")
        else:
            input("Press Enter to start (Ctrl-C to abort).")

        # ---- inner step loop ------------------------------------------
        for step in range(args.max_steps):
            if _abort["flag"]: break

            # Fail closed if either RealSense pipeline has stopped updating.
            _, base_t = base_cam.get()
            _, wrist_t = wrist_cam.get()
            base_age, wrist_age = _camera_ages(base_t, wrist_t)
            if base_age > args.max_camera_age or wrist_age > args.max_camera_age:
                log.error(
                    "stale camera before inference: base_age=%.3fs wrist_age=%.3fs "
                    "limit=%.3fs; aborting episode",
                    base_age,
                    wrist_age,
                    args.max_camera_age,
                )
                _abort["flag"] = True
                break

            # build cond from history (same roll-out pattern as flow-matching eval)
            q_hist_n = sn.normalize(np.stack(list(state_hist)))   # (To, 7) [-1,1]
            img_arr  = np.stack(list(img_hist))                    # (To, 6, 224, 224) [0,1]
            rgb0_t = torch.from_numpy(img_arr[:, :3])[None].to(device).float()  # (1,To,3,224,224)
            rgb1_t = torch.from_numpy(img_arr[:, 3:])[None].to(device).float()
            jnt_t  = torch.from_numpy(q_hist_n)[None].to(device).float()        # (1,To,7)

            obs_dict = {"sparse": {"rgb_0": rgb0_t, "rgb_1": rgb1_t, "joint_pos": jnt_t}}

            # diffusion inference
            t0 = time.monotonic()
            with torch.no_grad():
                result = policy.predict_action(obs_dict)
            infer_ms = (time.monotonic() - t0) * 1000.0
            # result["sparse"] is in dice-rl [-1,1] action space; denorm to i2rt raw
            actions_n = result["sparse"][0].cpu().numpy()  # (action_horizon, 7)
            actions   = an.denormalize(actions_n)          # (action_horizon, 7) i2rt raw
            g_norm = actions_n[:act_steps, 6]
            g_raw = actions[:act_steps, 6]

            if args.print_actions:
                log.info("[ep %d step %d] infer=%.1fms  q=%s  "
                         "grip_norm[min,max,last]=[%.3f, %.3f, %.3f] "
                         "grip_raw[min,max,last]=[%.3f, %.3f, %.3f]  "
                         "pred[0]=%s … [act-1]=%s",
                         ep+1, step, infer_ms,
                         np.round(_read_state(), 3).tolist(),
                         float(g_norm.min()), float(g_norm.max()), float(g_norm[-1]),
                         float(g_raw.min()), float(g_raw.max()), float(g_raw[-1]),
                         np.round(actions[0], 3).tolist(),
                         np.round(actions[act_steps-1], 3).tolist())
            else:
                log.info("[ep %d step %d] infer=%.1fms  q=%s  "
                         "grip_raw[min,max,last]=[%.3f, %.3f, %.3f]",
                         ep+1, step, infer_ms,
                         np.round(_read_state(), 3).tolist(),
                         float(g_raw.min()), float(g_raw.max()), float(g_raw[-1]))

            # optional obs dump
            if args.dump_obs_dir:
                physical_base, physical_wrist = _split_physical_images(
                    img_arr[-1], args.policy_camera_order
                )
                b_hwc = np.transpose(physical_base, (1,2,0))
                w_hwc = np.transpose(physical_wrist, (1,2,0))
                cv2.imwrite(os.path.join(args.dump_obs_dir, f"step_{step:04d}_base.jpg"),
                            cv2.cvtColor((b_hwc*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(args.dump_obs_dir, f"step_{step:04d}_wrist.jpg"),
                            cv2.cvtColor((w_hwc*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            # execute chunk, sampling history at 30 Hz inside inner loop
            chunk_start_t = time.monotonic()
            for i, q_target in enumerate(actions[:act_steps]):
                if _abort["flag"]: break
                target_t = chunk_start_t + i * period
                now = time.monotonic()
                if now < target_t: time.sleep(target_t - now)

                q_cmd = np.asarray(q_target, dtype=np.float64).copy()
                np.clip(q_cmd, YAM_JOINT_LIMITS_LOW, YAM_JOINT_LIMITS_HIGH, out=q_cmd)
                q_cur = _read_state()
                if not args.command_gripper: q_cmd[6] = q_cur[6]
                if not args.dry_run: robot.command_joint_pos(q_cmd)

                # update history at 30 Hz
                br, br_t = base_cam.get(); wr, wr_t = wrist_cam.get()
                base_age, wrist_age = _camera_ages(br_t, wr_t)
                if base_age > args.max_camera_age or wrist_age > args.max_camera_age:
                    log.error(
                        "stale camera during action chunk: base_age=%.3fs "
                        "wrist_age=%.3fs limit=%.3fs; aborting episode",
                        base_age,
                        wrist_age,
                        args.max_camera_age,
                    )
                    _abort["flag"] = True
                    break
                if br is not None: br = _preprocess_for_policy(br)
                if wr is not None: wr = _preprocess_for_policy(wr)
                if br is not None and wr is not None:
                    state_hist.append(q_cur.copy())
                    img_hist.append(
                        _pack_policy_images(br, wr, args.policy_camera_order)
                    )

        # ---- end of episode ------------------------------------------
        log.info("episode %d ended (%s)", ep+1, "aborted" if _abort["flag"] else "max_steps")
        _ramp_to_home()

        is_last = (ep + 1 >= args.num_episodes)
        aborted = _abort["flag"]
        if not is_last and (aborted or not args.no_prompt):
            tag = "ABORTED" if aborted else "done"
            try:
                ans = input(f"Episode {ep+1}/{args.num_episodes} {tag}. "
                            "Enter=next  q=quit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "q"
            if ans == "q": break

    log.info("shutting down…")
    _close_hardware()


if __name__ == '__main__':
    main()
