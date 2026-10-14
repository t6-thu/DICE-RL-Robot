"""DICE-RL Environment Runner for YAM joint-space diffusion policy.

Algorithm (one episode)
-----------------------
1. Observe (rgb_0, rgb_1, joint_pos) history via cameras + i2rt.
2. BC forward pass:
     features = bc_policy.extract_visual_features(obs)
     bc_action = bc_policy.predict_action_from_features(features)
3. Residual RL correction (if actor is available):
     noise     = randn(1, action_horizon, 7)
     delta     = residual_actor(features, noise)   # in normalized space
     final_act = bc_action + delta
4. Denormalize → i2rt joint targets → command_joint_pos().
   Real-time chunking: the loop ticks at control_hz and never blocks on the
   policy. The next chunk is inferred in a worker thread while the current one
   executes; the steps still to run are inpainted as a fixed prefix so the
   hand-over is continuous (dice_rl/env_runner/rtc.py). Frames where no fresh
   chunk was available are flagged action_valid=0 in the recording.
5. After episode: user labels success/failure → reward = 1.0 / 0.0.
6. Send episode (images, states, actions, rewards) to learner via ZMQ.
7. Receive updated actor weights from learner.
"""

from __future__ import annotations
import glob
import logging
import os
import pickle
import signal
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
import torch

from dice_rl.communication.actor_node import Actor
from dice_rl.env_runner.rtc import RTCScheduler
from dice_rl.model.distill_rl import DistilledActor
from utils.model_io import load_policy

log = logging.getLogger(__name__)

# ---- image helpers (same as eval_dp_yam.py) ----

def _short_side_crop(rgb: np.ndarray, target: int = 256) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = max(target / w, target / h)
    nw = max(target, int(np.ceil(w * scale)))
    nh = max(target, int(np.ceil(h * scale)))
    r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (nw - target) // 2, (nh - target) // 2
    return r[y0:y0+target, x0:x0+target]


def _preprocess(rgb: np.ndarray) -> np.ndarray:
    """640×480 RGB uint8 → (3, 224, 224) float32 [0,1]."""
    mode = os.environ.get("YAM_IMAGE_PREPROCESS", "center_crop").strip().lower()
    if mode in ("direct", "direct_resize", "resize"):
        rgb224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        return np.transpose(rgb224, (2, 0, 1)).astype(np.float32) / 255.0
    if mode not in ("center_crop", "centercrop", "crop"):
        raise ValueError(
            f"Unknown YAM_IMAGE_PREPROCESS={mode!r}; use center_crop or direct_resize"
        )
    rgb256 = _short_side_crop(rgb, 256)
    t = torch.from_numpy(rgb256).permute(2,0,1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, (224,224), mode="bilinear", align_corners=False)
    # Match process_hanoi_hdf5_to_npz.py exactly: bilinear resize, quantize to
    # uint8, then divide by 255 in the dataset / online policy input.
    rgb_u8 = t.squeeze(0).clamp(0,255).to(torch.uint8).numpy()
    return rgb_u8.astype(np.float32) / 255.0


def _pack_policy_images(base: np.ndarray, wrist: np.ndarray, order: str) -> np.ndarray:
    """Pack physical cameras into the checkpoint's rgb_0/rgb_1 channel order."""
    if order == "base_wrist":
        return np.concatenate([base, wrist], axis=0)
    if order == "wrist_base":
        return np.concatenate([wrist, base], axis=0)
    raise ValueError(f"unknown policy camera order: {order!r}")


def _camera_ages(base_t: float, wrist_t: float, now: float = None):
    now = time.monotonic() if now is None else float(now)
    base_age = float("inf") if base_t <= 0 else now - base_t
    wrist_age = float("inf") if wrist_t <= 0 else now - wrist_t
    return base_age, wrist_age


# ---- non-blocking camera (same as eval_dp_yam.py) ----

class _SyncCamera:
    def __init__(self, serial, w, h, fps, name):
        self.serial, self.width, self.height, self.fps, self.name = serial, w, h, fps, name
        self._pipe, self._latest, self._t = None, None, 0.0

    def _usb_type(self):
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            for dev in ctx.query_devices():
                if dev.get_info(rs.camera_info.serial_number) == self.serial:
                    return dev.get_info(rs.camera_info.usb_type_descriptor)
        except Exception:
            pass
        return "unknown"

    def start(self):
        import pyrealsense2 as rs
        usb = self._usb_type()
        if not str(usb).startswith("3"):
            log.warning(
                "camera %s (%s) is on USB=%s; D405 should use USB3 for reliable streaming",
                self.name, self.serial, usb)
        p = rs.pipeline(); c = rs.config()
        c.enable_device(self.serial)
        c.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        p.start(c)
        self._pipe = p
        # USB-2 cameras (the wrist D405 enumerates at USB 2.1) can take several
        # seconds and a few retries before the first frame arrives after a fresh
        # pipeline start. Retry up to ~12s instead of giving up after one 2s wait,
        # so env_runner startup doesn't crash on a slow-to-stream wrist cam.
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            try:
                f = p.wait_for_frames(timeout_ms=2000)
                cf = f.get_color_frame()
                if cf:
                    self._latest = np.asanyarray(cf.get_data())
                    self._t = time.monotonic()
                    return
            except Exception:
                time.sleep(0.3)
        self.stop()
        raise RuntimeError(
            f"camera {self.name} ({self.serial}) produced no frame within 12s "
            f"at {self.width}x{self.height}@{self.fps} rgb8; USB={usb}. "
            "Check cable/port and make sure it enumerates as USB3."
        )
    def get(self):
        if not self._pipe: return None, 0.0
        try:
            f = self._pipe.poll_for_frames()
            if f:
                c = f.get_color_frame()
                if c:
                    self._latest = np.asanyarray(c.get_data())
                    self._t = time.monotonic()
        except Exception: pass
        return self._latest, self._t
    def stop(self):
        if self._pipe:
            try: self._pipe.stop()
            except: pass
            self._pipe = None


class YAMRLEnvRunner:
    """Collects online RL rollouts on the real YAM arm."""

    def __init__(
        self,
        # Policy
        pretrained_policy_ckpt: str,
        norm_npz_path: str,
        # Cameras
        base_cam_serial: str,
        wrist_cam_serial: str,
        policy_camera_order: str = "base_wrist",
        # YAM hardware
        can_channel: str = "can_follower_l",
        gripper_type: str = "linear_4310",
        home_joint_pos: list = None,
        home_gripper_pos: float = 1.0,
        # Control
        control_hz: float = 30.0,
        max_episode_steps: int = 200,
        max_camera_age: float = 0.5,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        action_dim: int = 7,
        # Actor (residual RL)
        actor_hidden_dims: list = None,
        residual_scale: float = 1.0,  # scale on actor's delta — set <1 to soften RL effect
        max_joint_step: float = 0.08,
        raw_policy: bool = False,
        # Real-time chunking
        rtc_delay_steps: int = 3,
        gil_switch_interval_s: float = 0.001,
        # Data & ZMQ
        online_data_dir: str = "/tmp/yam_rl_rollouts",
        rl_checkpoint_dir: str = None,
        network_server_endpoint: str = "ipc:///tmp/feeds/rl_weights",
        network_weight_topic: str = "rl_network_weights_topic",
        transitions_server_endpoint: str = "ipc:///tmp/feeds/rl_transitions",
        transitions_topic: str = "rl_transitions_topic",
        # Misc
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.control_hz = control_hz
        self.period = 1.0 / control_hz
        self.max_episode_steps = max_episode_steps
        if policy_camera_order not in {"base_wrist", "wrist_base"}:
            raise ValueError(
                "policy_camera_order must be 'base_wrist' or 'wrist_base', "
                f"got {policy_camera_order!r}"
            )
        self.policy_camera_order = policy_camera_order
        self.max_camera_age = float(max_camera_age)
        self.home_joint_pos = np.array(home_joint_pos or [-0.01,0.833,0.903,-0.598,-0.028,-0.029],
                                       dtype=np.float32)
        self.home_gripper_pos = home_gripper_pos
        self.raw_policy = bool(raw_policy)
        self.residual_scale = None if self.raw_policy else float(residual_scale)
        self.max_joint_step = 0.0 if self.raw_policy else float(max_joint_step)
        self._last_delta_rms = 0.0
        # RTC: inference budget in frames. The next chunk is requested when this
        # many steps of the current chunk remain; those steps are frozen as the
        # inpainting prefix. Inference slower than rtc_delay_steps/control_hz
        # produces held frames (action_valid=0) instead of a stall.
        self.rtc_delay_steps = int(rtc_delay_steps)
        if not (0 <= self.rtc_delay_steps <= action_horizon - 2):
            raise ValueError(
                f"rtc_delay_steps must be in [0, {action_horizon - 2}], got {rtc_delay_steps}")
        # Inference runs in one worker thread. i2rt's CAN control thread needs the
        # GIL every ~4 ms; a shorter switch interval bounds how long the inference
        # thread can hold it (default 5 ms would exceed the CAN period).
        if gil_switch_interval_s and gil_switch_interval_s > 0:
            sys.setswitchinterval(float(gil_switch_interval_s))
        self._infer_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc_infer")
        self.online_data_dir = online_data_dir
        self.rl_checkpoint_dir = rl_checkpoint_dir
        self._latest_weights_path = (
            os.path.join(rl_checkpoint_dir, "latest_actor.pt") if rl_checkpoint_dir else None
        )
        self._last_weights_mtime = 0.0
        os.makedirs(online_data_dir, exist_ok=True)
        if self.raw_policy:
            log.info(
                "Env runner control: control_hz=%.1f policy_mode=raw "
                "residual_scale=off max_joint_step=off "
                "max_episode_steps=%d action_horizon=%d",
                self.control_hz, self.max_episode_steps, self.action_horizon,
            )
        else:
            log.info(
                "Env runner control: control_hz=%.1f residual_scale=%.3f "
                "max_joint_step=%.3f max_episode_steps=%d action_horizon=%d",
                self.control_hz, self.residual_scale, self.max_joint_step,
                self.max_episode_steps, self.action_horizon)
        log.info(
            "RTC: delay budget=%d frames (%.0f ms), max frames/episode=%d, "
            "GIL switch interval=%.1f ms",
            self.rtc_delay_steps, self.rtc_delay_steps * self.period * 1000.0,
            self.max_episode_steps * action_horizon,
            sys.getswitchinterval() * 1000.0,
        )
        log.info(
            "Image preprocessing: %s",
            os.environ.get("YAM_IMAGE_PREPROCESS", "center_crop").strip().lower(),
        )
        log.info(
            "Policy camera order: %s (rgb_0=%s rgb_1=%s)",
            self.policy_camera_order,
            "wrist" if self.policy_camera_order == "wrist_base" else "base",
            "base" if self.policy_camera_order == "wrist_base" else "wrist",
        )

        # ---- normalisation ----
        n = np.load(norm_npz_path)
        from scripts.eval_flow_matching_yam import MinMaxNorm
        self.state_norm  = MinMaxNorm(n["obs_min"],    n["obs_max"])
        self.action_norm = MinMaxNorm(n["action_min"], n["action_max"])

        # ---- load frozen BC policy ----
        log.info("Loading BC policy from %s", pretrained_policy_ckpt)
        self.bc_policy, _, bc_cfg = load_policy(pretrained_policy_ckpt, device)
        self.bc_policy.eval()
        for p in self.bc_policy.parameters():
            p.requires_grad = False
        obs_feature_dim = self.bc_policy.obs_feature_dim
        log.info(
            "BC policy obs_feature_dim=%d weights=%s",
            obs_feature_dim,
            "ema_model" if bc_cfg.training.use_ema else "model",
        )

        # ---- residual actor (starts as None; filled when weights arrive) ----
        self.actor: Optional[DistilledActor] = None
        self._actor_hidden_dims = actor_hidden_dims or [1024, 1024, 1024]
        self._obs_feature_dim = obs_feature_dim

        # ---- warm up GPU ----
        _w = torch.zeros(1, obs_horizon, 3, 224, 224, device=self.device)
        _j = torch.zeros(1, obs_horizon, 7, device=self.device)
        with torch.no_grad():
            for _ in range(2):
                self.bc_policy.predict_action({"sparse": {"rgb_0": _w, "rgb_1": _w, "joint_pos": _j}})
        torch.cuda.synchronize()
        log.info("GPU warmup done.")

        # ---- cameras ----
        # Validate cameras before touching the robot. If a RealSense cable/port
        # is bad, fail here instead of calibrating the gripper and leaving robot
        # background control threads running after startup aborts.
        self.robot = None
        self.base_cam  = _SyncCamera(base_cam_serial,  640, 480, 30, "base")
        self.wrist_cam = _SyncCamera(wrist_cam_serial, 640, 480, 30, "wrist")
        try:
            self.base_cam.start(); self.wrist_cam.start()
        except Exception:
            self.base_cam.stop()
            self.wrist_cam.stop()
            raise
        deadline = time.monotonic() + 8.0
        base_age = wrist_age = float("inf")
        while time.monotonic() < deadline:
            b, bt = self.base_cam.get()
            w, wt = self.wrist_cam.get()
            base_age, wrist_age = _camera_ages(bt, wt)
            if (
                b is not None and w is not None
                and base_age <= self.max_camera_age
                and wrist_age <= self.max_camera_age
            ):
                break
            time.sleep(0.05)
        if base_age > self.max_camera_age or wrist_age > self.max_camera_age:
            self.base_cam.stop()
            self.wrist_cam.stop()
            raise RuntimeError(
                "camera freshness check failed before robot startup: "
                f"base_age={base_age:.3f}s wrist_age={wrist_age:.3f}s "
                f"limit={self.max_camera_age:.3f}s"
            )
        log.info(
            "Cameras streaming: base_age=%.3fs wrist_age=%.3fs",
            base_age,
            wrist_age,
        )

        # ---- hardware ----
        from i2rt.robots.get_robot import get_yam_robot, GripperType
        try:
            self.robot = get_yam_robot(channel=can_channel,
                                       gripper_type=GripperType.from_string_name(gripper_type),
                                       zero_gravity_mode=True)
        except Exception:
            self.base_cam.stop()
            self.wrist_cam.stop()
            raise

        # ---- ZMQ ----
        try:
            self.actor_node = Actor(
                network_server_endpoint=network_server_endpoint,
                network_weight_topic=network_weight_topic,
                transitions_server_endpoint=transitions_server_endpoint,
                transitions_topic=transitions_topic,
                transitions_topic_expire_time_s=3600,
            )
        except Exception:
            self.base_cam.stop()
            self.wrist_cam.stop()
            self.robot.close()
            raise

        self._abort_episode  = {"flag": False}
        self._in_episode     = False
        self._closed         = False
        self._last_sigint_t  = 0.0   # debounce: ignore rapid duplicate SIGINTs
        self._actor_step     = 0
        signal.signal(signal.SIGINT,  self._sigint)
        signal.signal(signal.SIGTERM, self._sigterm)
        signal.signal(signal.SIGQUIT, self._sigquit)  # Ctrl-\  → instant kill

    # ---- helpers ----

    def _read_state(self) -> np.ndarray:
        obs = self.robot.get_observations()
        j = np.asarray(obs["joint_pos"], dtype=np.float32)
        g = np.asarray(obs.get("gripper_pos", [self.home_gripper_pos]), dtype=np.float32).reshape(-1)
        return np.concatenate([j[:6], g[:1]])

    def _make_obs_tensors(self, img_hist, state_hist):
        img_arr = np.stack(list(img_hist))  # (To, 6, H, W)
        s_arr   = np.stack(list(state_hist))
        s_norm  = self.state_norm.normalize(s_arr)
        rgb0 = torch.from_numpy(img_arr[:, :3])[None].to(self.device).float()
        rgb1 = torch.from_numpy(img_arr[:, 3:])[None].to(self.device).float()
        jnt  = torch.from_numpy(s_norm)[None].to(self.device).float()
        return {"sparse": {"rgb_0": rgb0, "rgb_1": rgb1, "joint_pos": jnt}}

    def _infer(self, obs_tensors, prefix_n: Optional[np.ndarray] = None) -> np.ndarray:
        """Run BC policy (+ optional residual actor) → (action_horizon, 7) raw i2rt.

        prefix_n: optional (n, 7) NORMALIZED actions already committed for the
        first n steps of the chunk (RTC). They are inpainted into the BC sample
        and the residual is zeroed on those steps so the returned chunk starts
        with exactly these actions.
        """
        # _make_obs_tensors already maps joint_pos into the normalized RL action
        # space used by the residual actor. Applying the BC policy normalizer
        # again would double-normalize HDF5-trained policies whose checkpoint
        # stores raw joint min/max.
        nobs = obs_tensors["sparse"]
        features = self.bc_policy.obs_encoder(nobs)  # (1, feat_dim)

        noise = torch.randn(1, self.action_horizon, self.action_dim, device=self.device)
        prefix_t = None
        n_prefix = 0
        if prefix_n is not None and len(prefix_n) > 0:
            n_prefix = int(len(prefix_n))
            prefix_t = torch.from_numpy(np.asarray(prefix_n, dtype=np.float32))[None].to(self.device)
        bc_act_n = self.bc_policy.predict_action_from_features(
            sparse_nobs_encode=features,
            init_noise=noise,
            unnormalize=False,
            fixed_action_prefix=prefix_t,
        )["sparse"]  # (1, H, 7) normalized

        if self.actor is not None:
            delta = self.actor(features.unsqueeze(1), noise)
            if not self.raw_policy:
                delta = delta * self.residual_scale
            if n_prefix > 0:
                # Committed steps must not move: the robot is already executing them.
                delta = delta.clone()
                delta[:, :n_prefix] = 0.0
            final_n = (bc_act_n + delta)
            self._last_delta_rms = float(delta.pow(2).mean().sqrt().item())
        else:
            final_n = bc_act_n
            self._last_delta_rms = 0.0

        final_n = final_n[0].cpu().numpy()          # (H, 7) normalized
        if n_prefix > 0:
            # Guard against float round-trip drift on the frozen steps.
            final_n[:n_prefix] = prefix_n
        final   = self.action_norm.denormalize(final_n)  # (H, 7) i2rt raw
        return final

    def _infer_async(self, obs_tensors, prefix_n: Optional[np.ndarray]):
        """Submit `_infer` to the single inference worker thread.

        Returns a Future resolving to (actions_raw, infer_ms, delta_rms).
        """
        def _job():
            t0 = time.monotonic()
            with torch.inference_mode():
                act = self._infer(obs_tensors, prefix_n)  # .cpu() inside syncs the stream
            return act, (time.monotonic() - t0) * 1000.0, self._last_delta_rms
        return self._infer_pool.submit(_job)

    def _clip_cmd(self, q_tgt: np.ndarray, last_cmd: np.ndarray) -> np.ndarray:
        q_cmd = np.clip(q_tgt.astype(np.float64),
                        [-2.767,-0.15,-0.15,-1.72,-1.72,-2.24,0.],
                        [ 3.28,  3.80, 3.28, 1.72, 1.72, 2.24,1.5])
        if self.max_joint_step > 0:
            q_cmd = np.clip(
                q_cmd,
                last_cmd - self.max_joint_step,
                last_cmd + self.max_joint_step,
            )
        return q_cmd

    def _grab_frame(self, where: str):
        """Poll both cameras; abort the episode if either is stale.

        Returns (base_rgb, wrist_rgb) or None (abort flag set)."""
        br, br_t = self.base_cam.get(); wr, wr_t = self.wrist_cam.get()
        base_age, wrist_age = _camera_ages(br_t, wr_t)
        if base_age > self.max_camera_age or wrist_age > self.max_camera_age:
            log.error(
                "stale camera %s: base_age=%.3fs wrist_age=%.3fs limit=%.3fs; "
                "aborting episode", where, base_age, wrist_age, self.max_camera_age,
            )
            self._abort_episode["flag"] = True
            return None
        return br, wr

    # ---- episode loop ----

    def run_episode(self) -> dict:
        """Execute one episode with real-time chunking.

        The control loop ticks at control_hz. Every tick sends exactly one joint
        command and records exactly one transition. Inference for the next chunk
        runs in a worker thread while the current chunk keeps executing; the
        steps still to be executed are frozen as the inpainting prefix of the
        new chunk so the hand-over is seamless (see dice_rl/env_runner/rtc.py).

        Returns {images, states, actions, rewards, dones, timestamps, action_valid}.
        """
        # pre-fill history. Cameras (esp. the USB-2 wrist sharing a hub with the
        # CAN adapter) can intermittently return None; retry for up to 5s instead
        # of crashing the whole run on a single dropped frame.
        q0 = self._read_state()
        b, bt = self.base_cam.get(); w, wt = self.wrist_cam.get()
        deadline = time.monotonic() + 5.0
        base_age, wrist_age = _camera_ages(bt, wt)
        while (
            b is None or w is None
            or base_age > self.max_camera_age
            or wrist_age > self.max_camera_age
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
            b, bt = self.base_cam.get(); w, wt = self.wrist_cam.get()
            base_age, wrist_age = _camera_ages(bt, wt)
        if (
            b is None or w is None
            or base_age > self.max_camera_age
            or wrist_age > self.max_camera_age
        ):
            raise RuntimeError(
                "camera frames unavailable/stale after 5s "
                f"(base={'ok' if b is not None else 'None'}, "
                f"wrist={'ok' if w is not None else 'None'}, "
                f"base_age={base_age:.3f}s, wrist_age={wrist_age:.3f}s) — "
                "check USB connection / move wrist cam to a USB-3 port")
        b = _preprocess(b); w = _preprocess(w)
        img0 = _pack_policy_images(b, w, self.policy_camera_order)
        state_hist = deque([q0.copy()] * self.obs_horizon, maxlen=self.obs_horizon)
        img_hist   = deque([img0.copy()] * self.obs_horizon, maxlen=self.obs_horizon)
        last_cmd = q0.astype(np.float64).copy()

        images_rec, states_rec, actions_rec = [], [], []
        ts_rec, valid_rec = [], []
        infer_ms_log = []
        self._abort_episode["flag"] = False
        self._in_episode = True

        H = self.action_horizon
        max_frames = self.max_episode_steps * H
        sched = RTCScheduler(H, self.rtc_delay_steps)

        # First chunk: nothing is executing yet, so infer synchronously and
        # start the clock only once it is available. No frame is recorded
        # before the clock starts, so no gap enters the data.
        t_sync = time.monotonic()
        with torch.inference_mode():
            first = self._infer(self._make_obs_tensors(img_hist, state_hist), None)
        infer_ms_log.append((time.monotonic() - t_sync) * 1000.0)
        sched.reset(first)

        pending = None  # (future, InferenceRequest)
        t0 = time.monotonic()
        frame = 0
        try:
            while frame < max_frames and not self._abort_episode["flag"]:
                # ---- pace to the frame grid ----
                wait = t0 + frame * self.period - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                t_frame = time.monotonic()

                # ---- send this frame's command ----
                q_tgt, valid = sched.action_for_frame(frame, last_cmd)
                q_cmd = self._clip_cmd(np.asarray(q_tgt), last_cmd)
                self.robot.command_joint_pos(q_cmd)
                last_cmd = q_cmd.copy()

                # ---- observe at 30 Hz ----
                q_cur = self._read_state()
                frames = self._grab_frame(f"at frame {frame}")
                if frames is None:
                    break
                br, wr = frames
                if br is not None and wr is not None:
                    img_cur = _pack_policy_images(
                        _preprocess(br), _preprocess(wr), self.policy_camera_order
                    )
                    state_hist.append(q_cur.copy())
                    img_hist.append(img_cur.copy())

                # ---- record exactly one transition per frame ----
                images_rec.append(img_hist[-1].copy())
                states_rec.append(self.state_norm.normalize(state_hist[-1]))
                actions_rec.append(self.action_norm.normalize(q_cmd))  # (7,)
                ts_rec.append(t_frame - t0)
                valid_rec.append(bool(valid))
                if not valid:
                    log.warning("[frame %4d] no fresh chunk; holding last target "
                                "(action_valid=0)", frame)

                # ---- collect a finished inference ----
                if pending is not None and pending[0].done():
                    fut, req = pending
                    pending = None
                    try:
                        act, ms, drms = fut.result()
                    except Exception:
                        log.exception("async inference failed; will relaunch")
                        sched.cancel_in_flight()
                    else:
                        infer_ms_log.append(ms)
                        adopted = sched.on_result(frame, req, act)
                        log.info("[chunk obs@%4d] infer=%.1fms prefix=%d delta_rms=%.3f %s",
                                 req.obs_frame, ms, len(req.prefix), drms,
                                 "adopted" if adopted else "LATE/discarded")

                # ---- launch the next inference on the fresh observation ----
                if pending is None:
                    req = sched.maybe_launch(frame, q_cmd)
                    if req is not None:
                        # The scheduler keeps the prefix in raw i2rt space; the
                        # policy inpaints in normalized action space.
                        prefix_n = self.action_norm.normalize(req.prefix).astype(np.float32)
                        obs_t = self._make_obs_tensors(img_hist, state_hist)
                        pending = (self._infer_async(obs_t, prefix_n), req)

                frame += 1
        finally:
            if pending is not None:
                try:
                    pending[0].result(timeout=5.0)
                except Exception:
                    pass
            self._in_episode = False

        st = sched.stats
        ts = np.asarray(ts_rec, dtype=np.float64)
        if len(ts) > 1:
            gaps = np.diff(ts)
            n_late = int((gaps > 1.5 * self.period).sum())
            log.info(
                "Episode timing: frames=%d hold=%d launches=%d late=%d early_switch=%d "
                "infer_ms mean=%.1f max=%.1f | frame gap mean=%.1fms max=%.1fms (>1.5x period: %d)",
                st.frames, st.hold_frames, st.launches, st.late_results, st.early_switches,
                float(np.mean(infer_ms_log)), float(np.max(infer_ms_log)),
                float(gaps.mean() * 1000), float(gaps.max() * 1000), n_late,
            )
            if st.hold_frames > 0 or n_late > 0:
                log.warning(
                    "Episode had %d held frames and %d late frames: inference "
                    "exceeded its %d-frame budget or the loop stalled. Consider a larger "
                    "rtc_delay_steps.", st.hold_frames, n_late, self.rtc_delay_steps)

        # ---- user labels success/failure ----
        import termios; termios.tcflush(sys.stdin, termios.TCIFLUSH)
        print("\nEpisode ended.  s=success  f=failure  d=discard: ", end="", flush=True)
        label = input().strip().lower()
        if label == "d":
            return {"discard": True}
        reward_val = 1.0 if label == "s" else 0.0
        T = len(states_rec)
        rewards = np.zeros(T, dtype=np.float32)
        if T > 0: rewards[-1] = reward_val  # sparse reward at end of episode
        dones   = np.zeros(T, dtype=bool)
        if T > 0: dones[-1]   = True

        return {
            "discard":  False,
            "images":   np.stack(images_rec).astype(np.float32),  # (T,6,H,W) [0,1]
            "states":   np.stack(states_rec).astype(np.float32),  # (T,7) norm
            "actions":  np.stack(actions_rec).astype(np.float32), # (T,7) norm
            "rewards":  rewards,
            "dones":    dones,
            "timestamps": ts,                                     # (T,) s since episode start
            "action_valid": np.asarray(valid_rec, dtype=bool),    # (T,)
            "success":  reward_val > 0.5,
            "policy_camera_order": np.asarray(self.policy_camera_order),
        }

    # ---- main loop ----

    def _move_to_home(self, ramp_seconds: float = 6.0) -> None:
        """Smooth ramp to home pose using i2rt move_joints (50-step interpolation)."""
        h = np.concatenate([self.home_joint_pos, [self.home_gripper_pos]])
        log.info("Moving to home pose (%.1fs ramp)…", ramp_seconds)
        self.robot.move_joints(h.astype(np.float64), time_interval_s=ramp_seconds)
        time.sleep(0.3)
        log.info("At home. q=%s", np.round(self._read_state(), 3).tolist())

    def close(self) -> None:
        """Stop cameras and release robot torque exactly once."""
        if self._closed:
            return
        self._closed = True
        try:
            self._infer_pool.shutdown(wait=False)
        except Exception:
            pass
        try:
            self.base_cam.stop()
            self.wrist_cam.stop()
        except Exception as exc:
            log.warning("camera shutdown failed: %s", exc)
        if self.robot is not None:
            try:
                self.robot.close()
            except Exception as exc:
                log.warning("robot shutdown failed: %s", exc)

    def run(self) -> None:
        try:
            self._run_loop()
        except KeyboardInterrupt:
            log.info("Interrupted; shutting down safely.")
        finally:
            self.close()

    def _run_loop(self) -> None:
        log.info("Env runner ready.")

        # Require an explicit acknowledgement before the first physical move.
        q_now = self._read_state()
        home = np.concatenate([self.home_joint_pos, [self.home_gripper_pos]])
        max_delta = float(np.max(np.abs(q_now - home)))
        log.info("Initial home check: max|delta|=%.3f rad", max_delta)
        confirm = input("Type MOVE to command a 6.0s ramp to home: ").strip()
        if confirm != "MOVE":
            raise RuntimeError("home motion cancelled (expected exact input: MOVE)")
        self._move_to_home()

        # Count already-saved episodes so numbering stays consistent across restarts.
        ep = len(glob.glob(os.path.join(self.online_data_dir, "episode_*.npz")))
        if ep > 0:
            log.info("Resuming: %d episodes already saved in %s", ep, self.online_data_dir)

        while True:
            # Check for updated actor weights from learner.
            self._try_update_actor()
            actor_info = (f"RL actor step={self._actor_step}"
                         if self.actor is not None else "pure BC")
            input(f"\n[Episode {ep+1} | {actor_info}] Press Enter to start, or Ctrl-C to exit.")
            # The learner may finish a training round while the operator is
            # waiting at this prompt. Recheck immediately before rollout so
            # episode 21 does not accidentally remain pure BC.
            self._try_update_actor()
            ep_data = self.run_episode()
            if ep_data.get("discard"):
                log.info("Episode discarded.")
                self._move_to_home()
                continue

            # Save episode to disk (survives learner/runner crashes).
            # Images stored as uint8 [0,255] for compact storage (~4× smaller
            # than float32, matches what RealSense produces natively).
            # replay_buffer._make_obs and hire_shaper auto-detect dtype and
            # convert back to float32 [0,1] on load.
            ep_path = os.path.join(self.online_data_dir, f"episode_{ep:04d}.npz")
            np.savez_compressed(ep_path,
                                images=(ep_data["images"] * 255.0).clip(0, 255).astype(np.uint8),
                                states=ep_data["states"],
                                actions=ep_data["actions"],
                                rewards=ep_data["rewards"],
                                dones=ep_data["dones"],
                                timestamps=ep_data["timestamps"],
                                action_valid=ep_data["action_valid"],
                                policy_camera_order=np.asarray(self.policy_camera_order))

            # Disk is the authoritative transport for YAMRLLearner. Avoid also
            # pickling/sending the float32 image tensor through ZMQ: a full
            # real-robot episode can be hundreds of MB and the learner's active
            # loop intentionally does not consume that queue.
            log.info("Episode %d saved (success=%s)", ep+1, ep_data["success"])
            ep += 1

            # Auto-home at end of every episode (ready for the next one).
            self._move_to_home()

    def _try_update_actor(self) -> None:
        """Pick up the latest actor weights from disk if they're newer than what we have."""
        if not self._latest_weights_path or not os.path.exists(self._latest_weights_path):
            return
        try:
            mtime = os.path.getmtime(self._latest_weights_path)
            if mtime <= self._last_weights_mtime:
                return  # nothing new
            # weights_only=False: we wrote this file ourselves and trust the pickle.
            payload = torch.load(self._latest_weights_path,
                                 map_location=self.device, weights_only=False)
            cfg = payload["actor_config"]
            if self.actor is None:
                self.actor = DistilledActor(
                    obs_dim=cfg["obs_dim"], action_dim=cfg["action_dim"],
                    cond_steps=cfg.get("cond_steps", 1),
                    horizon_steps=cfg["horizon_steps"],
                    hidden_dims=cfg.get("hidden_dims", [1024, 1024, 1024]),
                    activation_type="GELU", use_layernorm=True,
                ).to(self.device)
                self.actor.eval()
            self.actor.load_state_dict(payload["actor_state_dict"])
            self._actor_step = payload.get("training_step", -1)
            self._last_weights_mtime = mtime
            log.info("★ Actor weights updated (step=%d, %s) — takes effect next episode",
                     self._actor_step, os.path.basename(self._latest_weights_path))
        except Exception as e:
            log.warning("Failed to load actor weights from %s: %s",
                        self._latest_weights_path, e)

    def _sigint(self, *_) -> None:
        now = time.monotonic()
        if now - self._last_sigint_t < 0.5:
            return  # debounce: one physical keypress can fire the handler twice
        self._last_sigint_t = now

        if self._in_episode:
            log.info("Ctrl-C: aborting episode. Use Ctrl-\\ to force-quit.")
            self._abort_episode["flag"] = True
        else:
            raise KeyboardInterrupt

    def _sigterm(self, *_) -> None:
        log.info("SIGTERM received; shutting down safely.")
        raise SystemExit(0)

    def _sigquit(self, *_) -> None:
        log.info("Ctrl-\\ received: hard-killing immediately.")
        os._exit(131)
