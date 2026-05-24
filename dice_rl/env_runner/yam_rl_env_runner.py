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
from typing import Optional

import cv2
import numpy as np
import torch

from dice_rl.communication.actor_node import Actor
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
    rgb256 = _short_side_crop(rgb, 256)
    t = torch.from_numpy(rgb256).permute(2,0,1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, (224,224), mode="bilinear", align_corners=False)
    return t.squeeze(0).clamp(0,255).numpy() / 255.0


# ---- non-blocking camera (same as eval_dp_yam.py) ----

class _SyncCamera:
    def __init__(self, serial, w, h, fps, name):
        self.serial, self.width, self.height, self.fps, self.name = serial, w, h, fps, name
        self._pipe, self._latest, self._t = None, None, 0.0
    def start(self):
        import pyrealsense2 as rs
        p = rs.pipeline(); c = rs.config()
        c.enable_device(self.serial)
        c.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        p.start(c)
        try:
            f = p.wait_for_frames(timeout_ms=2000)
            self._latest = np.asanyarray(f.get_color_frame().get_data())
            self._t = time.monotonic()
        except Exception: pass
        self._pipe = p
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
        # YAM hardware
        can_channel: str = "can_follower_l",
        gripper_type: str = "linear_4310",
        home_joint_pos: list = None,
        home_gripper_pos: float = 1.0,
        # Control
        control_hz: float = 30.0,
        max_episode_steps: int = 200,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        action_dim: int = 7,
        # Actor (residual RL)
        actor_hidden_dims: list = None,
        # Data & ZMQ
        online_data_dir: str = "/tmp/yam_rl_rollouts",
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
        self.home_joint_pos = np.array(home_joint_pos or [-0.01,0.833,0.903,-0.598,-0.028,-0.029],
                                       dtype=np.float32)
        self.home_gripper_pos = home_gripper_pos
        self.online_data_dir = online_data_dir
        os.makedirs(online_data_dir, exist_ok=True)

        # ---- normalisation ----
        n = np.load(norm_npz_path)
        from scripts.eval_flow_matching_yam import MinMaxNorm
        self.state_norm  = MinMaxNorm(n["obs_min"],    n["obs_max"])
        self.action_norm = MinMaxNorm(n["action_min"], n["action_max"])

        # ---- load frozen BC policy ----
        log.info("Loading BC policy from %s", pretrained_policy_ckpt)
        self.bc_policy, _, _ = load_policy(pretrained_policy_ckpt, device)
        self.bc_policy.eval()
        for p in self.bc_policy.parameters():
            p.requires_grad = False
        obs_feature_dim = self.bc_policy.obs_feature_dim
        log.info("BC policy obs_feature_dim=%d", obs_feature_dim)

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

        # ---- hardware ----
        from i2rt.robots.get_robot import get_yam_robot, GripperType
        self.robot = get_yam_robot(channel=can_channel,
                                   gripper_type=GripperType.from_string_name(gripper_type),
                                   zero_gravity_mode=True)
        self.base_cam  = _SyncCamera(base_cam_serial,  640, 480, 30, "base")
        self.wrist_cam = _SyncCamera(wrist_cam_serial, 640, 480, 30, "wrist")
        self.base_cam.start(); self.wrist_cam.start()
        time.sleep(1.0)
        log.info("Cameras streaming.")

        # ---- ZMQ ----
        self.actor_node = Actor(
            network_server_endpoint=network_server_endpoint,
            network_weight_topic=network_weight_topic,
            transitions_server_endpoint=transitions_server_endpoint,
            transitions_topic=transitions_topic,
            transitions_topic_expire_time_s=3600,
        )

        self._abort_episode  = {"flag": False}
        self._in_episode     = False
        self._last_sigint_t  = 0.0   # debounce: ignore rapid duplicate SIGINTs
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

    def _infer(self, obs_tensors) -> np.ndarray:
        """Run BC policy (+ optional residual actor) → (action_horizon, 7) raw i2rt."""
        nobs = {k: self.bc_policy.sparse_normalizer[k].normalize(v)
                for k, v in obs_tensors["sparse"].items()}
        features = self.bc_policy.obs_encoder(nobs)  # (1, feat_dim)

        noise = torch.randn(1, self.action_horizon, self.action_dim, device=self.device)
        bc_act_n = self.bc_policy.predict_action_from_features(
            sparse_nobs_encode=features,
            init_noise=noise,
            unnormalize=False,
        )["sparse"]  # (1, H, 7) normalized

        if self.actor is not None:
            delta = self.actor(features.unsqueeze(1), noise)  # (1, H, 7)
            final_n = (bc_act_n + delta).clamp(-1.0, 1.0)
        else:
            final_n = bc_act_n

        final_n = final_n[0].cpu().numpy()          # (H, 7) normalized
        final   = self.action_norm.denormalize(final_n)  # (H, 7) i2rt raw
        return final

    # ---- episode loop ----

    def run_episode(self) -> dict:
        """Execute one episode. Returns {images, states, actions, rewards, dones}."""
        # pre-fill history
        q0 = self._read_state()
        b, _ = self.base_cam.get(); w, _ = self.wrist_cam.get()
        b = _preprocess(b); w = _preprocess(w)
        img0 = np.concatenate([b, w], axis=0)  # (6, 224, 224)
        state_hist = deque([q0.copy()] * self.obs_horizon, maxlen=self.obs_horizon)
        img_hist   = deque([img0.copy()] * self.obs_horizon, maxlen=self.obs_horizon)

        images_rec, states_rec, actions_rec, rewards_rec = [], [], [], []
        self._abort_episode["flag"] = False
        self._in_episode = True

        for step in range(self.max_episode_steps):
            if self._abort_episode["flag"]:
                break

            obs_t = self._make_obs_tensors(img_hist, state_hist)
            t0    = time.monotonic()
            with torch.no_grad():
                actions = self._infer(obs_t)  # (H, 7) raw
            infer_ms = (time.monotonic() - t0) * 1000.0

            # execute chunk
            chunk_start = time.monotonic()
            for i, q_tgt in enumerate(actions[:self.action_horizon]):
                if self._abort_episode["flag"]: break
                now = time.monotonic()
                wait = chunk_start + i * self.period - now
                if wait > 0: time.sleep(wait)

                q_cmd = np.clip(q_tgt.astype(np.float64),
                                [-2.767,-0.15,-0.15,-1.72,-1.72,-2.24,0.],
                                [ 3.28,  3.80, 3.28, 1.72, 1.72, 2.24,1.5])
                self.robot.command_joint_pos(q_cmd)
                q_cur = self._read_state()

                # update obs history at 30 Hz
                br, _ = self.base_cam.get(); wr, _ = self.wrist_cam.get()
                if br is not None:
                    b_p = _preprocess(br); w_p = _preprocess(wr)
                    img_cur = np.concatenate([b_p, w_p], axis=0)
                    state_hist.append(q_cur.copy())
                    img_hist.append(img_cur.copy())

            # record (first action of chunk as label)
            images_rec.append(img_hist[-1].copy())
            states_rec.append(self.state_norm.normalize(state_hist[-1]))
            actions_rec.append(self.action_norm.normalize(actions[0]))

            log.info("[step %3d] infer=%.1fms  q=%s",
                     step, infer_ms, np.round(self._read_state(), 3).tolist())

        # ---- user labels success/failure ----
        self._in_episode = False
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
            "success":  reward_val > 0.5,
        }

    # ---- main loop ----

    def _move_to_home(self, ramp_seconds: float = 6.0) -> None:
        """Smooth ramp to home pose using i2rt move_joints (50-step interpolation)."""
        h = np.concatenate([self.home_joint_pos, [self.home_gripper_pos]])
        log.info("Moving to home pose (%.1fs ramp)…", ramp_seconds)
        self.robot.move_joints(h.astype(np.float64), time_interval_s=ramp_seconds)
        time.sleep(0.3)
        log.info("At home. q=%s", np.round(self._read_state(), 3).tolist())

    def run(self) -> None:
        log.info("Env runner ready.")

        # Auto-home before the very first episode.
        self._move_to_home()

        # Count already-saved episodes so numbering stays consistent across restarts.
        ep = len(glob.glob(os.path.join(self.online_data_dir, "episode_*.npz")))
        if ep > 0:
            log.info("Resuming: %d episodes already saved in %s", ep, self.online_data_dir)

        while True:
            # Check for updated actor weights from learner.
            self._try_update_actor()

            input(f"\n[Episode {ep+1}] Press Enter to start, or Ctrl-C to exit.")
            ep_data = self.run_episode()
            if ep_data.get("discard"):
                log.info("Episode discarded.")
                self._move_to_home()
                continue

            # Save episode to disk (survives learner/runner crashes).
            ep_path = os.path.join(self.online_data_dir, f"episode_{ep:04d}.npz")
            np.savez_compressed(ep_path,
                                images=ep_data["images"],
                                states=ep_data["states"],
                                actions=ep_data["actions"],
                                rewards=ep_data["rewards"],
                                dones=ep_data["dones"])

            # Send episode to learner.
            self.actor_node.send_transitions(pickle.dumps(ep_data))
            log.info("Episode %d saved+sent (success=%s)", ep+1, ep_data["success"])
            ep += 1

            # Auto-home at end of every episode (ready for the next one).
            self._move_to_home()

    def _try_update_actor(self) -> None:
        try:
            data, _ = self.actor_node.network_weight_client.pop_data(
                topic=self.actor_node.network_weight_topic, order="latest", n=1,
            )
            if not data: return
            payload = pickle.loads(data[0])
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
            log.info("Actor weights updated (step=%d)", payload.get("training_step", -1))
        except Exception as e:
            log.debug("Weight update check: %s", e)

    def _sigint(self, *_) -> None:
        now = time.monotonic()
        if now - self._last_sigint_t < 0.5:
            return  # debounce: one physical keypress can fire the handler twice
        self._last_sigint_t = now

        if self._in_episode:
            log.info("Ctrl-C: aborting episode. Use Ctrl-\\ to force-quit.")
            self._abort_episode["flag"] = True
        else:
            log.info("Ctrl-C: exiting.")
            os._exit(130)

    def _sigterm(self, *_) -> None:
        log.info("SIGTERM received: force-quitting.")
        os._exit(0)

    def _sigquit(self, *_) -> None:
        log.info("Ctrl-\\ received: hard-killing immediately.")
        os._exit(131)
