"""Replay buffer for DICE-RL finetuning of a joint-space diffusion policy on YAM.

Design notes
------------
The buffer holds (obs, action, reward, next_obs, done) tuples where:
  - obs / next_obs: a dict with keys {"rgb_0", "rgb_1", "joint_pos"}, each a
    numpy array with a *cond_steps* time dimension (obs history).
  - action: 7-D normalized joint target (same space as training data, [-1,1]).
  - reward: scalar float (user-provided success/failure signal).
  - done: bool.

Unlike the original DICE-RL HybridReplayBuffer (which reads DICE-RL-Robot zarr
episodes processed from Cartesian-space SE(3) data), this buffer reads:
  1. *Expert data* directly from the dice-rl npz file (states/actions/images
     already in the joint-space normalized format used for BC training).
  2. *Online episodes* saved by YAMRLEnvRunner as simple npz dicts.

The RLPD batch is composed of (expert_ratio × batch_size) expert transitions
plus ((1-expert_ratio) × batch_size) online transitions.
"""

from __future__ import annotations
import glob
import logging
import os
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


class YAMReplayBuffer:
    """Simple RLPD-compatible replay buffer for YAM joint-space policy."""

    def __init__(
        self,
        expert_npz_path: str,
        online_data_dir: str,
        obs_horizon: int = 2,
        action_dim: int = 7,
        action_horizon: int = 16,
        max_online_size: int = 50_000,
        device: str = "cuda",
        hire_shaper=None,
        use_sparse_for_online_success: bool = False,
    ) -> None:
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.device = torch.device(device)
        # Optional HiRE reward shaper: if provided and `is_ready()` is True,
        # episodes' sparse rewards get PBRS dense shaping applied at insertion.
        self.hire_shaper = hire_shaper
        # Switch: when True, batches sampled from online SUCCESS episodes use the
        # original sparse reward instead of the HiRE-shaped one. Online failure
        # episodes always use the shaped reward; offline expert always uses
        # sparse (= 1.0 per `_sample_expert`).
        self.use_sparse_for_online_success = bool(use_sparse_for_online_success)

        # ---- expert buffer (preloaded from BC training npz) ----
        log.info("Loading expert data from %s", expert_npz_path)
        d = np.load(expert_npz_path)
        self._expert_states = d["states"].astype(np.float32)   # (T, 7) [-1,1]
        self._expert_actions = d["actions"].astype(np.float32) # (T, 7) [-1,1]
        self._expert_images = d["images"]                       # (T, 6, H, W) uint8
        self._expert_traj_lengths = d["traj_lengths"].astype(int)
        ep_starts = np.concatenate([[0], np.cumsum(self._expert_traj_lengths[:-1])])

        # Build valid (t, ep_start) index pairs for expert buffer.
        self._expert_indices = []
        for s, length in zip(ep_starts, self._expert_traj_lengths):
            for t in range(s, s + int(length) - 1):  # -1 so next_t exists
                self._expert_indices.append((t, int(s)))
        self._expert_indices = np.array(self._expert_indices, dtype=np.int64)
        log.info("Expert buffer: %d transitions from %d episodes",
                 len(self._expert_indices), len(self._expert_traj_lengths))

        # ---- online buffer (ring buffer for rollout data) ----
        self.online_data_dir = online_data_dir
        os.makedirs(online_data_dir, exist_ok=True)
        self._max_online = max_online_size
        self._online: deque = deque(maxlen=max_online_size)
        self._num_online_episodes = 0
        self._load_existing_episodes()

    # ------------------------------------------------------------------
    # Episode insertion
    # ------------------------------------------------------------------

    def _load_existing_episodes(self) -> None:
        self.loaded_paths: list = []
        paths = sorted(glob.glob(os.path.join(self.online_data_dir, "episode_*.npz")))
        if not paths:
            return
        log.info("Loading %d saved episodes from disk (please wait)…", len(paths))
        for i, p in enumerate(paths):
            d = np.load(p)
            self.add_episode({k: d[k] for k in d.files})
            self.loaded_paths.append(p)
            if (i + 1) % 5 == 0 or (i + 1) == len(paths):
                log.info("  … %d/%d episodes loaded", i + 1, len(paths))
        log.info("Online buffer restored: %d transitions from %d episodes",
                 len(self._online), self._num_online_episodes)

    def add_episode(self, episode: dict) -> None:
        """Add one online rollout episode to the buffer.

        episode dict keys:
          images  : (T, 6, H, W) uint8
          states  : (T, 7) float32 normalized
          actions : (T, 7) float32 normalized
          rewards : (T,) float32
          dones   : (T,) bool
        """
        S = episode["states"]
        A = episode["actions"]
        R_sparse = np.asarray(episode["rewards"], dtype=np.float32)
        D = episode["dones"]
        I = episode["images"]
        T = len(S)
        ep_start = 0

        # Episode-level success flag (used at sample time for the
        # `use_sparse_for_online_success` switch).
        success = bool(R_sparse[-1] > 0.5) if len(R_sparse) > 0 else False

        # HiRE PBRS shaping: r_t_shaped = r_t_sparse + γ·Φ(s_{t+1}) − Φ(s_t)
        # We store BOTH sparse and shaped per transition so sample time can
        # pick between them via the switch.
        if self.hire_shaper is not None and self.hire_shaper.is_ready():
            R_shaped = self.hire_shaper.shape_rewards(R_sparse, I)
        else:
            R_shaped = R_sparse.copy()

        for t in range(T - 1):
            obs      = self._make_obs(I, S, t,   ep_start)
            next_obs = self._make_obs(I, S, t+1, ep_start)
            a = A[t]
            if a.ndim == 1:  # single action (7,) → tile to (H, 7)
                a = np.tile(a, (self.action_horizon, 1))
            # Tuple format: (obs, action, r_shaped, r_sparse, is_success, next_obs, done)
            self._online.append((obs, a,
                                 float(R_shaped[t]), float(R_sparse[t]),
                                 success, next_obs, D[t]))

        self._num_online_episodes += 1
        log.debug("Online buffer: %d transitions from %d episodes",
                  len(self._online), self._num_online_episodes)

    def _make_obs(self, images, states, t, ep_start):
        """Build the obs history dict at time t (padded at episode start)."""
        frames, jnts = [], []
        uint8 = (images.dtype == np.uint8)
        for k in range(self.obs_horizon - 1, -1, -1):
            idx = max(t - k, ep_start)
            raw = images[idx].astype(np.float32)
            # Expert images are uint8 [0,255]; online images are float32 [0,1] already.
            frames.append(raw / 255.0 if uint8 else raw)  # (6, H, W) [0,1]
            jnts.append(states[idx])
        return {
            "rgb_0":     np.stack([f[:3] for f in frames]),   # (To, 3, H, W)
            "rgb_1":     np.stack([f[3:] for f in frames]),   # (To, 3, H, W)
            "joint_pos": np.stack(jnts),                      # (To, 7)
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        expert_ratio: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> dict:
        """Sample a mixed expert + online batch."""
        dev = device or self.device
        n_expert = int(batch_size * expert_ratio)
        n_online = batch_size - n_expert

        batches = []
        if n_expert > 0 and len(self._expert_indices) > 0:
            batches.append(self._sample_expert(n_expert, dev))
        if n_online > 0 and len(self._online) > 0:
            batches.append(self._sample_online(n_online, dev))

        if not batches:
            return {}
        if len(batches) == 1:
            return batches[0]
        return _cat_batches(batches)

    def _sample_expert(self, n: int, dev: torch.device) -> dict:
        idxs = np.random.randint(0, len(self._expert_indices), n)
        rows = self._expert_indices[idxs]
        obs_list, next_obs_list, acts, rews, dones = [], [], [], [], []
        for t, ep_start in rows:
            o  = self._make_obs(self._expert_images, self._expert_states, t,   ep_start)
            no = self._make_obs(self._expert_images, self._expert_states, t+1, ep_start)
            obs_list.append(o); next_obs_list.append(no)
            acts.append(np.tile(self._expert_actions[t], (self.action_horizon, 1)))
            rews.append(1.0)   # expert demonstrations treated as success
            dones.append(False)
        return _pack(obs_list, acts, rews, next_obs_list, dones, dev, is_expert=True)

    def _sample_online(self, n: int, dev: torch.device) -> dict:
        online_list = list(self._online)
        idxs = np.random.randint(0, len(online_list), n)
        obs_list, next_obs_list, acts, rews, dones = [], [], [], [], []
        for i in idxs:
            o, a, r_shaped, r_sparse, is_success, no, d = online_list[i]
            # Online-success switch: when ON, success transitions revert to the
            # sparse reward (matches offline expert's sparse-style supervision).
            # Online failures always use the HiRE-shaped reward.
            if self.use_sparse_for_online_success and is_success:
                r = r_sparse
            else:
                r = r_shaped
            obs_list.append(o); next_obs_list.append(no)
            acts.append(a); rews.append(r); dones.append(d)
        return _pack(obs_list, acts, rews, next_obs_list, dones, dev, is_expert=False)

    @property
    def num_online_transitions(self) -> int:
        return len(self._online)

    @property
    def num_expert_transitions(self) -> int:
        return len(self._expert_indices)


# ---- helpers ----

def _pack(obs_list, acts, rews, next_obs_list, dones, dev, is_expert: bool = False) -> dict:
    def _t(x): return torch.from_numpy(np.stack(x)).to(dev, non_blocking=True)
    def _obs(lst):
        return {
            "rgb_0":     _t([o["rgb_0"] for o in lst]),
            "rgb_1":     _t([o["rgb_1"] for o in lst]),
            "joint_pos": _t([o["joint_pos"] for o in lst]).float(),
        }
    n = len(obs_list)
    return {
        "obs":      _obs(obs_list),
        "action":   _t(acts).float(),
        "reward":   _t(rews).float().unsqueeze(-1),
        "next_obs": _obs(next_obs_list),
        "done":     torch.tensor(dones, dtype=torch.float32, device=dev).unsqueeze(-1),
        # 1.0 for expert (BC demos), 0.0 for online (env rollouts).
        # Used by learner's disable_q_loss_for_expert_data flag.
        "is_expert": torch.full((n, 1), float(is_expert), dtype=torch.float32, device=dev),
    }


def _cat_batches(batches: list) -> dict:
    result = {}
    for k in batches[0]:
        v0 = batches[0][k]
        if isinstance(v0, dict):
            result[k] = {kk: torch.cat([b[k][kk] for b in batches]) for kk in v0}
        else:
            result[k] = torch.cat([b[k] for b in batches])
    return result
