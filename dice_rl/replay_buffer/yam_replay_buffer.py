"""Replay buffer for DICE-RL finetuning of a joint-space diffusion policy on YAM.

Design notes
------------
Expert transitions are represented by indices into the expert arrays. Online
transitions are represented by compact ``(episode_id, t)`` references; each
episode's image/state/action arrays are retained exactly once. Observation
histories and action chunks are materialised only for the sampled mini-batch.
This is important for 224x224 real-robot images: eagerly storing obs and
next_obs for every transition costs roughly 4.6 MiB per transition.

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

    @staticmethod
    def _load_curation_include_ids(path: Optional[str], n_eps: int) -> Optional[list]:
        """Read curation JSON and return list of episode indices to include.

        Returns None if `path` is falsy / missing / contains empty `include`.
        Out-of-range indices are dropped. Output is sorted unique.
        """
        if not path or not os.path.isfile(path):
            return None
        try:
            import json
            with open(path) as f:
                cur = json.load(f)
        except Exception as e:
            log.warning("Replay buffer: failed to read curation %s (%s)", path, e)
            return None
        raw = sorted(set(int(x) for x in cur.get("include", [])))
        valid = [ep for ep in raw if 0 <= ep < n_eps]
        if not valid:
            log.warning("Replay buffer: curation %s has empty/invalid `include` list", path)
            return None
        return valid

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
        robometer_shaper=None,
        use_sparse_for_online_success: bool = False,
        expert_curation_path: Optional[str] = None,
        expected_policy_camera_order: Optional[str] = None,
    ) -> None:
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.device = torch.device(device)
        # Optional dense reward shapers (HiRE or Robometer; mutually exclusive).
        self.hire_shaper = hire_shaper
        self.robometer_shaper = robometer_shaper
        if hire_shaper is not None and robometer_shaper is not None:
            raise ValueError("hire_shaper and robometer_shaper are mutually exclusive")
        # Switch: when True, batches sampled from online SUCCESS episodes use the
        # original sparse reward instead of the HiRE-shaped one. Online failure
        # episodes always use the shaped reward. Offline expert demos use a
        # sparse +1 only on the terminal transition (matches online success).
        self.use_sparse_for_online_success = bool(use_sparse_for_online_success)
        self.expected_policy_camera_order = expected_policy_camera_order

        # ---- expert buffer (preloaded from BC training npz) ----
        log.info("Loading expert data from %s", expert_npz_path)
        d = np.load(expert_npz_path)
        self._expert_states = d["states"].astype(np.float32)   # (T, 7) [-1,1]
        self._expert_actions = d["actions"].astype(np.float32) # (T, 7) [-1,1]
        self._expert_images = d["images"]                       # (T, 6, H, W) uint8
        self._expert_traj_lengths = d["traj_lengths"].astype(int)
        ep_starts = np.concatenate([[0], np.cumsum(self._expert_traj_lengths[:-1])])
        n_eps = int(len(self._expert_traj_lengths))

        # Optional curation: filter which expert episodes are used for RL
        # training (via the same JSON sidecar consumed by HiRE).  Without
        # curation, all `n_eps` trajectories are included.
        include_eps = self._load_curation_include_ids(expert_curation_path, n_eps)
        if include_eps is None:
            include_eps = list(range(n_eps))
        else:
            log.info("Replay buffer: curation %s → using %d/%d expert episodes for training",
                     os.path.basename(expert_curation_path), len(include_eps), n_eps)
        include_set = set(int(x) for x in include_eps)

        # Build valid (t, ep_start, ep_end_t) index triples for the expert
        # buffer.  Only include positions where a full action_horizon-step chunk
        # fits within the episode (mirrors BC training's valid index range).
        # `ep_end_t` is the last valid t (= s + L - action_horizon - 1).
        # This keeps next_obs at t+H inside the same episode and mirrors the
        # online-buffer loop `for t in range(T - H)`.
        self._expert_indices = []
        for ep, (s, length) in enumerate(zip(ep_starts, self._expert_traj_lengths)):
            if ep not in include_set:
                continue
            s = int(s); L = int(length)
            if L <= self.action_horizon:
                continue  # episode too short to form even one valid chunk
            ep_end_t = s + L - self.action_horizon - 1
            for t in range(s, s + L - self.action_horizon):
                self._expert_indices.append((t, s, ep_end_t))
        self._expert_indices = np.array(self._expert_indices, dtype=np.int64)
        log.info("Expert buffer: %d transitions from %d episodes (of %d total in npz)",
                 len(self._expert_indices), len(include_eps), n_eps)

        # ---- online buffer (ring buffer for rollout data) ----
        self.online_data_dir = online_data_dir
        os.makedirs(online_data_dir, exist_ok=True)
        self._max_online = int(max_online_size)
        if self._max_online <= 0:
            raise ValueError("max_online_size must be positive")
        self._online: deque = deque(maxlen=max_online_size)
        # Compact episode store. `_online` contains only (episode_id, t)
        # references; refcounts let the ring buffer release an episode once all
        # of its transitions have aged out.
        self._online_episodes: Dict[int, dict] = {}
        self._online_episode_refcounts: Dict[int, int] = {}
        self._next_online_episode_id = 0
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
            with np.load(p) as d:
                episode = {k: d[k] for k in d.files}
            self.add_episode(episode)
            self.loaded_paths.append(p)
            if (i + 1) % 5 == 0 or (i + 1) == len(paths):
                log.info("  … %d/%d episodes loaded", i + 1, len(paths))
        log.info(
            "Online buffer restored: %d transitions from %d episodes "
            "(compact episode arrays: %.2f GiB)",
            len(self._online), self._num_online_episodes,
            self.online_storage_bytes / (1024 ** 3),
        )

    @staticmethod
    def _compact_images(images: np.ndarray) -> np.ndarray:
        """Store online images once as contiguous uint8 in policy channel order."""
        images = np.asarray(images)
        if images.dtype == np.uint8:
            return np.ascontiguousarray(images)
        images_f = images.astype(np.float32, copy=False)
        if images_f.size:
            lo = float(images_f.min())
            hi = float(images_f.max())
            if lo < 0.0 or hi > 1.0:
                raise ValueError(
                    "float online images must be in [0, 1], "
                    f"got min={lo:.4f} max={hi:.4f}"
                )
        return np.rint(images_f * 255.0).clip(0, 255).astype(np.uint8)

    def _append_online_ref(self, episode_id: int, t: int) -> None:
        """Append one compact ref and release any ring-buffer eviction."""
        if len(self._online) == self._max_online:
            evicted_episode_id, _ = self._online[0]
            remaining = self._online_episode_refcounts[evicted_episode_id] - 1
            if remaining == 0 and evicted_episode_id != episode_id:
                del self._online_episode_refcounts[evicted_episode_id]
                del self._online_episodes[evicted_episode_id]
            else:
                self._online_episode_refcounts[evicted_episode_id] = remaining
        self._online.append((episode_id, int(t)))
        self._online_episode_refcounts[episode_id] = (
            self._online_episode_refcounts.get(episode_id, 0) + 1
        )

    def add_episode(self, episode: dict) -> None:
        """Add one online rollout episode to the buffer.

        episode dict keys:
          images  : (T, 6, H_img, W) float32 [0,1]
          states  : (T, 7) float32 normalized
          actions : (T, 7) float32 normalized  ← dense single-step actions
          rewards : (T,) float32
          dones   : (T,) bool

        Fine-stride chunk sampling (mirrors original DICE-RL's
        _process_complete_episode):
          For each inner step t in [0, T-action_horizon):
            obs      = obs history ending at t
            next_obs = obs history ending at t + action_horizon   ← one chunk ahead
            action   = A[t : t+action_horizon]                    ← real consecutive chunk
            reward   = R_sparse[t + action_horizon]               ← reward on arrival
            done     = D[t + action_horizon]
        """
        if self.expected_policy_camera_order:
            if "policy_camera_order" not in episode:
                raise ValueError(
                    "online episode is missing policy_camera_order metadata; "
                    "discard it or collect it again with the aligned env runner"
                )
            actual_order = str(np.asarray(episode["policy_camera_order"]).item())
            if actual_order != self.expected_policy_camera_order:
                raise ValueError(
                    "online episode camera order mismatch: "
                    f"expected {self.expected_policy_camera_order!r}, "
                    f"got {actual_order!r}"
                )

        S = np.ascontiguousarray(episode["states"], dtype=np.float32)
        A = np.ascontiguousarray(episode["actions"], dtype=np.float32)
        R_sparse = np.ascontiguousarray(episode["rewards"], dtype=np.float32)
        D = np.ascontiguousarray(episode["dones"], dtype=bool)
        I = self._compact_images(episode["images"])
        T = len(S)
        H = self.action_horizon

        if not (len(A) == len(R_sparse) == len(D) == len(I) == T):
            raise ValueError(
                "online episode arrays must have equal leading lengths: "
                f"states={T} actions={len(A)} rewards={len(R_sparse)} "
                f"dones={len(D)} images={len(I)}"
            )

        if T <= H:
            log.debug("Episode too short (%d frames) for even one chunk, skipping", T)
            self._num_online_episodes += 1
            return

        success = bool(R_sparse[-1] > 0.5) if len(R_sparse) > 0 else False

        # HiRE PBRS shaping with H-step lookahead:
        #   r̃_t = R_sparse[t+H] + γ·Φ(s_{t+H}) − Φ(s_t)
        # Terminal boundary: Φ(s_{T-1}) = 0 (last frame of episode).
        if self.robometer_shaper is not None and self.robometer_shaper.is_ready():
            # Robometer-only dense reward (no sparse terminal term), like HiRE-Dice
            # robometer RLPD online rollouts.
            R_shaped_tr = self.robometer_shaper.shape_rewards(I, horizon=H)
        elif self.hire_shaper is not None and self.hire_shaper.is_ready():
            R_shaped_tr = self.hire_shaper.shape_rewards(R_sparse, I, horizon=H)
        else:
            # Fallback: use sparse reward at t+H with no shaping.
            R_shaped_tr = np.array(
                [float(R_sparse[t + H]) for t in range(T - H)], dtype=np.float32
            )
        if len(R_shaped_tr) != T - H:
            raise ValueError(
                "reward shaper returned the wrong number of transitions: "
                f"expected {T - H}, got {len(R_shaped_tr)}"
            )

        episode_id = self._next_online_episode_id
        self._next_online_episode_id += 1
        self._online_episodes[episode_id] = {
            "images": I,
            "states": S,
            "actions": A,
            "rewards_sparse": R_sparse,
            "rewards_shaped": np.ascontiguousarray(R_shaped_tr, dtype=np.float32),
            "dones": D,
            "success": success,
        }
        self._online_episode_refcounts[episode_id] = 0
        for t in range(T - H):
            self._append_online_ref(episode_id, t)

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
            # Expert and compact online images are uint8 [0,255]. Legacy float
            # arrays remain supported for callers outside this replay buffer.
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
        H = self.action_horizon
        for t, ep_start, ep_end_t in rows:
            o  = self._make_obs(self._expert_images, self._expert_states, t,     ep_start)
            no = self._make_obs(self._expert_images, self._expert_states, t + H, ep_start)
            obs_list.append(o); next_obs_list.append(no)
            acts.append(self._expert_actions[t : t + H])
            # next_obs is one full action chunk ahead (t+H), matching the original
            # SequenceSampler which sets next_query_time = query_time + chunk_duration_ms.
            # ep_end_t = s + L - H - 1, so at the terminal t, next_obs lands on
            # the episode's last frame exactly without crossing trajectories.
            is_terminal = (int(t) == int(ep_end_t))
            rews.append(1.0 if is_terminal else 0.0)
            dones.append(is_terminal)
        return _pack(obs_list, acts, rews, next_obs_list, dones, dev, is_expert=True)

    def _sample_online(self, n: int, dev: torch.device) -> dict:
        online_list = list(self._online)
        idxs = np.random.randint(0, len(online_list), n)
        obs_list, next_obs_list, acts, rews, dones = [], [], [], [], []
        for i in idxs:
            episode_id, t = online_list[i]
            episode = self._online_episodes[episode_id]
            H = self.action_horizon
            o = self._make_obs(
                episode["images"], episode["states"], t, ep_start=0
            )
            no = self._make_obs(
                episode["images"], episode["states"], t + H, ep_start=0
            )
            a = episode["actions"][t:t + H]
            r_shaped = float(episode["rewards_shaped"][t])
            r_sparse = float(episode["rewards_sparse"][t + H])
            is_success = bool(episode["success"])
            d = bool(episode["dones"][t + H])
            # Online-success switch (HiRE only): revert success transitions to sparse.
            # Robometer mode always uses shaped rewards for all online transitions.
            if (
                self.robometer_shaper is None
                and self.use_sparse_for_online_success
                and is_success
            ):
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
    def online_storage_bytes(self) -> int:
        """Bytes held by unique compact online episode arrays (no double-counting)."""
        return sum(
            array.nbytes
            for episode in self._online_episodes.values()
            for array in episode.values()
            if isinstance(array, np.ndarray)
        )

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
