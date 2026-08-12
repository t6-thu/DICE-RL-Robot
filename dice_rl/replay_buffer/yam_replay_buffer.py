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
import re
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

_EPISODE_FILE_RE = re.compile(r"^episode_\d+\.npz$")


def list_episode_npz_paths(directory: str) -> list:
    """Return saved rollout episode files, excluding sidecar cache files."""
    return sorted(
        p for p in glob.glob(os.path.join(directory, "episode_*.npz"))
        if _EPISODE_FILE_RE.match(os.path.basename(p))
    )


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
        max_online_episodes: Optional[int] = None,
        device: str = "cuda",
        hire_shaper=None,
        robometer_shaper=None,
        defer_robometer_reward: bool = True,
        use_sparse_for_online_success: bool = False,
        expert_curation_path: Optional[str] = None,
    ) -> None:
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.device = torch.device(device)
        # Optional dense reward shapers (HiRE or Robometer; mutually exclusive).
        self.hire_shaper = hire_shaper
        self.robometer_shaper = robometer_shaper
        self.defer_robometer_reward = bool(defer_robometer_reward)
        self._robometer_pending: list = []
        if hire_shaper is not None and robometer_shaper is not None:
            raise ValueError("hire_shaper and robometer_shaper are mutually exclusive")
        # Switch: when True, batches sampled from online SUCCESS episodes use the
        # original sparse reward instead of the HiRE-shaped one. Online failure
        # episodes always use the shaped reward. Offline expert demos use a
        # sparse +1 only on the terminal transition (matches online success).
        self.use_sparse_for_online_success = bool(use_sparse_for_online_success)

        # ---- expert buffer (preloaded from BC training npz) ----
        log.info("Loading expert data from %s", expert_npz_path)
        # The recovered Hanoi dataset has a ``train_images.npy`` sidecar.  Map
        # it instead of eagerly decompressing the image member of the npz: it
        # has identical samples but avoids a multi-GiB RAM spike when the
        # learner starts alongside the Robometer service.
        image_sidecar = os.path.splitext(expert_npz_path)[0] + "_images.npy"
        with np.load(expert_npz_path) as d:
            self._expert_states = d["states"].astype(np.float32)   # (T, 7) [-1,1]
            self._expert_actions = d["actions"].astype(np.float32) # (T, 7) [-1,1]
            self._expert_traj_lengths = d["traj_lengths"].astype(int)
            if os.path.isfile(image_sidecar):
                self._expert_images = np.load(image_sidecar, mmap_mode="r")
                if (
                    self._expert_images.ndim != 4
                    or self._expert_images.shape[0] != len(self._expert_states)
                    or self._expert_images.shape[1] != 6
                ):
                    raise ValueError(
                        f"Invalid expert image sidecar {image_sidecar}: "
                        f"expected (T, 6, H, W) with T={len(self._expert_states)}, "
                        f"got {self._expert_images.shape}"
                    )
                log.info("Expert images memory-mapped from %s", image_sidecar)
            else:
                self._expert_images = d["images"]  # (T, 6, H, W) uint8
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
        # `ep_end_t` is the last valid t (= s + L - action_horizon - 1): its
        # action chunk ends at the final frame and its next observation at
        # t+H remains inside this episode.  This is intentionally the same
        # [0, T-H) range used for online rollouts.
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
        self._max_online = max_online_size
        self._max_online_episodes = (
            int(max_online_episodes)
            if max_online_episodes is not None and int(max_online_episodes) > 0
            else None
        )
        self._total_disk_episodes = 0
        # Online data is stored episode-wise, not transition-wise. The old
        # implementation expanded every transition into copied obs/next_obs
        # image windows at startup; 50 saved rollouts could inflate into tens
        # of GB. Keeping each episode image tensor once preserves the sampling
        # distribution while making learner restarts cheap.
        self._online_episodes: list = []
        self._online_indices: deque = deque(maxlen=max_online_size)
        self._num_online_episodes = 0
        self._load_existing_episodes()

    # ------------------------------------------------------------------
    # Episode insertion
    # ------------------------------------------------------------------

    def _load_existing_episodes(self) -> None:
        self.loaded_paths: list = []
        paths = list_episode_npz_paths(self.online_data_dir)
        self._total_disk_episodes = len(paths)
        if not paths:
            return
        if self._max_online_episodes is not None and len(paths) > self._max_online_episodes:
            skipped = paths[:-self._max_online_episodes]
            paths_to_load = paths[-self._max_online_episodes:]
            # Mark older files as seen so the learner does not immediately
            # reload them through the disk polling path. They are represented
            # by the resumed checkpoint; the in-memory buffer only needs the
            # recent online data used for the next staged training round.
            self.loaded_paths.extend(skipped)
            log.info(
                "Loading last %d/%d saved episodes from disk (skipping older %d to cap memory)…",
                len(paths_to_load), len(paths), len(skipped))
        else:
            paths_to_load = paths
            log.info("Loading %d saved episodes from disk (please wait)…", len(paths_to_load))
        for i, p in enumerate(paths_to_load):
            d = np.load(p)
            ep = {k: d[k] for k in d.files}
            ep["__path__"] = p
            self.add_episode(ep)
            self.loaded_paths.append(p)
            if (i + 1) % 5 == 0 or (i + 1) == len(paths_to_load):
                log.info("  … %d/%d episodes loaded", i + 1, len(paths_to_load))
        log.info("Online buffer restored: %d transitions from %d episodes",
                 len(self._online_indices), self._num_online_episodes)

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
        S = np.asarray(episode["states"], dtype=np.float32)
        A = np.asarray(episode["actions"], dtype=np.float32)
        R_sparse = np.asarray(episode["rewards"], dtype=np.float32)
        D = np.asarray(episode["dones"], dtype=bool)
        I = np.asarray(episode["images"])
        ep_path = episode.get("__path__")
        T = len(S)
        H = self.action_horizon

        if T <= H:
            log.debug("Episode too short (%d frames) for even one chunk, skipping", T)
            self._num_online_episodes += 1
            return

        success = bool(R_sparse[-1] > 0.5) if len(R_sparse) > 0 else False

        # HiRE PBRS shaping with H-step lookahead:
        #   r̃_t = R_sparse[t+H] + γ·Φ(s_{t+H}) − Φ(s_t)
        # Terminal boundary: Φ(s_{T-1}) = 0 (last frame of episode).
        defer_robometer = (
            self.robometer_shaper is not None
            and self.robometer_shaper.is_ready()
            and self.defer_robometer_reward
        )
        cached = None
        if defer_robometer:
            cached = self._load_cached_robometer_rewards(ep_path, H)

        if cached is not None:
            R_shaped_tr = cached
        elif (
            self.robometer_shaper is not None
            and self.robometer_shaper.is_ready()
            and not defer_robometer
        ):
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
                f"episode reward length mismatch: got {len(R_shaped_tr)} "
                f"expected {T - H}"
            )

        ep_idx = len(self._online_episodes)
        self._online_episodes.append({
            "images": I,
            "states": S,
            "actions": A,
            "rewards_sparse": R_sparse,
            "rewards_shaped": np.asarray(R_shaped_tr, dtype=np.float32),
            "dones": D,
            "success": success,
            "path": ep_path,
            "horizon": H,
        })
        for t in range(T - H):
            self._online_indices.append((ep_idx, t))

        if defer_robometer and cached is None and ep_path:
            self._robometer_pending.append({
                "ep_idx": ep_idx,
                "path": ep_path,
                "horizon": H,
            })

        self._num_online_episodes += 1
        log.debug("Online buffer: %d transitions from %d episodes",
                  len(self._online_indices), self._num_online_episodes)

    @property
    def num_pending_robometer_episodes(self) -> int:
        return len(self._robometer_pending)

    @staticmethod
    def robometer_reward_cache_path(episode_path: Optional[str]) -> Optional[str]:
        if not episode_path:
            return None
        root, ext = os.path.splitext(str(episode_path))
        if ext != ".npz":
            return str(episode_path) + ".robometer_rewards.npz"
        return root + ".robometer_rewards.npz"

    def _robometer_cache_key(self, horizon: int) -> str:
        if self.robometer_shaper is None:
            return ""
        if hasattr(self.robometer_shaper, "cache_key"):
            return self.robometer_shaper.cache_key(horizon)
        return f"horizon={int(horizon)}"

    def _load_cached_robometer_rewards(
        self,
        episode_path: Optional[str],
        horizon: int,
    ) -> Optional[np.ndarray]:
        cache_path = self.robometer_reward_cache_path(episode_path)
        if not cache_path or not os.path.exists(cache_path):
            return None
        try:
            d = np.load(cache_path, allow_pickle=False)
            rewards = np.asarray(d["rewards"], dtype=np.float32)
            cached_horizon = int(np.asarray(d["horizon"]).item())
            cache_key = str(np.asarray(d["cache_key"]).item())
        except Exception as e:
            log.warning("Robometer: ignoring unreadable cache %s (%s)", cache_path, e)
            return None
        if cached_horizon != int(horizon):
            log.info(
                "Robometer: ignoring cache %s due to horizon mismatch %d != %d",
                cache_path, cached_horizon, int(horizon))
            return None
        expected_key = self._robometer_cache_key(horizon)
        if cache_key != expected_key:
            log.info("Robometer: ignoring stale cache %s (config changed)", cache_path)
            return None
        return rewards

    def _save_cached_robometer_rewards(
        self,
        episode_path: Optional[str],
        horizon: int,
        rewards: np.ndarray,
    ) -> None:
        cache_path = self.robometer_reward_cache_path(episode_path)
        if not cache_path:
            return
        tmp = cache_path + ".tmp"
        rewards = np.asarray(rewards, dtype=np.float32)
        np.savez_compressed(
            tmp,
            rewards=rewards,
            horizon=np.array(int(horizon), dtype=np.int64),
            cache_key=np.array(self._robometer_cache_key(horizon)),
        )
        # np.savez appends ".npz" if the filename does not already end with it.
        actual_tmp = tmp if os.path.exists(tmp) else tmp + ".npz"
        os.replace(actual_tmp, cache_path)

    def shape_pending_robometer_rewards(self) -> int:
        """Run deferred Robometer scoring and patch online transition rewards.

        Episode insertion stays lightweight during robot rollout. This method is
        called by the learner immediately before a training round, when it is
        acceptable for the workstation to run the Robometer server.
        """
        if self.robometer_shaper is None or not self.robometer_shaper.is_ready():
            return 0
        pending = self._robometer_pending
        if not pending:
            return 0

        shaped = 0
        log.info("Robometer: shaping %d pending online episodes before training", len(pending))
        for item in pending:
            ep_idx = int(item["ep_idx"])
            horizon = int(item["horizon"])
            if ep_idx < 0 or ep_idx >= len(self._online_episodes):
                log.warning("Robometer: skipped stale pending ep_idx=%d", ep_idx)
                continue
            ep = self._online_episodes[ep_idx]
            rewards = self._load_cached_robometer_rewards(ep.get("path"), horizon)
            if rewards is None:
                rewards = self.robometer_shaper.shape_rewards(
                    ep["images"],
                    horizon=horizon,
                )
                self._save_cached_robometer_rewards(ep.get("path"), horizon, rewards)
            if len(rewards) != len(ep["rewards_shaped"]):
                log.warning(
                    "Robometer: skipped %s due to reward length mismatch %d != %d",
                    ep.get("path"), len(rewards), len(ep["rewards_shaped"]))
                continue
            ep["rewards_shaped"] = np.asarray(rewards, dtype=np.float32)
            shaped += 1
        self._robometer_pending = []
        log.info("Robometer: shaped %d/%d pending online episodes", shaped, len(pending))
        return shaped

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
        if n_online > 0 and len(self._online_indices) > 0:
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
            # ep_end_t = s + L - H - 1, so at the terminal t, next_obs lands
            # on the episode's last frame exactly (never the next episode).
            is_terminal = (int(t) == int(ep_end_t))
            rews.append(1.0 if is_terminal else 0.0)
            dones.append(is_terminal)
        return _pack(obs_list, acts, rews, next_obs_list, dones, dev, is_expert=True)

    def _sample_online(self, n: int, dev: torch.device) -> dict:
        online_indices = list(self._online_indices)
        idxs = np.random.randint(0, len(online_indices), n)
        obs_list, next_obs_list, acts, rews, dones = [], [], [], [], []
        for i in idxs:
            ep_idx, t = online_indices[i]
            ep = self._online_episodes[int(ep_idx)]
            H = int(ep["horizon"])
            o = self._make_obs(ep["images"], ep["states"], int(t), 0)
            no = self._make_obs(ep["images"], ep["states"], int(t) + H, 0)
            a = ep["actions"][int(t) : int(t) + H]
            r_shaped = float(ep["rewards_shaped"][int(t)])
            r_sparse = float(ep["rewards_sparse"][int(t) + H])
            is_success = bool(ep["success"])
            d = bool(ep["dones"][int(t) + H])
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
        return len(self._online_indices)

    @property
    def num_expert_transitions(self) -> int:
        return len(self._expert_indices)

    @property
    def total_disk_episodes(self) -> int:
        return int(self._total_disk_episodes)


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
