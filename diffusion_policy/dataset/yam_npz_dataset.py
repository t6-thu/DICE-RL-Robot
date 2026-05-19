"""Dataset for YAM picknplace tasks, reading from dice-rl-style .npz files.

The npz file contains:
  states       (T, 7)        float32, normalized to [-1, 1]  joint_0..5 + gripper
  actions      (T, 7)        float32, normalized to [-1, 1]  same space
  images       (T, 6, 128, 128)  uint8   base_cam (3ch) + wrist_cam (3ch)
  traj_lengths (E,)          int64    per-episode lengths

Batch format returned (matches train_diffusion_unet_image_workspace.py):
  obs.sparse:
    rgb_0        (B, T_obs_img,  3, H, W)  base camera
    rgb_1        (B, T_obs_img,  3, H, W)  wrist camera
    joint_pos    (B, T_obs_low, 7)         joint positions
  action.sparse  (B, T_action, 7)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_range_normalizer_from_stat,
    get_image_identity_normalizer,
)


class YAMNpzDataset(BaseImageDataset):
    """Load pre-processed YAM data from a dice-rl npz file and serve it as
    action-chunked observation-action pairs for the DICE-RL-Robot diffusion
    policy training workspace.
    """

    def __init__(
        self,
        dataset_path: str,
        obs_horizon: int = 1,
        action_horizon: int = 16,
        image_size: int = 224,
        val_ratio: float = 0.02,
        seed: int = 42,
    ) -> None:
        super().__init__()
        data = np.load(dataset_path, allow_pickle=False)
        self.states = data["states"].astype(np.float32)       # (T, 7) in [-1, 1]
        self.actions = data["actions"].astype(np.float32)     # (T, 7) in [-1, 1]
        self.images = data["images"]                           # (T, 6, 128, 128) uint8
        self.traj_lengths = data["traj_lengths"].astype(int)  # (E,)

        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size

        # Build per-episode start/end indices.
        ep_starts = np.concatenate([[0], np.cumsum(self.traj_lengths[:-1])])
        ep_ends = ep_starts + self.traj_lengths

        # Build valid query indices: each query must have enough room for the
        # full action horizon within the same episode.
        indices = []
        for s, e in zip(ep_starts, ep_ends):
            for t in range(s, e - action_horizon + 1):
                indices.append(t)
        indices = np.array(indices, dtype=np.int64)

        # Train / val split (last val_ratio of episodes).
        rng = np.random.default_rng(seed)
        n_val_ep = max(1, int(len(self.traj_lengths) * val_ratio))
        val_ep_set = set(rng.choice(len(self.traj_lengths), n_val_ep, replace=False))
        val_starts = set(int(ep_starts[i]) for i in val_ep_set)

        def _episode_start(t):
            idx = np.searchsorted(ep_starts, t, side="right") - 1
            return int(ep_starts[idx])

        self._train_indices = np.array(
            [t for t in indices if _episode_start(t) not in val_starts], dtype=np.int64
        )
        self._val_indices = np.array(
            [t for t in indices if _episode_start(t) in val_starts], dtype=np.int64
        )
        self._ep_starts = ep_starts
        self._is_val = False
        self.action_type = "joint_pos"  # workspace inspects this attr

        # Map each global step → episode start, for padding at episode start.
        self._ep_start_for = np.empty(len(self.states), dtype=np.int64)
        for s, e in zip(ep_starts, ep_ends):
            self._ep_start_for[s:e] = s

    # -------------------------------------------------------------------
    # Split support

    def get_validation_dataset(self) -> "YAMNpzDataset":
        copy = YAMNpzDataset.__new__(YAMNpzDataset)
        copy.__dict__.update(self.__dict__)
        copy._is_val = True
        return copy

    # -------------------------------------------------------------------
    # Normalizer

    def get_normalizer(self, **kwargs):
        """Return (sparse_normalizer, None)."""
        sparse_norm = LinearNormalizer()
        state_stat = array_to_stats(self.states)
        action_stat = array_to_stats(self.actions)
        sparse_norm["joint_pos"] = get_range_normalizer_from_stat(state_stat)
        sparse_norm["action"] = get_range_normalizer_from_stat(action_stat)
        sparse_norm["rgb_0"] = get_image_identity_normalizer()
        sparse_norm["rgb_1"] = get_image_identity_normalizer()
        return sparse_norm, None   # (sparse, dense)

    # -------------------------------------------------------------------
    # Dataset protocol

    def __len__(self) -> int:
        idx = self._val_indices if self._is_val else self._train_indices
        return len(idx)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        idx = self._val_indices if self._is_val else self._train_indices
        t = int(idx[i])
        ep_s = int(self._ep_start_for[t])

        # Obs history: pad at episode start by repeating the first frame.
        obs_frames = []
        for k in range(self.obs_horizon - 1, -1, -1):   # oldest → newest
            tobs = max(t - k, ep_s)
            obs_frames.append(tobs)

        state_seq = self.states[obs_frames]   # (T_obs, 7)
        img_seq = self.images[obs_frames]     # (T_obs, 6, 128, 128)

        # Split two cameras: channels 0-2 = base, 3-5 = wrist.
        rgb0 = img_seq[:, :3].astype(np.float32) / 255.0   # (T_obs, 3, 128, 128)
        rgb1 = img_seq[:, 3:].astype(np.float32) / 255.0

        # Action chunk.
        action_seq = self.actions[t : t + self.action_horizon]  # (T_act, 7)

        obs_sparse: Dict[str, torch.Tensor] = {
            "rgb_0": torch.from_numpy(rgb0),      # (T_obs, 3, 128, 128)
            "rgb_1": torch.from_numpy(rgb1),
            "joint_pos": torch.from_numpy(state_seq),  # (T_obs, 7)
        }

        return {
            "obs": {"sparse": obs_sparse},
            "action": {"sparse": torch.from_numpy(action_seq)},
        }
