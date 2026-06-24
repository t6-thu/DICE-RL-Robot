"""Dataset adapter for YAM HDF5 demonstrations.

Expected HDF5 layout:
  rgb_0        (T,) variable-length uint8 JPEG bytes
  rgb_1        (T,) variable-length uint8 JPEG bytes
  trajectory   (T, 7) float joint/gripper positions
  segment_ids  (T,) int episode ids

It returns the same batch structure as ``YAMNpzDataset`` so the existing
diffusion policy pretraining workspace can be reused directly.
"""

from __future__ import annotations

from typing import Dict

import cv2
import h5py
import numpy as np
import torch

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class YAMHdf5Dataset(BaseImageDataset):
    def __init__(
        self,
        dataset_path: str,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        image_size: int = 224,
        val_ratio: float = 0.05,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.dataset_path = str(dataset_path)
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.image_size = int(image_size)
        self._h5 = None

        with h5py.File(self.dataset_path, "r") as f:
            self.trajectory = np.asarray(f["trajectory"], dtype=np.float32)
            segment_ids = np.asarray(f["segment_ids"], dtype=np.int64)
            self._num_frames = int(self.trajectory.shape[0])

        if self.trajectory.ndim != 2:
            raise ValueError(f"trajectory must be (T, D), got {self.trajectory.shape}")
        if self._num_frames != len(segment_ids):
            raise ValueError("trajectory and segment_ids have different lengths")

        # HDF5 stores joint/gripper trajectory only; use current trajectory as
        # state and future trajectory chunks as absolute joint-target actions.
        self.states = self.trajectory
        self.actions = self.trajectory

        changes = np.flatnonzero(segment_ids[1:] != segment_ids[:-1]) + 1
        ep_starts = np.concatenate([[0], changes]).astype(np.int64)
        ep_ends = np.concatenate([changes, [self._num_frames]]).astype(np.int64)
        self.traj_lengths = (ep_ends - ep_starts).astype(np.int64)

        indices = []
        episode_of_index = []
        for ep, (s, e) in enumerate(zip(ep_starts, ep_ends)):
            for t in range(int(s), int(e) - self.action_horizon + 1):
                indices.append(t)
                episode_of_index.append(ep)
        indices = np.asarray(indices, dtype=np.int64)
        episode_of_index = np.asarray(episode_of_index, dtype=np.int64)
        if len(indices) == 0:
            raise ValueError(
                f"No trainable windows in {self.dataset_path}; "
                f"action_horizon={self.action_horizon}"
            )

        rng = np.random.default_rng(seed)
        n_val_ep = max(1, int(len(self.traj_lengths) * float(val_ratio)))
        val_eps = set(rng.choice(len(self.traj_lengths), n_val_ep, replace=False).tolist())
        is_val = np.asarray([ep in val_eps for ep in episode_of_index], dtype=bool)

        self._train_indices = indices[~is_val]
        self._val_indices = indices[is_val]
        self._ep_starts = ep_starts
        self._ep_ends = ep_ends
        self._ep_start_for = np.empty(self._num_frames, dtype=np.int64)
        for s, e in zip(ep_starts, ep_ends):
            self._ep_start_for[int(s):int(e)] = int(s)

        self._is_val = False
        self.action_type = "joint_pos"

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    @property
    def h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_path, "r")
        return self._h5

    def get_validation_dataset(self) -> "YAMHdf5Dataset":
        copy = YAMHdf5Dataset.__new__(YAMHdf5Dataset)
        copy.__dict__.update(self.__dict__)
        copy._h5 = None
        copy._is_val = True
        return copy

    def get_normalizer(self, **kwargs):
        sparse_norm = LinearNormalizer()
        state_stat = array_to_stats(self.states)
        action_stat = array_to_stats(self.actions)
        sparse_norm["joint_pos"] = get_range_normalizer_from_stat(state_stat)
        sparse_norm["action"] = get_range_normalizer_from_stat(action_stat)
        sparse_norm["rgb_0"] = get_image_identity_normalizer()
        sparse_norm["rgb_1"] = get_image_identity_normalizer()
        return sparse_norm, None

    def __len__(self) -> int:
        return len(self._val_indices if self._is_val else self._train_indices)

    def _decode_rgb_chw(self, key: str, idx: int) -> np.ndarray:
        encoded = np.asarray(self.h5[key][idx], dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to decode {key}[{idx}] from {self.dataset_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.image_size > 0 and rgb.shape[:2] != (self.image_size, self.image_size):
            rgb = cv2.resize(
                rgb,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        return np.ascontiguousarray(chw)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        idx = self._val_indices if self._is_val else self._train_indices
        t = int(idx[i])
        ep_s = int(self._ep_start_for[t])

        obs_frames = []
        for k in range(self.obs_horizon - 1, -1, -1):
            obs_frames.append(max(t - k, ep_s))

        state_seq = self.states[obs_frames]
        rgb0 = np.stack([self._decode_rgb_chw("rgb_0", j) for j in obs_frames], axis=0)
        rgb1 = np.stack([self._decode_rgb_chw("rgb_1", j) for j in obs_frames], axis=0)
        action_seq = self.actions[t : t + self.action_horizon]

        obs_sparse: Dict[str, torch.Tensor] = {
            "rgb_0": torch.from_numpy(rgb0),
            "rgb_1": torch.from_numpy(rgb1),
            "joint_pos": torch.from_numpy(state_seq),
        }
        return {
            "obs": {"sparse": obs_sparse},
            "action": {"sparse": torch.from_numpy(action_seq)},
        }
