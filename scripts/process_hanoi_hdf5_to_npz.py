#!/usr/bin/env python3
"""Convert the Hanoi HDF5 demonstrations into YAM RL/Diffusion NPZ format.

Input HDF5 layout:
  rgb_0, rgb_1    (T,) JPEG byte arrays
  trajectory      (T, 7) raw joint/gripper positions
  segment_ids     (T,) episode ids

Output directory:
  train.npz          states/actions/images/traj_lengths for BC/RL replay
  normalization.npz  raw min/max used by envrunner for live state/action scaling
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch


DEFAULT_HDF5 = "/home/bike/Documents/niu/Dataset/test/hanoi_stack6.hdf5"
DEFAULT_OUT = "~/data/real_processed/stack_green_hanoi_cube_224"


def _resize_short_side_and_center_crop(rgb: np.ndarray, target: int = 256) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = max(target / w, target / h)
    new_w = max(target, int(np.ceil(w * scale)))
    new_h = max(target, int(np.ceil(h * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (new_w - target) // 2
    y0 = (new_h - target) // 2
    return resized[y0:y0 + target, x0:x0 + target]


def _decode_chw(encoded, image_size: int) -> np.ndarray:
    data = np.asarray(encoded, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV failed to decode JPEG frame")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = _resize_short_side_and_center_crop(rgb, target=256)
    if image_size > 0 and rgb.shape[:2] != (image_size, image_size):
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        rgb_t = torch.nn.functional.interpolate(
            rgb_t,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        return rgb_t.squeeze(0).clamp(0, 255).to(torch.uint8).numpy()
    return np.transpose(rgb, (2, 0, 1)).astype(np.uint8, copy=False)


def _normalize(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    y = 2.0 * (x - lo) / (hi - lo + 1e-6) - 1.0
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default=DEFAULT_HDF5)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    hdf5_path = Path(os.path.expanduser(args.hdf5))
    out_dir = Path(os.path.expanduser(args.out_dir))
    train_npz = out_dir / "train.npz"
    norm_npz = out_dir / "normalization.npz"

    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)
    if train_npz.exists() and norm_npz.exists() and not args.force:
        print(f"[process] exists: {train_npz}")
        print(f"[process] exists: {norm_npz}")
        print("[process] use --force to regenerate")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        trajectory = np.asarray(f["trajectory"], dtype=np.float32)
        segment_ids = np.asarray(f["segment_ids"], dtype=np.int64)
        if trajectory.ndim != 2 or trajectory.shape[1] != 7:
            raise ValueError(f"trajectory must be (T, 7), got {trajectory.shape}")
        if len(segment_ids) != len(trajectory):
            raise ValueError("segment_ids length does not match trajectory")

        changes = np.flatnonzero(segment_ids[1:] != segment_ids[:-1]) + 1
        ep_starts = np.concatenate([[0], changes]).astype(np.int64)
        ep_ends = np.concatenate([changes, [len(segment_ids)]]).astype(np.int64)
        traj_lengths = (ep_ends - ep_starts).astype(np.int64)

        obs_min = trajectory.min(axis=0).astype(np.float32)
        obs_max = trajectory.max(axis=0).astype(np.float32)
        states = _normalize(trajectory, obs_min, obs_max)
        actions = states.copy()

        T = int(len(trajectory))
        H = int(args.image_size)
        images = np.empty((T, 6, H, H), dtype=np.uint8)
        for i in range(T):
            images[i, :3] = _decode_chw(f["rgb_0"][i], H)
            images[i, 3:] = _decode_chw(f["rgb_1"][i], H)
            if (i + 1) % 500 == 0 or (i + 1) == T:
                print(f"[process] decoded {i + 1}/{T} frames", flush=True)

    rewards = np.zeros((len(states),), dtype=np.float32)
    terminals = np.zeros((len(states),), dtype=bool)

    tmp_train = train_npz.with_suffix(".npz.tmp")
    tmp_norm = norm_npz.with_suffix(".npz.tmp")
    print(f"[process] saving {train_npz}")
    with open(tmp_train, "wb") as f:
        np.savez_compressed(
            f,
            states=states,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            traj_lengths=traj_lengths,
            images=images,
        )
    print(f"[process] saving {norm_npz}")
    with open(tmp_norm, "wb") as f:
        np.savez(
            f,
            obs_min=obs_min,
            obs_max=obs_max,
            action_min=obs_min.copy(),
            action_max=obs_max.copy(),
        )
    os.replace(tmp_train, train_npz)
    os.replace(tmp_norm, norm_npz)

    print("[process] done")
    print(f"  episodes: {len(traj_lengths)}")
    print(f"  frames:   {len(states)}")
    print(f"  lengths:  min={traj_lengths.min()} mean={traj_lengths.mean():.2f} max={traj_lengths.max()}")
    print(f"  states:   min={states.min():.3f} max={states.max():.3f}")
    print(f"  images:   {images.shape} {images.dtype}")


if __name__ == "__main__":
    main()
