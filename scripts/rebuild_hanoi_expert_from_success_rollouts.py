#!/usr/bin/env python3
"""Build a replacement Hanoi expert NPZ from successful saved rollouts.

This is a recovery tool for when the original teleoperation ``train.npz`` is
irretrievable. It deliberately uses only manually labelled successful rollout
episodes from the top level of one training run; evaluation episodes and failed
episodes are excluded.

The output remains provenance-labelled and must not be described as a recovery
of the original demonstrations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = (
    "~/文档/data/real_processed/"
    "yam_rl_rollouts_hanoi_hire_npz_epoch0500_wristbase_v2"
)
DEFAULT_NORM = "~/文档/data/real_processed/stack_green_hanoi_cube_224/normalization.npz"
DEFAULT_OUT = (
    "~/文档/data/real_processed/"
    "stack_green_hanoi_cube_224_recovered_success"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_success(path: Path, expected_order: str) -> tuple[int, tuple[int, int, int]] | None:
    with np.load(path, allow_pickle=False) as episode:
        required = {"images", "states", "actions", "rewards", "dones", "policy_camera_order"}
        missing = required.difference(episode.files)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")

        rewards = np.asarray(episode["rewards"])
        if not len(rewards) or float(rewards[-1]) <= 0.5:
            return None

        order = str(np.asarray(episode["policy_camera_order"]).item())
        if order != expected_order:
            raise ValueError(f"{path}: camera order {order!r}, expected {expected_order!r}")

        states = episode["states"]
        actions = episode["actions"]
        images = episode["images"]
        dones = episode["dones"]
        length = len(states)
        if states.shape != (length, 7) or actions.shape != (length, 7):
            raise ValueError(
                f"{path}: expected states/actions (T,7), got {states.shape}/{actions.shape}"
            )
        if images.ndim != 4 or images.shape[0] != length or images.shape[1] != 6:
            raise ValueError(f"{path}: expected images (T,6,H,W), got {images.shape}")
        if images.dtype != np.uint8:
            raise ValueError(f"{path}: images must be uint8, got {images.dtype}")
        if len(rewards) != length or len(dones) != length:
            raise ValueError(f"{path}: episode arrays have inconsistent lengths")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"{path}: non-finite state/action values")
        if float(states.min()) < -1.1 or float(states.max()) > 1.1:
            raise ValueError(f"{path}: states are outside the expected normalized range")
        if float(actions.min()) < -1.1 or float(actions.max()) > 1.1:
            raise ValueError(f"{path}: actions are outside the expected normalized range")
        return length, tuple(int(x) for x in images.shape[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE)
    parser.add_argument("--normalization", default=DEFAULT_NORM)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--camera-order", default="wrist_base")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_dir = Path(os.path.expanduser(args.source_dir)).resolve()
    norm_path = Path(os.path.expanduser(args.normalization)).resolve()
    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()
    train_path = out_dir / "train.npz"
    images_path = out_dir / "train_images.npy"
    output_norm = out_dir / "normalization.npz"
    manifest_path = out_dir / "recovery_manifest.json"

    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if not norm_path.is_file():
        raise FileNotFoundError(norm_path)
    outputs = (train_path, images_path, output_norm, manifest_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "recovery output already exists; inspect it or pass --force:\n  "
            + "\n  ".join(existing)
        )

    selected: list[tuple[Path, int]] = []
    frame_shape = None
    # Top-level glob is intentional: do not leak checkpoint evaluation data
    # into the replacement expert set.
    for path in sorted(source_dir.glob("episode_*.npz")):
        inspected = _inspect_success(path, args.camera_order)
        if inspected is None:
            continue
        length, shape = inspected
        if frame_shape is None:
            frame_shape = shape
        elif shape != frame_shape:
            raise ValueError(f"{path}: image shape {shape} differs from {frame_shape}")
        selected.append((path, length))

    if not selected or frame_shape is None:
        raise RuntimeError(f"no successful episodes found in {source_dir}")

    lengths = np.asarray([length for _, length in selected], dtype=np.int64)
    total_frames = int(lengths.sum())
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_images = out_dir / "train_images.tmp.npy"
    tmp_train = out_dir / "train.npz.tmp"
    tmp_norm = out_dir / "normalization.npz.tmp"
    tmp_manifest = out_dir / "recovery_manifest.json.tmp"

    print(
        f"[recover] selected {len(selected)} successful episodes, "
        f"{total_frames} frames from {source_dir}"
    )
    image_store = np.lib.format.open_memmap(
        tmp_images,
        mode="w+",
        dtype=np.uint8,
        shape=(total_frames, *frame_shape),
    )
    states = np.empty((total_frames, 7), dtype=np.float32)
    actions = np.empty((total_frames, 7), dtype=np.float32)

    cursor = 0
    sources = []
    for index, (path, length) in enumerate(selected, start=1):
        end = cursor + length
        with np.load(path, allow_pickle=False) as episode:
            image_store[cursor:end] = episode["images"]
            states[cursor:end] = episode["states"]
            actions[cursor:end] = episode["actions"]
        sources.append(
            {
                "file": path.name,
                "frames": length,
                "bytes": path.stat().st_size,
            }
        )
        cursor = end
        print(f"[recover] copied {index:02d}/{len(selected)}: {path.name} ({length} frames)")
    image_store.flush()

    rewards = np.zeros(total_frames, dtype=np.float32)
    terminals = np.zeros(total_frames, dtype=bool)
    print(f"[recover] writing compressed compatibility NPZ: {tmp_train}")
    with tmp_train.open("wb") as stream:
        np.savez_compressed(
            stream,
            states=states,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            traj_lengths=lengths,
            images=image_store,
            policy_camera_order=np.asarray(args.camera_order),
            recovery_source=np.asarray("successful_online_rollouts"),
        )
    del image_store

    shutil.copyfile(norm_path, tmp_norm)
    manifest = {
        "kind": "replacement_expert_from_successful_online_rollouts",
        "warning": "This is not the lost original teleoperation dataset.",
        "source_dir": str(source_dir),
        "camera_order": args.camera_order,
        "episodes": len(selected),
        "frames": total_frames,
        "image_shape": [total_frames, *frame_shape],
        "normalization_source": str(norm_path),
        "normalization_sha256": _sha256(norm_path),
        "sources": sources,
    }
    with tmp_manifest.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    os.replace(tmp_images, images_path)
    os.replace(tmp_train, train_path)
    os.replace(tmp_norm, output_norm)
    os.replace(tmp_manifest, manifest_path)
    print("[recover] complete")
    print(f"  train:         {train_path}")
    print(f"  image sidecar: {images_path}")
    print(f"  normalization: {output_norm}")
    print(f"  manifest:      {manifest_path}")


if __name__ == "__main__":
    main()
