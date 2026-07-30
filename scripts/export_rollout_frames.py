#!/usr/bin/env python3
"""Export sampled frames from YAM rollout episode .npz files to PNG images.

Base camera uses the same top-left crop as extract_episode_starts.py.
Wrist camera is saved full-frame. No text overlays.

Success/fail label: rewards[-1] > 0.5 → SUCCESS, else FAILURE.

Usage:
    # one episode, every 10th frame (default)
    python scripts/export_rollout_frames.py \
        /path/to/episode_0000.npz

    # first 3 failure episodes in a directory
    python scripts/export_rollout_frames.py \
        /home/bike/data/real_processed/yam_rl_rollouts_chunkplusH_curated_2000_1000 \
        --label fail --limit 3
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

DEFAULT_CROP_FRACTION = 0.72
DEFAULT_DATA_DIR = os.path.expanduser(
    "~/data/real_processed/yam_rl_rollouts_chunkplusH_curated_2000_1000"
)
CAMERA_SLICES = {"base": slice(0, 3), "wrist": slice(3, 6)}


def episode_label(rewards: np.ndarray) -> str:
    if len(rewards) == 0:
        return "EMPTY"
    return "SUCCESS" if float(rewards[-1]) > 0.5 else "FAILURE"


def _data_root(path: str) -> str:
    """Dataset root: directory itself, or parent dir for episode_*.npz file."""
    if os.path.isdir(path):
        return path
    return os.path.dirname(os.path.abspath(path))


def _default_out_dir(path: str) -> str:
    return os.path.join(_data_root(path), "rollout_frames")


def _resolve_episodes(
    path: str,
    label: str | None,
    limit: int | None,
) -> list[str]:
    if os.path.isfile(path):
        return [path]

    files = sorted(glob.glob(os.path.join(path, "episode_*.npz")))
    if not files:
        raise FileNotFoundError(f"no episode_*.npz under {path}")

    if label is not None:
        want = label.lower()
        if want in ("fail", "failure"):
            want = "FAILURE"
        elif want in ("succ", "success"):
            want = "SUCCESS"
        else:
            raise ValueError(f"unknown label filter: {label}")

        filtered: list[str] = []
        for f in files:
            rewards = np.load(f).get("rewards", np.zeros(0, dtype=np.float32))
            if episode_label(rewards) == want:
                filtered.append(f)
        files = filtered
        if not files:
            raise ValueError(f"no episodes with label={label!r} under {path}")

    if limit is not None:
        files = files[:max(0, limit)]
    return files


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        arr = rgb
    else:
        arr = (rgb.clip(0, 1) * 255).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))


def _crop_top_left(frame: np.ndarray, fraction: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x2 = max(1, int(round(w * fraction)))
    y2 = max(1, int(round(h * fraction)))
    return frame[0:y2, 0:x2]


def extract_frame(
    images: np.ndarray,
    frame_idx: int,
    camera: str,
    crop_fraction: float | None,
) -> np.ndarray:
    i = max(0, min(frame_idx, len(images) - 1))
    frame = _to_uint8_rgb(images[i, CAMERA_SLICES[camera]])
    if camera == "base" and crop_fraction is not None:
        frame = _crop_top_left(frame, crop_fraction)
    return frame


def sample_indices(total: int, stride: int, include_last: bool = True) -> list[int]:
    if total <= 0:
        return []
    idx = list(range(0, total, max(1, stride)))
    if include_last and idx[-1] != total - 1:
        idx.append(total - 1)
    return idx


def export_rollout(
    episode_path: str,
    out_dir: str,
    stride: int = 10,
    crop_fraction: float = DEFAULT_CROP_FRACTION,
    name_prefix: str = "",
) -> None:
    if not os.path.isfile(episode_path):
        raise FileNotFoundError(episode_path)

    d = np.load(episode_path)
    images = d["images"]
    rewards = d.get("rewards", np.zeros(len(images), dtype=np.float32))
    name = os.path.splitext(os.path.basename(episode_path))[0]
    if name_prefix:
        name = f"{name_prefix}{name}"
    label = episode_label(rewards)
    T = len(images)
    indices = sample_indices(T, stride=stride)

    base_dir = os.path.join(out_dir, name, "base")
    wrist_dir = os.path.join(out_dir, name, "wrist")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(wrist_dir, exist_ok=True)

    print(f"Exporting {name}  [{label}]")
    print(f"  source : {episode_path}")
    print(f"  frames : {T} total, exporting {len(indices)} (stride={stride})")
    print(f"  reward : final={float(rewards[-1]) if len(rewards) else 'n/a'}")
    print(f"  crop   : base top-left {crop_fraction:.0%}, wrist none")
    print(f"  output : {out_dir}")

    for i in indices:
        base = extract_frame(images, i, "base", crop_fraction)
        wrist = extract_frame(images, i, "wrist", crop_fraction=None)
        tag = f"frame_{i:04d}.png"
        cv2.imwrite(os.path.join(base_dir, tag), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(wrist_dir, tag), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))

    print(f"  saved  : {len(indices)} base + {len(indices)} wrist PNGs")
    print(f"  size   : base {base.shape[1]}x{base.shape[0]}, wrist {wrist.shape[1]}x{wrist.shape[0]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_DATA_DIR,
        help="episode .npz file OR directory of episode_*.npz",
    )
    p.add_argument("--out", default=None,
                   help="output root (default: <data_dir>/rollout_frames)")
    p.add_argument("--stride", type=int, default=10,
                   help="export every N-th frame (default: 10)")
    p.add_argument("--crop-fraction", type=float, default=DEFAULT_CROP_FRACTION,
                   help="base camera top-left crop fraction (default: 0.72)")
    p.add_argument("--label", choices=["success", "fail", "failure"], default=None,
                   help="only export episodes with this label (directory mode)")
    p.add_argument("--limit", type=int, default=None,
                   help="max number of episodes to export (directory mode)")
    p.add_argument("--name-prefix", default="",
                   help="prefix for output episode folder names (avoid collisions)")
    args = p.parse_args()

    try:
        episodes = _resolve_episodes(args.path, label=args.label, limit=args.limit)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        sys.exit(1)

    out_dir = args.out or _default_out_dir(args.path)

    print(f"Selected {len(episodes)} episode(s)"
          + (f"  label={args.label}" if args.label else ""))
    print(f"Output root: {out_dir}")

    for episode_path in episodes:
        try:
            export_rollout(
                episode_path,
                out_dir=out_dir,
                stride=args.stride,
                crop_fraction=args.crop_fraction,
                name_prefix=args.name_prefix,
            )
            print()
        except FileNotFoundError as exc:
            print(exc)
            sys.exit(1)


if __name__ == "__main__":
    main()
