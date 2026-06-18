#!/usr/bin/env python3
"""Export every episode_*.npz in a directory to an mp4 (skipping chosen indices).

Layout per frame: [base | wrist] stitched horizontally, like view_episode.py,
but the base camera is cropped to its BOTTOM-LEFT 3/4 × 3/4 region and then
resized back up to the wrist camera's resolution so the two panes match.

    base:  crop x∈[0, 3/4 W),  y∈[1/4 H, H)   → resize to (wrist_H, wrist_W)
    wrist: full frame

Usage:
    . ./prepare.sh
    # whole dir, skip episode_0056, default fps 15
    python scripts/export_episodes_mp4.py \
        /home/bike/data/real_processed/yam_rl_rollouts_chunkplusH_curated_2000_1000 \
        --skip 56

    # custom out dir / fps / crop fraction
    python scripts/export_episodes_mp4.py <dir> --skip 56 --fps 30 \
        --out /tmp/ep_mp4 --crop 0.75
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

CAMERA_SLICES = {"base": slice(0, 3), "wrist": slice(3, 6)}


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    """(3,H,W) float[0,1] or uint8 → (H,W,3) RGB uint8."""
    arr = rgb if rgb.dtype == np.uint8 else (rgb.clip(0, 1) * 255).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))


def _crop_bottom_left(frame: np.ndarray, fraction: float) -> np.ndarray:
    """Keep the bottom-left fraction×fraction region of the frame."""
    h, w = frame.shape[:2]
    x2 = max(1, int(round(w * fraction)))          # 0 .. 3/4 W
    y1 = h - max(1, int(round(h * fraction)))      # 1/4 H .. H
    return frame[y1:h, 0:x2]


def _render_frame(images, rewards, name, success, i, crop_fraction):
    """Frame i → BGR uint8 (H, 2W, 3): [base(cropped+resized) | wrist]."""
    base_full  = _to_uint8_rgb(images[i, CAMERA_SLICES["base"]])
    wrist      = _to_uint8_rgb(images[i, CAMERA_SLICES["wrist"]])
    wh, ww = wrist.shape[:2]

    base_crop = _crop_bottom_left(base_full, crop_fraction)
    base = cv2.resize(base_crop, (ww, wh), interpolation=cv2.INTER_AREA)

    base_bgr  = cv2.cvtColor(base,  cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
    side = np.concatenate([base_bgr, wrist_bgr], axis=1)

    cv2.putText(side, "base", (8, wh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(side, "wrist", (ww + 8, wh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return side


def export_one(path, out_dir, fps, crop_fraction):
    d = np.load(path)
    images  = d["images"]
    rewards = d.get("rewards", np.zeros(len(images), dtype=np.float32))
    name    = os.path.splitext(os.path.basename(path))[0]
    success = bool(len(rewards) and rewards[-1] > 0.5)
    T, _, H, W = images.shape

    # Output size = stitched [resized-base | wrist] = (H, 2W) since base→wrist size.
    out_path = os.path.join(out_dir, f"{name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W * 2, H))
    for i in range(T):
        writer.write(_render_frame(images, rewards, name, success, i, crop_fraction))
    writer.release()
    print(f"  {name}  [{'SUCCESS' if success else 'FAILURE'}]  {T} frames  → {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", help="directory of episode_*.npz OR a single episode_*.npz file")
    p.add_argument("--skip", type=int, nargs="*", default=[],
                   help="episode index/indices to skip, e.g. --skip 56")
    p.add_argument("--out", default=None,
                   help="output dir (default: <dir>/mp4_exports)")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--crop", type=float, default=0.75,
                   help="base bottom-left crop fraction (default 0.75 = 3/4)")
    args = p.parse_args()

    if os.path.isfile(args.dir):
        files = [args.dir]
        base_dir = os.path.dirname(os.path.abspath(args.dir))
    elif os.path.isdir(args.dir):
        files = sorted(glob.glob(os.path.join(args.dir, "episode_*.npz")))
        base_dir = args.dir
        if not files:
            print(f"no episode_*.npz under {args.dir}"); sys.exit(1)
    else:
        print(f"not a file or directory: {args.dir}"); sys.exit(1)

    skip_set = set(args.skip)
    def ep_idx(f):
        return int(os.path.splitext(os.path.basename(f))[0].split("_")[1])
    selected = [f for f in files if ep_idx(f) not in skip_set]

    out_dir = args.out or os.path.join(base_dir, "mp4_exports")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Found {len(files)} episodes; skipping {sorted(skip_set)} "
          f"→ exporting {len(selected)}")
    print(f"Base crop: bottom-left {args.crop:.0%}×{args.crop:.0%}, resized to wrist size")
    print(f"Output dir: {out_dir}\n")

    for f in selected:
        export_one(f, out_dir, fps=args.fps, crop_fraction=args.crop)

    print(f"\n✓ Done. {len(selected)} mp4s in {out_dir}")


if __name__ == "__main__":
    main()
