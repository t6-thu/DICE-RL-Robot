#!/usr/bin/env python3
"""Export every episode_*.npz in a directory to an mp4 (skipping chosen indices).

Exports base, wrist, or [base | wrist] without text overlays. The base camera
is cropped to its BOTTOM-LEFT 3/4 × 3/4 region and resized to the requested
output size.

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
import subprocess
import sys

import cv2
import numpy as np

def _camera_slices(policy_camera_order):
    """Return physical camera slices for the order stored in an episode."""
    if policy_camera_order == "base_wrist":
        return {"base": slice(0, 3), "wrist": slice(3, 6)}
    if policy_camera_order == "wrist_base":
        return {"wrist": slice(0, 3), "base": slice(3, 6)}
    raise ValueError(
        "policy_camera_order must be base_wrist or wrist_base, got "
        f"{policy_camera_order!r}"
    )


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


def _resize_and_sharpen(frame, output_size, sharpen):
    """Upscale for display and optionally apply a mild unsharp mask."""
    if frame.shape[:2] != (output_size, output_size):
        frame = cv2.resize(
            frame, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4
        )
    if sharpen > 0:
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.0)
        frame = cv2.addWeighted(frame, 1.0 + sharpen, blurred, -sharpen, 0)
    return frame


def _render_frame(images, i, crop_fraction, camera_slices, camera,
                  output_size, sharpen):
    """Render physical base, wrist, or [base | wrist], without overlays."""
    base_full = _to_uint8_rgb(images[i, camera_slices["base"]])
    wrist = _to_uint8_rgb(images[i, camera_slices["wrist"]])

    base_crop = _crop_bottom_left(base_full, crop_fraction)
    base = _resize_and_sharpen(base_crop, output_size, sharpen)
    wrist = _resize_and_sharpen(wrist, output_size, sharpen)

    base_bgr  = cv2.cvtColor(base,  cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
    if camera == "base":
        side = base_bgr
    elif camera == "wrist":
        side = wrist_bgr
    else:
        side = np.concatenate([base_bgr, wrist_bgr], axis=1)
    return side


def export_one(path, out_dir, fps, crop_fraction, camera, output_size,
               crf, sharpen, suffix):
    with np.load(path, allow_pickle=False) as d:
        images = d["images"]
        rewards = d.get("rewards", np.zeros(len(images), dtype=np.float32))
        policy_camera_order = (
            str(np.asarray(d["policy_camera_order"]).item())
            if "policy_camera_order" in d else "base_wrist"
        )
    camera_slices = _camera_slices(policy_camera_order)
    name    = os.path.splitext(os.path.basename(path))[0]
    success = bool(len(rewards) and rewards[-1] > 0.5)
    T = len(images)

    # Use ffmpeg/libx264 instead of OpenCV's low-bitrate mp4v writer.  Upscaling
    # does not invent detail, but it avoids poor player-side enlargement and
    # CRF controls the actual compression quality.
    out_path = os.path.join(out_dir, f"{name}{suffix}.mp4")
    out_w = output_size * 2 if camera == "both" else output_size
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{output_size}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path,
    ]
    try:
        writer = subprocess.Popen(command, stdin=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("high-quality export requires ffmpeg on PATH") from exc
    try:
        assert writer.stdin is not None
        for i in range(T):
            frame = _render_frame(
                images, i, crop_fraction, camera_slices, camera,
                output_size, sharpen,
            )
            writer.stdin.write(frame.tobytes())
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while writing {out_path}")
    finally:
        if writer.stdin is not None and not writer.stdin.closed:
            writer.stdin.close()
        if writer.poll() is None:
            writer.kill()
            writer.wait()
    print(
        f"  {name}  [{'SUCCESS' if success else 'FAILURE'}]  {T} frames  "
        f"order={policy_camera_order} camera={camera} size={out_w}x{output_size} "
        f"crf={crf} sharpen={sharpen:.2f} → {out_path}"
    )
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", help="directory of episode_*.npz OR a single episode_*.npz file")
    p.add_argument("--skip", type=int, nargs="*", default=[],
                   help="episode index/indices to skip, e.g. --skip 56")
    p.add_argument("--out", default=None,
                   help="output dir (default: <dir>/mp4_exports)")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--camera", choices=["base", "wrist", "both"], default="both",
                   help="physical camera view to export (default: both)")
    p.add_argument("--crop", type=float, default=0.75,
                   help="base bottom-left crop fraction (default 0.75 = 3/4)")
    p.add_argument("--output-size", type=int, default=224,
                   help="height/width of each camera pane (default: 224)")
    p.add_argument("--crf", type=int, default=12,
                   help="H.264 quality, lower is clearer/larger (default: 12)")
    p.add_argument("--sharpen", type=float, default=0.0,
                   help="unsharp-mask strength; 0 disables it (default: 0)")
    p.add_argument("--suffix", default="",
                   help="append to output basename, e.g. --suffix _hd_clean")
    args = p.parse_args()
    if not 0 < args.crop <= 1:
        p.error("--crop must be in (0, 1]")
    if args.output_size <= 0:
        p.error("--output-size must be > 0")
    if not 0 <= args.crf <= 51:
        p.error("--crf must be between 0 and 51")
    if args.sharpen < 0:
        p.error("--sharpen must be >= 0")

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
    print(
        f"Base crop: bottom-left {args.crop:.0%}×{args.crop:.0%}; "
        f"each pane resized to {args.output_size}x{args.output_size}"
    )
    print(f"Output dir: {out_dir}\n")

    for f in selected:
        export_one(
            f, out_dir, fps=args.fps, crop_fraction=args.crop,
            camera=args.camera, output_size=args.output_size,
            crf=args.crf, sharpen=args.sharpen, suffix=args.suffix,
        )

    print(f"\n✓ Done. {len(selected)} mp4s in {out_dir}")


if __name__ == "__main__":
    main()
