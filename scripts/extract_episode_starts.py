#!/usr/bin/env python3
"""Extract the first-frame camera image from every YAM RL episode .npz.

Supports base and wrist cameras. No text overlays. Base crop is optional;
wrist is always saved full-frame. Saves lossless PNG at native resolution.

Usage:
    python scripts/extract_episode_starts.py

    python scripts/extract_episode_starts.py --camera wrist

    # offline expert train.npz (uses traj_lengths for episode starts)
    python scripts/extract_episode_starts.py \
        /home/bike/data/real_processed/yam_picknplace_arizonabottle --overlay

    python scripts/extract_episode_starts.py \
        /home/bike/data/real_processed/yam_rl_rollouts_chunkplusH_curated_2000_1000

    python scripts/extract_episode_starts.py /path/to/episodes --grid --cols 10

    # overlap all start frames to inspect initial-state distribution
    python scripts/extract_episode_starts.py --overlay-only \
        --overlay-dir /path/to/episode_starts_base
"""
import argparse
import glob
import math
import os
import sys

import cv2
import numpy as np

DEFAULT_DIR = os.path.expanduser(
    "~/data/real_processed/yam_rl_rollouts_chunkplusH_curated_2000_1000"
)
DEFAULT_CROP_FRACTION = 0.72
CAMERA_SLICES = {"base": slice(0, 3), "wrist": slice(3, 6)}


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    """(3, H, W) float32 in [0,1] or uint8 → (H, W, 3) RGB uint8."""
    if rgb.dtype == np.uint8:
        arr = rgb
    else:
        arr = (rgb.clip(0, 1) * 255).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))


def _crop_top_left(frame: np.ndarray, fraction: float) -> np.ndarray:
    """Crop from top-left origin, keeping `fraction` of width and height."""
    if not (0 < fraction <= 1):
        raise ValueError(f"crop fraction must be in (0, 1], got {fraction}")
    h, w = frame.shape[:2]
    x2 = max(1, int(round(w * fraction)))
    y2 = max(1, int(round(h * fraction)))
    return frame[0:y2, 0:x2]


def extract_start_frame(
    images: np.ndarray,
    frame_idx: int = 0,
    camera: str = "base",
    crop_fraction: float | None = DEFAULT_CROP_FRACTION,
) -> np.ndarray:
    """Return RGB uint8 camera frame at `frame_idx`."""
    if camera not in CAMERA_SLICES:
        raise ValueError(f"unknown camera: {camera}")
    T = len(images)
    i = max(0, min(frame_idx, T - 1))
    frame = _to_uint8_rgb(images[i, CAMERA_SLICES[camera]])
    if camera == "base" and crop_fraction is not None:
        frame = _crop_top_left(frame, crop_fraction)
    return frame


def _default_out_dir(data_dir: str, camera: str) -> str:
    return os.path.join(data_dir, f"episode_starts_{camera}")


def _resolve_train_npz(path: str) -> str | None:
    if os.path.isfile(path) and path.endswith(".npz") and os.path.basename(path) == "train.npz":
        return path
    if os.path.isdir(path):
        candidate = os.path.join(path, "train.npz")
        if os.path.isfile(candidate):
            return candidate
    return None


def _iter_rollout_episodes(path: str, limit: int | None) -> list[tuple[str, str]]:
    files = sorted(glob.glob(os.path.join(path, "episode_*.npz")))
    if limit is not None:
        files = files[:max(0, limit)]
    return [(os.path.splitext(os.path.basename(f))[0], f) for f in files]


def _iter_expert_episode_starts(
    train_npz: str,
    limit: int | None,
) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    """Yield (episode_name, images_T6HW, rewards_or_None) for each expert traj."""
    d = np.load(train_npz)
    if "traj_lengths" not in d or "images" not in d:
        raise ValueError(f"{train_npz} is missing traj_lengths/images")
    images = d["images"]
    traj_lengths = d["traj_lengths"].astype(int)
    rewards = d.get("rewards", None)
    ep_starts = np.concatenate([[0], np.cumsum(traj_lengths[:-1])])
    n_episodes = len(traj_lengths)
    if limit is not None:
        n_episodes = min(n_episodes, max(0, limit))

    episodes: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    for ep in range(n_episodes):
        start = int(ep_starts[ep])
        end = start + int(traj_lengths[ep])
        ep_images = images[start:end]
        ep_rewards = None if rewards is None else rewards[start:end]
        episodes.append((f"episode_{ep:04d}", ep_images, ep_rewards))
    return episodes


def _thumb(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _load_rgb_png(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_overlay(
    frames: list[np.ndarray],
    mode: str = "mean",
) -> np.ndarray:
    """Stack frames with equal weight; mean shows consensus, max shows all peaks."""
    if not frames:
        raise ValueError("no frames to overlay")
    stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
    if mode == "mean":
        out = stack.mean(axis=0)
    elif mode == "max":
        out = stack.max(axis=0)
    else:
        raise ValueError(f"unknown overlay mode: {mode}")
    return out.clip(0, 255).astype(np.uint8)


def build_direct_overlay(frames: list[np.ndarray], alpha: float = 0.4) -> np.ndarray:
    """Direct translucent stack: each frame painted on top of the previous."""
    if not frames:
        raise ValueError("no frames to overlay")
    out = frames[0].astype(np.float32)
    for frame in frames[1:]:
        f = frame.astype(np.float32)
        out = out * (1.0 - alpha) + f * alpha
    return out.clip(0, 255).astype(np.uint8)


def build_std_heatmap(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel RGB std → single-channel heatmap for position spread."""
    stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
    std = stack.std(axis=0).mean(axis=2)  # (H, W)
    if std.max() > 0:
        norm = (std / std.max() * 255).astype(np.uint8)
    else:
        norm = np.zeros(std.shape, dtype=np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def save_overlays(
    frames: list[np.ndarray],
    out_dir: str,
    mode: str = "all",
    direct_alpha: float = 0.4,
) -> None:
    if mode in ("all", "direct"):
        direct = build_direct_overlay(frames, alpha=direct_alpha)
        direct_path = os.path.join(out_dir, "overlay_direct.png")
        cv2.imwrite(direct_path, cv2.cvtColor(direct, cv2.COLOR_RGB2BGR))
        print(f"  overlay direct: {direct_path}")

    if mode in ("all", "stats"):
        mean = build_overlay(frames, mode="mean")
        mx = build_overlay(frames, mode="max")
        std = build_std_heatmap(frames)

        mean_path = os.path.join(out_dir, "overlay_mean.png")
        max_path = os.path.join(out_dir, "overlay_max.png")
        std_path = os.path.join(out_dir, "overlay_std_heatmap.png")
        cv2.imwrite(mean_path, cv2.cvtColor(mean, cv2.COLOR_RGB2BGR))
        cv2.imwrite(max_path, cv2.cvtColor(mx, cv2.COLOR_RGB2BGR))
        cv2.imwrite(std_path, cv2.cvtColor(std, cv2.COLOR_RGB2BGR))
        print(f"  overlay mean : {mean_path}")
        print(f"  overlay max  : {max_path}")
        print(f"  overlay std  : {std_path}")


def build_grid(thumbs: list[np.ndarray], cols: int, pad: int = 4) -> np.ndarray:
    if not thumbs:
        raise ValueError("no thumbnails to grid")
    cell_h = max(t.shape[0] for t in thumbs)
    cell_w = max(t.shape[1] for t in thumbs)
    rows = math.ceil(len(thumbs) / cols)

    canvas_h = rows * cell_h + (rows + 1) * pad
    canvas_w = cols * cell_w + (cols + 1) * pad
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        y = pad + r * (cell_h + pad)
        x = pad + c * (cell_w + pad)
        h, w = thumb.shape[:2]
        canvas[y:y + h, x:x + w] = thumb
    return canvas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default=DEFAULT_DIR,
                   help="directory containing episode_*.npz (default: %(default)s)")
    p.add_argument("--camera", choices=["base", "wrist", "both"], default="base",
                   help="which camera to extract (default: base)")
    p.add_argument("--out", default=None,
                   help="output directory (default: <path>/episode_starts_<camera>)")
    p.add_argument("--frame", type=int, default=0,
                   help="frame index to extract (default: 0 = episode start)")
    p.add_argument("--crop-fraction", type=float, default=DEFAULT_CROP_FRACTION,
                   help="keep this fraction from top-left (default: 0.72)")
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N episodes (for quick tests)")
    p.add_argument("--no-crop", action="store_true",
                   help="disable crop and save full base frame")
    p.add_argument("--grid", action="store_true",
                   help="also save a contact-sheet grid of all start frames")
    p.add_argument("--cols", type=int, default=10,
                   help="columns in contact sheet (default: 10)")
    p.add_argument("--thumb-width", type=int, default=224,
                   help="max width per thumbnail in contact sheet")
    p.add_argument("--overlay", action="store_true",
                   help="also save mean/max/std overlay images")
    p.add_argument("--overlay-only", action="store_true",
                   help="build overlays from existing *_start.png files and exit")
    p.add_argument("--overlay-dir", default=None,
                   help="directory of *_start.png for --overlay-only "
                        "(default: <path>/episode_starts_base)")
    p.add_argument("--overlay-mode", choices=["all", "direct", "stats"], default="all",
                   help="overlay outputs: direct stack, stats, or both (default: all)")
    p.add_argument("--direct-alpha", type=float, default=0.4,
                   help="per-frame alpha for direct overlay (default: 0.4)")
    args = p.parse_args()

    if args.overlay_only:
        overlay_dir = args.overlay_dir
        if overlay_dir is None:
            if not os.path.isdir(args.path):
                print(f"Not a directory: {args.path}")
                sys.exit(1)
            overlay_dir = os.path.join(args.path, "episode_starts_base")
        if not os.path.isdir(overlay_dir):
            print(f"Overlay dir not found: {overlay_dir}")
            sys.exit(1)
        pngs = sorted(glob.glob(os.path.join(overlay_dir, "episode_*_start.png")))
        if not pngs:
            print(f"No episode_*_start.png found under {overlay_dir}")
            sys.exit(1)
        frames = [_load_rgb_png(p) for p in pngs]
        print(f"Overlapping {len(frames)} start frame(s) from {overlay_dir}")
        save_overlays(
            frames, overlay_dir,
            mode=args.overlay_mode,
            direct_alpha=args.direct_alpha,
        )
        return

    train_npz = _resolve_train_npz(args.path)
    rollout_root = args.path if os.path.isdir(args.path) else os.path.dirname(args.path)
    rollout_eps = _iter_rollout_episodes(rollout_root, args.limit) if train_npz is None else []
    expert_eps = _iter_expert_episode_starts(train_npz, args.limit) if train_npz else []

    if not rollout_eps and not expert_eps:
        print(f"No episode_*.npz or train.npz found under {args.path}")
        sys.exit(1)

    data_root = rollout_root if train_npz is None else os.path.dirname(train_npz)
    cameras = ["base", "wrist"] if args.camera == "both" else [args.camera]
    crop_fraction = None if args.no_crop else args.crop_fraction

    for camera in cameras:
        if args.out is not None and len(cameras) > 1:
            out_dir = os.path.join(args.out, f"episode_starts_{camera}")
        else:
            out_dir = args.out or _default_out_dir(data_root, camera)
        os.makedirs(out_dir, exist_ok=True)

        thumbs: list[np.ndarray] = []
        overlay_frames: list[np.ndarray] = []
        n_succ = n_fail = 0
        n_saved = 0

        if train_npz:
            n_items = len(expert_eps)
            source = train_npz
        else:
            n_items = len(rollout_eps)
            source = rollout_root

        print(f"Extracting {camera}-cam frame {args.frame} from {n_items} episode(s)")
        print(f"  input : {source}")
        print(f"  output: {out_dir}")
        if camera == "base":
            if crop_fraction is None:
                print("  crop  : disabled (full frame)")
            else:
                print(f"  crop  : top-left origin, keep {crop_fraction:.0%}")
        else:
            print("  crop  : disabled (wrist full frame)")

        if train_npz:
            for name, images, rewards in expert_eps:
                frame = extract_start_frame(
                    images,
                    frame_idx=args.frame,
                    camera=camera,
                    crop_fraction=crop_fraction,
                )
                out_path = os.path.join(out_dir, f"{name}_start.png")
                cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                n_saved += 1
                if rewards is not None and len(rewards) > 0:
                    success = bool(rewards[-1] > 0.5)
                    n_succ += int(success)
                    n_fail += int(not success)
                thumbs.append(_thumb(frame, args.thumb_width))
                if camera == "base":
                    overlay_frames.append(frame)
        else:
            for name, path in rollout_eps:
                d = np.load(path)
                images = d["images"]
                rewards = d.get("rewards", np.zeros(len(images), dtype=np.float32))

                frame = extract_start_frame(
                    images,
                    frame_idx=args.frame,
                    camera=camera,
                    crop_fraction=crop_fraction,
                )
                out_path = os.path.join(out_dir, f"{name}_start.png")
                cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                n_saved += 1

                success = len(rewards) > 0 and bool(rewards[-1] > 0.5)
                n_succ += int(success)
                n_fail += int(not success)
                thumbs.append(_thumb(frame, args.thumb_width))
                if camera == "base":
                    overlay_frames.append(frame)

        if n_succ + n_fail > 0:
            print(f"  saved {n_saved} PNGs  (success={n_succ}, failure={n_fail})")
        else:
            print(f"  saved {n_saved} PNGs")
        if thumbs:
            print(f"  size  : {thumbs[0].shape[1]}x{thumbs[0].shape[0]} px")

        if args.grid:
            grid = build_grid(thumbs, cols=max(1, args.cols))
            grid_path = os.path.join(out_dir, "all_starts_grid.png")
            cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"  grid  : {grid_path}")

        if args.overlay and overlay_frames:
            save_overlays(
                overlay_frames, out_dir,
                mode=args.overlay_mode,
                direct_alpha=args.direct_alpha,
            )


if __name__ == "__main__":
    main()
