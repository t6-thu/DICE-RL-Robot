#!/usr/bin/env python3
"""View saved YAM RL episode videos with an interactive scrubber.

Usage:
    # play a single episode
    python scripts/view_episode.py /path/to/episode_0007.npz

    # play all episodes in a directory (default = configured ONLINE_DATA_DIR)
    python scripts/view_episode.py

    # save to mp4 instead of live playback
    python scripts/view_episode.py /path/to/episode_0007.npz --save out.mp4

    # change playback FPS (default 15)
    python scripts/view_episode.py /path/to/episode_0007.npz --fps 30

Interactive controls:
    [drag scrollbar]  scrub to any frame (auto-pauses)
    SPACE             play / pause
    ← / a             previous frame
    → / d             next frame
    Home              jump to first frame
    End               jump to last frame
    r                 replay from start
    n                 next episode (directory mode)
    q                 quit
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

DEFAULT_DIR = os.path.expanduser("~/data/real_processed/yam_rl_rollouts_hire_lambda09")


def _to_uint8_bgr(rgb_float: np.ndarray) -> np.ndarray:
    """(3, H, W) float32 in [0,1] or uint8 → (H, W, 3) BGR uint8 for cv2."""
    if rgb_float.dtype == np.uint8:
        rgb = rgb_float
    else:
        rgb = (rgb_float.clip(0, 1) * 255).astype(np.uint8)
    # (C,H,W) → (H,W,C)
    rgb = np.transpose(rgb, (1, 2, 0))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _render_frame(images, rewards, name, T, H, W, success, i, ended=False):
    """Render frame i with overlay text. Returns BGR uint8 (H, 2W, 3)."""
    base  = _to_uint8_bgr(images[i, :3])
    wrist = _to_uint8_bgr(images[i, 3:])
    side  = np.concatenate([base, wrist], axis=1)
    r_i = float(rewards[i]) if i < len(rewards) else 0.0

    if ended:
        label = (f"{name}  END  step {i+1}/{T}  r={r_i:.2f}  "
                 f"{'SUCCESS' if success else 'FAILURE'}   "
                 f"[scroll/<-/->: scrub  SPACE: play  r: replay  n: next  q: quit]")
        color = (0, 255, 0) if success else (0, 0, 255)
    else:
        label = (f"{name}  step {i+1}/{T}  r={r_i:.2f}  "
                 f"{'SUCCESS' if success else 'FAILURE'}")
        color = (255, 255, 255)

    cv2.putText(side, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)
    cv2.putText(side, "base camera", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(side, "wrist camera", (W + 8, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return side


def play_episode(path: str, fps: float, save_path: str | None = None) -> str:
    d = np.load(path)
    images  = d["images"]    # (T, 6, H, W)
    rewards = d.get("rewards", np.zeros(len(images), dtype=np.float32))
    states  = d.get("states", None)
    name    = os.path.basename(path)
    T, _, H, W = images.shape
    success = bool(rewards[-1] > 0.5) if len(rewards) > 0 else False

    print(f"\n=== {name} ===")
    print(f"  steps (action chunks): {T}")
    print(f"  image shape: {H}x{W}, dtype={images.dtype}")
    print(f"  final reward: {rewards[-1] if len(rewards) > 0 else 'n/a'}  ({'SUCCESS' if success else 'FAILURE'})")
    if states is not None:
        print(f"  state shape: {states.shape}")

    # ----- Save-only fast path -----
    if save_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (W * 2, H))
        print(f"  → saving to {save_path}")
        for i in range(T):
            writer.write(_render_frame(images, rewards, name, T, H, W, success, i))
        writer.release()
        return "done"

    # ----- Interactive playback with trackbar -----
    title = f"{name}  [{'SUCCESS' if success else 'FAILURE'}]"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    # State shared with the trackbar callback.
    state = {"frame": 0, "paused": False, "suppress_cb": False}

    def on_seek(pos: int) -> None:
        # User dragged the bar → jump to that frame and pause.
        if state["suppress_cb"]:
            return
        state["frame"] = max(0, min(T - 1, pos))
        state["paused"] = True

    # If T==1 OpenCV refuses to make a slider with max=0; fall back to a 1-tick bar.
    cv2.createTrackbar("frame", title, 0, max(T - 1, 1), on_seek)

    delay = max(1, int(1000.0 / fps))
    while True:
        i = state["frame"]
        ended = (i >= T - 1) and state["paused"]
        frame = _render_frame(images, rewards, name, T, H, W, success, i, ended=ended)
        cv2.imshow(title, frame)

        # Keep trackbar in sync without triggering the callback again.
        state["suppress_cb"] = True
        cv2.setTrackbarPos("frame", title, i)
        state["suppress_cb"] = False

        wait = delay if not state["paused"] else 30
        key = cv2.waitKey(wait) & 0xFFFF       # 16-bit catches arrow / Home / End on most platforms

        # Letter keys
        k8 = key & 0xFF
        if k8 == ord("q") or k8 == 27:          # q or ESC
            cv2.destroyAllWindows()
            return "quit"
        if k8 == ord("n"):
            cv2.destroyAllWindows()
            return "next"
        if k8 == ord(" "):
            # at the very end, SPACE rewinds + plays
            if state["frame"] >= T - 1:
                state["frame"] = 0
            state["paused"] = not state["paused"]
            continue
        if k8 == ord("r"):
            state["frame"] = 0
            state["paused"] = False
            continue
        if k8 == ord("a"):                       # vim-style prev
            state["frame"] = max(0, state["frame"] - 1)
            state["paused"] = True
            continue
        if k8 == ord("d"):                       # vim-style next
            state["frame"] = min(T - 1, state["frame"] + 1)
            state["paused"] = True
            continue

        # Arrow keys & Home/End — codes vary across backends; check both 8- and 16-bit
        if key in (81, 0x250000, 2424832):       # ←
            state["frame"] = max(0, state["frame"] - 1); state["paused"] = True; continue
        if key in (83, 0x270000, 2555904):       # →
            state["frame"] = min(T - 1, state["frame"] + 1); state["paused"] = True; continue
        if key in (80, 0x240000, 2359296):       # Home
            state["frame"] = 0; state["paused"] = True; continue
        if key in (87, 0x230000, 2293760):       # End
            state["frame"] = T - 1; state["paused"] = True; continue

        # Auto-advance if playing
        if not state["paused"]:
            if state["frame"] < T - 1:
                state["frame"] += 1
            else:
                state["paused"] = True            # reached end → pause


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default=DEFAULT_DIR,
                   help="episode .npz file OR directory containing episode_*.npz (default: %(default)s)")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--save", type=str, default=None,
                   help="save to this .mp4 path (single-episode mode only)")
    p.add_argument("--start", type=int, default=0,
                   help="when path is a dir, start from this episode index")
    args = p.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "episode_*.npz")))
        if not files:
            print(f"No episode_*.npz found under {args.path}")
            sys.exit(1)
        files = files[args.start:]
        print(f"Found {len(files)} episode(s) under {args.path} (starting from index {args.start})")
        for f in files:
            r = play_episode(f, args.fps, save_path=None)
            if r == "quit":
                break
    else:
        play_episode(args.path, args.fps, save_path=args.save)


if __name__ == "__main__":
    main()
