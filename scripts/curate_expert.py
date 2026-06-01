#!/usr/bin/env python3
"""Interactive curation viewer for the offline expert npz.

Browse each expert demonstration one by one and tag it as INCLUDE / EXCLUDE
for the HiRE positive buffer. Decisions are auto-saved to a JSON sidecar so
you can resume any time.

Usage:
    python scripts/curate_expert.py
        # → opens train.npz (default), starts at the first unmarked episode

    python scripts/curate_expert.py --start 50
        # → start at episode index 50

    python scripts/curate_expert.py --npz /path/to/train.npz \
            --out  /path/to/expert_curation.json

Interactive controls (inside the OpenCV window):
    +           INCLUDE this episode  (move to next)
    -           EXCLUDE this episode  (move to next)
    SPACE       play / pause
    n           next episode (no decision)
    p           previous episode
    [drag bar]  scrub frames
    ← / a       previous frame   |  → / d  next frame
    Home / End  first / last frame
    s           print current summary
    q / ESC     save & quit

Output JSON format (auto-saved on every mark):
{
    "source":   "/abs/path/to/train.npz",
    "n_episodes": 117,
    "include":  [0, 1, 3, ...],
    "exclude":  [2, 5, 10, ...]
}

Then point the learner config at this JSON via `HIRE_EXPERT_CURATION_PATH`.
"""
import argparse
import json
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np


def _to_uint8_bgr(rgb: np.ndarray) -> np.ndarray:
    """(3, H, W) uint8 → (H, W, 3) BGR uint8."""
    if rgb.dtype != np.uint8:
        rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(np.transpose(rgb, (1, 2, 0)), cv2.COLOR_RGB2BGR)


def _load_curation(out_path: str, source: str, n_episodes: int) -> dict:
    if os.path.isfile(out_path):
        with open(out_path) as f:
            d = json.load(f)
        d.setdefault("include", []); d.setdefault("exclude", [])
        return d
    return {"source": os.path.abspath(source),
            "n_episodes": n_episodes,
            "include": [], "exclude": []}


def _save_curation(cur: dict, out_path: str) -> None:
    cur["include"] = sorted(set(int(x) for x in cur["include"]))
    cur["exclude"] = sorted(set(int(x) for x in cur["exclude"]))
    cur["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(out_path, "w") as f:
        json.dump(cur, f, indent=2)


def _status(ep: int, cur: dict) -> str:
    if ep in cur.get("include", []): return "INCLUDED"
    if ep in cur.get("exclude", []): return "EXCLUDED"
    return "UNMARKED"


def _mark(ep: int, cur: dict, kind: str) -> None:
    inc = set(cur.get("include", []))
    exc = set(cur.get("exclude", []))
    if kind == "include":
        inc.add(ep); exc.discard(ep)
    elif kind == "exclude":
        exc.add(ep); inc.discard(ep)
    cur["include"] = sorted(inc)
    cur["exclude"] = sorted(exc)


def _summary(cur: dict, n: int) -> str:
    inc = len(cur.get("include", []))
    exc = len(cur.get("exclude", []))
    return (f"include={inc}  exclude={exc}  unmarked={n - inc - exc}  "
            f"({100*inc/n:.0f}% / {100*exc/n:.0f}% / {100*(n-inc-exc)/n:.0f}%)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz",
        default=os.path.expanduser(
            "~/data/real_processed/yam_picknplace_arizonabottle_224/train.npz"))
    p.add_argument("--out", default=None,
        help="curation JSON path (default = <npz_dir>/expert_curation.json)")
    p.add_argument("--start", type=int, default=None,
        help="start episode index (default: first unmarked)")
    p.add_argument("--fps", type=float, default=15.0)
    args = p.parse_args()

    npz_path = os.path.expanduser(args.npz)
    if not os.path.isfile(npz_path):
        print(f"npz not found: {npz_path}"); sys.exit(1)
    out_path = args.out or os.path.join(os.path.dirname(npz_path),
                                        "expert_curation.json")

    # Prefer uncompressed sidecar npy for fast random-access mmap.
    sidecar = os.path.splitext(npz_path)[0] + "_images.npy"
    print(f"Loading expert episodes from {npz_path}")
    d = np.load(npz_path)
    if os.path.isfile(sidecar):
        images = np.load(sidecar, mmap_mode="r")
        print(f"  using mmap sidecar {sidecar}")
    else:
        images = d["images"]
        print(f"  no sidecar; loading images from npz (slower)")
    traj_lengths = d["traj_lengths"].astype(int)
    n_episodes = int(len(traj_lengths))
    ep_starts = np.concatenate([[0], np.cumsum(traj_lengths)])
    H, W = int(images.shape[2]), int(images.shape[3])

    cur = _load_curation(out_path, source=npz_path, n_episodes=n_episodes)
    print(f"Curation file: {out_path}")
    print(f"  current: {_summary(cur, n_episodes)}\n")

    # pick a starting episode
    if args.start is not None:
        ep = max(0, min(n_episodes - 1, int(args.start)))
    else:
        marked = set(cur["include"]) | set(cur["exclude"])
        ep = 0
        while ep < n_episodes and ep in marked:
            ep += 1
        if ep >= n_episodes: ep = 0

    title = "Curate Expert (+/- = include/exclude, n/p = next/prev, q = save+quit)"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    delay = max(1, int(1000.0 / args.fps))
    state = {"frame": 0, "paused": False, "suppress_cb": False, "T": 0}

    def on_seek(pos: int) -> None:
        if state["suppress_cb"]:
            return
        state["frame"] = max(0, min(state["T"] - 1, pos))
        state["paused"] = True

    cv2.createTrackbar("frame", title, 0, 1, on_seek)
    last_ep_shown = -1

    while True:
        # Per-episode setup (rebuilds when ep changes).
        if ep != last_ep_shown:
            start_idx = int(ep_starts[ep])
            T = int(traj_lengths[ep])
            state["T"] = T
            state["frame"] = 0
            state["paused"] = True
            state["suppress_cb"] = True
            cv2.setTrackbarMax("frame", title, max(T - 1, 1))
            cv2.setTrackbarPos("frame", title, 0)
            state["suppress_cb"] = False
            last_ep_shown = ep
            print(f"\n=== Episode {ep+1}/{n_episodes}   "
                  f"T={T} frames   status={_status(ep, cur)} ===")

        # Render current frame.
        i = state["frame"]
        global_idx = int(ep_starts[ep]) + i
        sel = np.asarray(images[global_idx])      # (6, H, W) uint8
        base  = _to_uint8_bgr(sel[:3])
        wrist = _to_uint8_bgr(sel[3:])
        side  = np.concatenate([base, wrist], axis=1)

        st = _status(ep, cur)
        color = ((0, 255, 0)   if st == "INCLUDED"
                 else (0, 0, 255) if st == "EXCLUDED"
                 else (200, 200, 200))
        label1 = f"Ep {ep+1}/{n_episodes}  step {i+1}/{state['T']}  [{st}]"
        label2 = f"+ include   - exclude   n next   p prev   space play/pause   q quit"
        cv2.putText(side, label1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(side, label2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(side, "base",  (8, H - 10),     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(side, "wrist", (W + 8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(title, side)

        state["suppress_cb"] = True
        cv2.setTrackbarPos("frame", title, i)
        state["suppress_cb"] = False

        key = cv2.waitKey(delay if not state["paused"] else 30) & 0xFFFF
        k8 = key & 0xFF

        # Episode-level commands
        if k8 in (ord("q"), 27):
            _save_curation(cur, out_path)
            print(f"\nSaved curation → {out_path}")
            print(f"Final: {_summary(cur, n_episodes)}")
            cv2.destroyAllWindows()
            return
        if k8 == ord("+") or k8 == ord("="):     # `=` is `+` without shift
            _mark(ep, cur, "include")
            _save_curation(cur, out_path)
            print(f"  → marked Ep {ep+1} INCLUDE   {_summary(cur, n_episodes)}")
            ep = min(n_episodes - 1, ep + 1)
            continue
        if k8 == ord("-") or k8 == ord("_"):
            _mark(ep, cur, "exclude")
            _save_curation(cur, out_path)
            print(f"  → marked Ep {ep+1} EXCLUDE   {_summary(cur, n_episodes)}")
            ep = min(n_episodes - 1, ep + 1)
            continue
        if k8 == ord("n"):
            ep = min(n_episodes - 1, ep + 1); continue
        if k8 == ord("p"):
            ep = max(0, ep - 1); continue
        if k8 == ord("s"):
            print(f"  summary: {_summary(cur, n_episodes)}"); continue

        # Frame-level commands
        if k8 == ord(" "):
            if state["frame"] >= state["T"] - 1: state["frame"] = 0
            state["paused"] = not state["paused"]; continue
        if k8 == ord("r"):
            state["frame"] = 0; state["paused"] = False; continue
        if k8 == ord("a"):
            state["frame"] = max(0, state["frame"] - 1); state["paused"] = True; continue
        if k8 == ord("d"):
            state["frame"] = min(state["T"] - 1, state["frame"] + 1); state["paused"] = True; continue
        # arrow / Home / End
        if key in (81, 0x250000, 2424832):   # ←
            state["frame"] = max(0, state["frame"] - 1); state["paused"] = True; continue
        if key in (83, 0x270000, 2555904):   # →
            state["frame"] = min(state["T"] - 1, state["frame"] + 1); state["paused"] = True; continue
        if key in (80, 0x240000, 2359296):   # Home
            state["frame"] = 0; state["paused"] = True; continue
        if key in (87, 0x230000, 2293760):   # End
            state["frame"] = state["T"] - 1; state["paused"] = True; continue

        # Auto-advance
        if not state["paused"]:
            if state["frame"] < state["T"] - 1:
                state["frame"] += 1
            else:
                state["paused"] = True


if __name__ == "__main__":
    main()
