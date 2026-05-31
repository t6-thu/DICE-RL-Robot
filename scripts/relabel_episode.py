#!/usr/bin/env python3
"""Relabel a saved YAM episode (flip success ↔ failure or set explicitly).

Usage:
    # show current label of one episode
    python scripts/relabel_episode.py path/to/episode_0005.npz

    # flip the label (success → failure or failure → success)
    python scripts/relabel_episode.py path/to/episode_0005.npz --flip

    # force to failure
    python scripts/relabel_episode.py path/to/episode_0005.npz --to fail

    # force to success
    python scripts/relabel_episode.py path/to/episode_0005.npz --to success

    # show labels of every episode under a directory
    python scripts/relabel_episode.py /path/to/dir --list

After relabeling, RESTART the learner so it re-scans the file and rebuilds
its HiRE pos/neg buffers + replay-buffer success flags from scratch.
"""
import argparse
import glob
import os
import sys

import numpy as np


def label_of(rewards: np.ndarray) -> str:
    if len(rewards) == 0:
        return "EMPTY"
    return "SUCCESS" if float(rewards[-1]) > 0.5 else "FAILURE"


def relabel(path: str, want: str | None, flip: bool) -> None:
    if not os.path.isfile(path):
        print(f"Not a file: {path}")
        sys.exit(1)
    d = dict(np.load(path))
    if "rewards" not in d:
        print(f"{path} has no `rewards` field"); sys.exit(1)
    r = d["rewards"].astype(np.float32).copy()
    old = label_of(r)
    if flip:
        target = 1.0 - float(r[-1] > 0.5)
    elif want == "success":
        target = 1.0
    elif want in ("fail", "failure"):
        target = 0.0
    else:
        print(f"{os.path.basename(path)}: {old}  (final reward = {r[-1]:.2f}, T = {len(r)})")
        return

    r[-1] = float(target)
    new = label_of(r)
    if old == new:
        print(f"{os.path.basename(path)}: already {old}, no change")
        return

    confirm = input(f"Relabel {os.path.basename(path)}: {old} → {new}? [y/N] ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Cancelled.")
        return

    d["rewards"] = r
    np.savez_compressed(path, **d)
    print(f"✓ Saved {path}: {old} → {new}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="episode .npz file OR a directory of episodes")
    p.add_argument("--flip", action="store_true",
                   help="flip the current label (success↔failure)")
    p.add_argument("--to", choices=["success", "fail", "failure"],
                   default=None,
                   help="force the label to this value")
    p.add_argument("--list", action="store_true",
                   help="list labels of all episodes under a directory (no edit)")
    args = p.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "episode_*.npz")))
        if not files:
            print(f"No episode_*.npz under {args.path}")
            sys.exit(1)
        if args.list or (not args.flip and args.to is None):
            n_s = n_f = 0
            for f in files:
                d = np.load(f)
                lbl = label_of(d["rewards"])
                tag = "✅" if lbl == "SUCCESS" else "❌"
                print(f"  {tag}  {os.path.basename(f)}  ({lbl}, T={len(d['rewards'])})")
                if lbl == "SUCCESS": n_s += 1
                else: n_f += 1
            print(f"\nTotal: {n_s} success / {n_f} failure (= {100*n_s/(n_s+n_f):.0f}%)")
            return
        # bulk edit not supported on purpose; iterate one by one with confirmation
        for f in files:
            relabel(f, want=args.to, flip=args.flip)
    else:
        relabel(args.path, want=args.to, flip=args.flip)


if __name__ == "__main__":
    main()
