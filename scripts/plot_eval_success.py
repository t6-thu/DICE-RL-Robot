#!/usr/bin/env python3
"""Plot eval success rate per checkpoint.

Reads <ONLINE_DATA_DIR>/eval/<label>/episode_*.npz, computes
   s/n  using rewards[-1] > 0.5
for each label, maps the label to the training episode # at which the ckpt was
saved, and draws a single line.

Defaults to plotting ckpts at episodes 30 40 50 60 70 (skips bc_only and the
first ckpt). Override via --eps or --include.

Usage:
    . ./prepare.sh
    python scripts/plot_eval_success.py                       # ep 30..70
    python scripts/plot_eval_success.py --eps 0 20 30 40 50 60 70 80
    python scripts/plot_eval_success.py --include bc_only checkpoint_006000
    python scripts/plot_eval_success.py --run other_run_name
    python scripts/plot_eval_success.py --out /tmp/eval.png
"""
from __future__ import annotations
import argparse, glob, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dice_rl.config.yam_rl_config import RUN_NAME, ONLINE_DATA_DIR, TRAINING


def label_to_episode(label: str) -> int | None:
    """ckpt label → training episode # at which that ckpt was saved.
    Schedule: warmup of W eps, then ckpt every B new eps.
      bc_only        → 0
      checkpoint_NNNNNN → W + (round_index - 1) * B,
        where round_index = (NNNNNN - first_round_steps) / subsequent_round_steps + 1
    Here first round = `gradient_steps` (=2000) and subsequent = gradient_steps//2 (=1000)."""
    W = TRAINING["num_episodes_before_first_training"]
    B = TRAINING["update_every_x_episode"]
    GS = TRAINING["gradient_steps"]
    SUB = GS // 2

    if label == "bc_only":
        return 0
    if label.startswith("checkpoint_"):
        try:
            gs = int(label.split("_")[1])
        except ValueError:
            return None
        rounds_after_first = (gs - GS) // SUB
        return W + rounds_after_first * B
    return None


def succ_of(d: str) -> tuple[int, int]:
    files = sorted(glob.glob(os.path.join(d, "episode_*.npz")))
    if not files: return 0, 0
    s = sum(int(np.load(f)["rewards"][-1] > 0.5) for f in files)
    return s, len(files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=RUN_NAME, help="run name (default: current)")
    p.add_argument("--root", default=None,
                   help="eval dir root (default: <ONLINE_DATA_DIR>/eval)")
    p.add_argument("--eps", type=int, nargs="+", default=[30, 40, 50, 60, 70],
                   help="which training-episode #s to plot (default: 30 40 50 60 70)")
    p.add_argument("--include", type=str, nargs="+", default=None,
                   help="explicit eval-label whitelist; overrides --eps")
    p.add_argument("--out", default=None,
                   help="output png path (default: scripts/eval_success_<run>.png)")
    args = p.parse_args()

    root = args.root or os.path.join(
        ONLINE_DATA_DIR if args.run == RUN_NAME
                       else os.path.join(os.path.dirname(ONLINE_DATA_DIR),
                                          f"yam_rl_rollouts_{args.run}"),
        "eval",
    )
    if not os.path.isdir(root):
        print(f"eval root not found: {root}", file=sys.stderr); sys.exit(1)
    out_png = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"eval_success_{args.run}.png",
    )

    # Scan available labels.
    available = sorted(d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d)))
    print(f"eval dirs under {root}:")
    rows_all = []
    for lab in available:
        ep = label_to_episode(lab)
        s, n = succ_of(os.path.join(root, lab))
        rate = (s / n * 100.0) if n else 0.0
        marker = "  "
        rows_all.append((ep, lab, s, n, rate))
        print(f"  {lab:<25s}  ep={ep}  s/n={s}/{n}  rate={rate:.1f}%")

    if args.include is not None:
        wanted = set(args.include)
        rows = [(ep, l, s, n, r) for ep, l, s, n, r in rows_all if l in wanted]
    else:
        wanted_eps = set(args.eps)
        rows = [(ep, l, s, n, r) for ep, l, s, n, r in rows_all if ep in wanted_eps]

    rows = [r for r in rows if r[0] is not None and r[3] > 0]
    rows.sort()
    if not rows:
        print("nothing to plot. did the chosen eps/labels exist?", file=sys.stderr); sys.exit(1)

    xs = [ep for ep, _, _, _, _ in rows]
    ys = [r  for _, _, _, _, r in rows]
    ns = [n  for _, _, _, n, _ in rows]

    print(f"\nPLOTTING:")
    for ep, lab, s, n, r in rows:
        print(f"  ep={ep:>3d}  {lab:<22s}  {s:>2d}/{n:<3d}  {r:5.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, "o-", color="#1f77b4", lw=2.5, ms=10)
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"{y:.0f}%\n(n={n})", (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, color="#1f77b4")

    ax.set_xlabel("training episode # at ckpt save")
    ax.set_ylabel("eval success rate (%)")
    ax.set_title(f"{args.run}  —  ckpt eval success rate")
    ax.set_xticks(xs)
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(out_png, dpi=130)
    print(f"\nsaved → {out_png}")


if __name__ == "__main__":
    main()
