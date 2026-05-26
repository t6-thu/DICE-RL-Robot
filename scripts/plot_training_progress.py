#!/usr/bin/env python3
"""Plot YAM RL training progress from on-disk artifacts.

Reads:
  * episode_*.npz under ONLINE_DATA_DIR    → per-episode success / length / time
  * checkpoint_*.pt under RL_CKPT_DIR      → training step boundaries

Produces a multi-panel figure showing:
  (1) Per-episode success (binary) + rolling success rate
  (2) Episode length (action chunks per episode)
  (3) Deployed actor step over time (step function)
  (4) Cumulative success count

Usage:
    python scripts/plot_training_progress.py                    # shows + saves
    python scripts/plot_training_progress.py --save out.png     # custom path
    python scripts/plot_training_progress.py --window 10        # rolling-mean window
    python scripts/plot_training_progress.py --no-show          # save only, no GUI
"""
import argparse
import glob
import os
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from dice_rl.config.yam_rl_config import ONLINE_DATA_DIR, RL_CKPT_DIR, TRAINING


def load_episode_meta(online_dir: str):
    """Return list of dicts: {idx, name, success, T, mtime}."""
    paths = sorted(glob.glob(os.path.join(online_dir, "episode_*.npz")))
    out = []
    for i, p in enumerate(paths):
        d = np.load(p)
        r = d.get("rewards", np.zeros(1, dtype=np.float32))
        T = int(len(d["states"])) if "states" in d.files else int(len(r))
        success = bool(r[-1] > 0.5) if len(r) > 0 else False
        out.append({
            "idx":     i + 1,
            "name":    os.path.basename(p),
            "success": success,
            "T":       T,
            "mtime":   os.path.getmtime(p),
        })
    return out


def load_checkpoint_meta(ckpt_dir: str):
    """Return list of dicts: {step, mtime} from checkpoint_*.pt filenames."""
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pt")))
    out = []
    for p in paths:
        try:
            step = int(os.path.basename(p).split("_")[1].split(".")[0])
        except Exception:
            continue
        out.append({"step": step, "mtime": os.path.getmtime(p)})
    return sorted(out, key=lambda x: x["step"])


def deployed_step_per_episode(eps, ckpts):
    """For each episode, return the actor step that was deployed when it was collected.

    Logic: the latest checkpoint whose mtime <= episode mtime is the one
    `latest_actor.pt` pointed to.
    """
    steps = []
    for e in eps:
        s = 0
        for c in ckpts:
            if c["mtime"] <= e["mtime"]:
                s = c["step"]
            else:
                break
        steps.append(s)
    return steps


def rolling_mean(x, w):
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x
    pad = w - 1
    csum = np.concatenate([np.zeros(1), np.cumsum(x)])
    # window i covers x[max(0,i-w+1):i+1]
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - pad)
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--online-dir", default=ONLINE_DATA_DIR)
    p.add_argument("--ckpt-dir",   default=RL_CKPT_DIR)
    p.add_argument("--save",       default=None, help="output PNG path (default = ckpt_dir/training_progress.png)")
    p.add_argument("--window",     type=int, default=10, help="rolling-mean window for success rate")
    p.add_argument("--no-show",    action="store_true")
    args = p.parse_args()

    eps   = load_episode_meta(args.online_dir)
    ckpts = load_checkpoint_meta(args.ckpt_dir)

    if not eps:
        print(f"No episodes under {args.online_dir}")
        return

    n_ep        = len(eps)
    idxs        = np.array([e["idx"]     for e in eps])
    successes   = np.array([e["success"] for e in eps], dtype=float)
    T_per_ep    = np.array([e["T"]       for e in eps])
    deploy_step = np.array(deployed_step_per_episode(eps, ckpts))
    roll        = rolling_mean(successes, args.window)
    cum_succ    = np.cumsum(successes)

    print(f"Loaded {n_ep} episodes from {args.online_dir}")
    print(f"Loaded {len(ckpts)} checkpoints from {args.ckpt_dir}")
    print(f"Overall success: {int(successes.sum())}/{n_ep} "
          f"({100*successes.mean():.1f}%)")
    print(f"Mean episode length: {T_per_ep.mean():.1f} chunks "
          f"(min={T_per_ep.min()}, max={T_per_ep.max()})")

    warmup = TRAINING.get("num_episodes_before_first_training", 20)

    # ---- Plot ----
    if args.no_show:
        matplotlib.use("Agg")
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(
        f"YAM RL training progress  ({n_ep} episodes, {len(ckpts)} checkpoints)",
        fontsize=13, fontweight="bold")

    # Step-boundary helper: episode # at which each ckpt was saved
    # (closest episode mtime ≥ ckpt mtime)
    ep_at_ckpt = []
    for c in ckpts:
        found = None
        for e in eps:
            if e["mtime"] >= c["mtime"]:
                found = e["idx"]; break
        ep_at_ckpt.append((c["step"], found if found is not None else n_ep))

    def draw_ckpt_lines(ax, label_top=False):
        for step, ep_idx in ep_at_ckpt:
            ax.axvline(ep_idx, color="0.7", linestyle=":", linewidth=0.7, zorder=0)
            if label_top:
                ax.text(ep_idx, ax.get_ylim()[1], f"step={step}",
                        rotation=90, fontsize=7, color="0.4",
                        ha="right", va="top")
        ax.axvline(warmup, color="orange", linestyle="--", linewidth=1.0,
                   label=f"warmup ends ({warmup} eps)" if label_top else None)

    # (1) Per-episode success + rolling mean
    ax = axes[0]
    succ_idx = idxs[successes > 0.5]
    fail_idx = idxs[successes < 0.5]
    ax.scatter(succ_idx, np.ones_like(succ_idx),
               color="green", s=30, label="success", zorder=3)
    ax.scatter(fail_idx, np.zeros_like(fail_idx),
               color="red",   s=30, label="failure", zorder=3)
    ax.plot(idxs, roll, color="blue", linewidth=2,
            label=f"rolling success rate (window={args.window})")
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("success rate")
    ax.set_title("(1) Per-episode outcome + rolling success rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    draw_ckpt_lines(ax, label_top=True)

    # (2) Episode length
    ax = axes[1]
    ax.plot(idxs, T_per_ep, "o-", color="purple", markersize=3, linewidth=0.8)
    ax.set_ylabel("episode length (chunks)")
    ax.set_title("(2) Episode length (action chunks per episode)")
    ax.grid(alpha=0.3)
    draw_ckpt_lines(ax)

    # (3) Deployed actor step (step function)
    ax = axes[2]
    ax.step(idxs, deploy_step, where="post", color="teal", linewidth=1.5)
    ax.fill_between(idxs, 0, deploy_step, step="post", color="teal", alpha=0.15)
    ax.set_ylabel("deployed step")
    ax.set_title("(3) Actor checkpoint step deployed at each episode")
    ax.grid(alpha=0.3)
    draw_ckpt_lines(ax)

    # (4) Cumulative successes
    ax = axes[3]
    ax.plot(idxs, cum_succ, color="darkgreen", linewidth=2)
    ax.set_xlabel("episode #")
    ax.set_ylabel("cumulative successes")
    ax.set_title("(4) Cumulative success count")
    ax.grid(alpha=0.3)
    draw_ckpt_lines(ax)

    plt.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = args.save or os.path.join(args.ckpt_dir, "training_progress.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved figure → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
