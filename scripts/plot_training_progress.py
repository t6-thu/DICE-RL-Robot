#!/usr/bin/env python3
"""Plot YAM RL training progress.

Reads:
  * episode_*.npz under ONLINE_DATA_DIR    → per-episode success / length / time
  * checkpoint_*.pt under RL_CKPT_DIR      → training step boundaries
  * learner.log (optional) under RL_CKPT_DIR → loss / expert-ratio curves

Produces a multi-panel figure with:
  (1) Per-episode success + rolling success rate
  (2) Episode length (action chunks)
  (3) Deployed actor step
  (4) Cumulative successes
  (5) Critic loss curve (across all rounds)
  (6) Actor loss curve
  (7) Expert ratio anneal

Usage:
    python scripts/plot_training_progress.py
    python scripts/plot_training_progress.py --log /path/to/learner.log
    python scripts/plot_training_progress.py --window 10 --no-show
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from dice_rl.config.yam_rl_config import ONLINE_DATA_DIR, RL_CKPT_DIR, TRAINING


# ----- regex for parsing learner log -----
_RE_STEP   = re.compile(r"\[step\s+(\d+)/\s*(\d+)\]\s+critic_loss=([-\d.]+)\s+"
                        r"actor_loss=([-\d.]+)\s+expert_ratio=([-\d.]+)")
_RE_ROUND  = re.compile(r"Training round \(episode (\d+)\): expected=(\d+) done=(\d+)")


def load_episode_meta(online_dir: str):
    paths = sorted(glob.glob(os.path.join(online_dir, "episode_*.npz")))
    out = []
    for i, p in enumerate(paths):
        d = np.load(p)
        r = d.get("rewards", np.zeros(1, dtype=np.float32))
        T = int(len(d["states"])) if "states" in d.files else int(len(r))
        success = bool(r[-1] > 0.5) if len(r) > 0 else False
        out.append({"idx": i + 1, "name": os.path.basename(p),
                    "success": success, "T": T,
                    "mtime": os.path.getmtime(p)})
    return out


def load_checkpoint_meta(ckpt_dir: str):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pt")))
    out = []
    for p in paths:
        try:
            step = int(os.path.basename(p).split("_")[1].split(".")[0])
        except Exception:
            continue
        out.append({"step": step, "mtime": os.path.getmtime(p)})
    return sorted(out, key=lambda x: x["step"])


def parse_learner_log(log_path: str):
    """Return dict with parallel arrays: step, critic, actor, expert_ratio,
    plus list of round-boundary episodes."""
    steps_global, critic, actor, expert, round_eps = [], [], [], [], []
    if not os.path.isfile(log_path):
        return None
    current_round_idx = 0   # 0-indexed
    last_local_step   = -1
    with open(log_path) as f:
        for line in f:
            m = _RE_ROUND.search(line)
            if m:
                round_eps.append(int(m.group(1)))
                current_round_idx = len(round_eps) - 1
                last_local_step = -1
                continue
            m = _RE_STEP.search(line)
            if m:
                local_step = int(m.group(1))
                grad_per_round = int(m.group(2))
                # if step number resets to 0 → new round (safety)
                if local_step < last_local_step:
                    current_round_idx += 1
                last_local_step = local_step
                global_step = current_round_idx * grad_per_round + local_step
                steps_global.append(global_step)
                critic.append(float(m.group(3)))
                actor .append(float(m.group(4)))
                expert.append(float(m.group(5)))
    if not steps_global:
        return None
    return {
        "step":         np.asarray(steps_global),
        "critic":       np.asarray(critic),
        "actor":        np.asarray(actor),
        "expert_ratio": np.asarray(expert),
        "round_eps":    round_eps,
    }


def deployed_step_per_episode(eps, ckpts):
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
    if w <= 1: return x
    out = np.empty_like(x)
    csum = np.concatenate([np.zeros(1), np.cumsum(x)])
    for i in range(len(x)):
        lo = max(0, i - (w - 1))
        out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--online-dir", default=ONLINE_DATA_DIR)
    p.add_argument("--ckpt-dir",   default=RL_CKPT_DIR)
    p.add_argument("--log",        default=None,
                   help="learner.log path (default = <ckpt-dir>/learner.log)")
    p.add_argument("--save",       default=None,
                   help="output PNG (default = <ckpt-dir>/training_progress.png)")
    p.add_argument("--window",     type=int, default=10)
    p.add_argument("--no-show",    action="store_true")
    args = p.parse_args()

    eps   = load_episode_meta(args.online_dir)
    ckpts = load_checkpoint_meta(args.ckpt_dir)
    log_path = args.log or os.path.join(args.ckpt_dir, "learner.log")
    log_data = parse_learner_log(log_path)

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

    print(f"Episodes loaded: {n_ep}  from {args.online_dir}")
    print(f"Checkpoints:     {len(ckpts)}  from {args.ckpt_dir}")
    print(f"Overall success: {int(successes.sum())}/{n_ep} ({100*successes.mean():.1f}%)")
    print(f"Mean episode length: {T_per_ep.mean():.1f} chunks "
          f"(min={T_per_ep.min()}, max={T_per_ep.max()})")
    if log_data is not None:
        print(f"Log parsed:  {len(log_data['step'])} gradient-step records, "
              f"{len(log_data['round_eps'])} training rounds")
    else:
        print(f"(no log file at {log_path} — loss curves will be skipped)")

    warmup = TRAINING.get("num_episodes_before_first_training", 20)

    # ---- Plot ----
    if args.no_show:
        matplotlib.use("Agg")
    has_log = log_data is not None
    n_rows  = 7 if has_log else 4
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 2.0 * n_rows + 1),
                             sharex=False)
    fig.suptitle(
        f"YAM RL training progress  ({n_ep} episodes, "
        f"{len(ckpts)} checkpoints, "
        f"{len(log_data['step']) if has_log else 0} loss records)",
        fontsize=13, fontweight="bold")

    # checkpoint-vs-episode mapping
    ep_at_ckpt = []
    for c in ckpts:
        for e in eps:
            if e["mtime"] >= c["mtime"]:
                ep_at_ckpt.append((c["step"], e["idx"])); break
        else:
            ep_at_ckpt.append((c["step"], n_ep))

    def draw_ckpt_lines(ax, label_top=False):
        for step, ep_idx in ep_at_ckpt:
            ax.axvline(ep_idx, color="0.7", ls=":", lw=0.7, zorder=0)
            if label_top:
                ax.text(ep_idx, ax.get_ylim()[1], f"step={step}",
                        rotation=90, fontsize=7, color="0.4",
                        ha="right", va="top")
        ax.axvline(warmup, color="orange", ls="--", lw=1.0,
                   label=f"warmup ends ({warmup} eps)" if label_top else None)

    # (1) Per-episode success
    ax = axes[0]
    succ_idx = idxs[successes > 0.5]; fail_idx = idxs[successes < 0.5]
    ax.scatter(succ_idx, np.ones_like(succ_idx),  color="green", s=22, zorder=3, label="success")
    ax.scatter(fail_idx, np.zeros_like(fail_idx), color="red",   s=22, zorder=3, label="failure")
    ax.plot(idxs, roll, color="blue", lw=2,
            label=f"rolling success (w={args.window})")
    ax.set_ylim(-0.1, 1.1); ax.set_ylabel("success")
    ax.set_title("(1) Per-episode outcome + rolling success rate")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
    draw_ckpt_lines(ax, label_top=True)

    # (2) Episode length
    ax = axes[1]
    ax.plot(idxs, T_per_ep, "o-", color="purple", ms=3, lw=0.8)
    ax.set_ylabel("length (chunks)")
    ax.set_title("(2) Episode length")
    ax.grid(alpha=0.3); draw_ckpt_lines(ax)

    # (3) Deployed actor step
    ax = axes[2]
    ax.step(idxs, deploy_step, where="post", color="teal", lw=1.5)
    ax.fill_between(idxs, 0, deploy_step, step="post", color="teal", alpha=0.15)
    ax.set_ylabel("deployed step")
    ax.set_title("(3) Actor checkpoint deployed at each episode")
    ax.grid(alpha=0.3); draw_ckpt_lines(ax)

    # (4) Cumulative successes
    ax = axes[3]
    ax.plot(idxs, cum_succ, color="darkgreen", lw=2)
    ax.set_ylabel("cum. successes")
    ax.set_title("(4) Cumulative success count")
    ax.set_xlabel("episode #")
    ax.grid(alpha=0.3); draw_ckpt_lines(ax)

    if has_log:
        steps   = log_data["step"]
        critic  = log_data["critic"]
        actor_l = log_data["actor"]
        expert  = log_data["expert_ratio"]
        warmup_grad_steps = TRAINING.get("q_filtering_warmup_steps", 25000) \
            if False else 25000  # codebase value

        # (5) Critic loss
        ax = axes[4]
        ax.plot(steps, critic, color="crimson", lw=1.0)
        ax.set_ylabel("critic_loss")
        ax.set_title("(5) Critic loss vs gradient step")
        ax.axvline(warmup_grad_steps, color="orange", ls="--", lw=1.0,
                   label=f"BC filter activates (step {warmup_grad_steps})")
        ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

        # (6) Actor loss
        ax = axes[5]
        ax.plot(steps, actor_l, color="navy", lw=1.0)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_ylabel("actor_loss")
        ax.set_title("(6) Actor loss vs gradient step")
        ax.axvline(warmup_grad_steps, color="orange", ls="--", lw=1.0)
        ax.grid(alpha=0.3)

        # (7) Expert ratio
        ax = axes[6]
        ax.plot(steps, expert, color="darkorange", lw=1.5)
        ax.set_ylabel("expert_ratio")
        ax.set_title("(7) Expert ratio anneal (0.7 → 0.2 over 30000 steps)")
        ax.set_xlabel("gradient step (cumulative)")
        ax.axhline(0.2, color="gray", ls=":", lw=0.7)
        ax.axhline(0.7, color="gray", ls=":", lw=0.7)
        ax.axvline(warmup_grad_steps, color="orange", ls="--", lw=1.0)
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    out_path = args.save or os.path.join(args.ckpt_dir, "training_progress.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved figure → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
