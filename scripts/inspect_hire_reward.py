#!/usr/bin/env python3
"""Visualize HiRE PBRS reward & potential curves for online success/failure eps.

Reads the same run's online episodes + offline expert npz + curation, builds the
HiRE shaper with the run's hyper-parameters, then for a few success/failure
episodes:
  - Computes Φ(s_t) per chunk-image
  - Computes shaped reward r̃_t = sparse[t+1] + γ·Φ_{t+1} − Φ_t   (length T-1)
  - Plots Φ, dense PBRS, full shaped reward, and cumulative shaped reward
  - Prints summary stats (min/max/mean of Φ and r̃; rough Q-target estimate)

Usage:
    . ./prepare.sh
    python scripts/inspect_hire_reward.py
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
import torch

from dice_rl.config.yam_rl_config import (
    EXPERT_NPZ, ONLINE_DATA_DIR, HIRE_EXPERT_CURATION_PATH, TRAINING,
)
from dice_rl.reward.hire_shaper import DinoV2Encoder, HireRewardShaper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=ONLINE_DATA_DIR,
                    help="online episode dir; default = current RUN_NAME")
    ap.add_argument("--num-each", type=int, default=4,
                    help="how many success / failure eps to plot")
    ap.add_argument("--out", default="hire_reward_curves.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[inspect] using run dir: {args.run_dir}")
    print(f"[inspect] device: {device}")

    # Build HiRE shaper with the current TRAINING config
    enc = DinoV2Encoder(device=device)
    shaper = HireRewardShaper(
        encoder=enc, cameras=("rgb_0", "rgb_1"),
        reward_weight=TRAINING["hire_reward_weight"],
        contrastive_lambda=TRAINING["hire_contrastive_lambda"],
        logsumexp_beta_pos=TRAINING["hire_logsumexp_beta_pos"],
        logsumexp_beta_neg=TRAINING["hire_logsumexp_beta_neg"],
        gamma_pbrs=TRAINING["hire_gamma_pbrs"],
        sample_K=TRAINING["hire_sample_K"],
        online_success_frames=TRAINING["hire_online_success_frames"],
        online_failure_frames=TRAINING["hire_online_failure_frames"],
        expert_frame_stride=TRAINING["hire_expert_frame_stride"],
        max_pos_buffer_size=TRAINING["hire_max_pos_buffer_size"],
        max_neg_buffer_size=TRAINING["hire_max_neg_buffer_size"],
        online_pos_ratio=TRAINING["hire_online_pos_ratio"],
    )

    # 1) Build pos_buffer_expert from curated train.npz
    shaper.build_from_expert_npz(EXPERT_NPZ, curation_path=HIRE_EXPERT_CURATION_PATH)

    # 2) Build pos_buffer_online and neg_buffer from online episodes
    eps = sorted(glob.glob(os.path.join(args.run_dir, "episode_*.npz")))
    succ_paths, fail_paths = [], []
    for p in eps:
        z = np.load(p)
        if z["rewards"][-1] > 0.5: succ_paths.append(p)
        else:                     fail_paths.append(p)
    print(f"[inspect] found {len(succ_paths)} success / {len(fail_paths)} failure episodes")

    # Reserve last N of each for evaluation (so they're NOT in the buffers)
    N = args.num_each
    eval_succ = succ_paths[-N:] if len(succ_paths) >= N else succ_paths
    eval_fail = fail_paths[-N:] if len(fail_paths) >= N else fail_paths
    pool_succ = [p for p in succ_paths if p not in eval_succ]
    pool_fail = [p for p in fail_paths if p not in eval_fail]

    for p in pool_succ:
        z = np.load(p); shaper.add_episode_to_buffer(z["images"], success=True)
    for p in pool_fail:
        z = np.load(p); shaper.add_episode_to_buffer(z["images"], success=False)

    print(f"[inspect] buffers built — sampling {N} success + {N} failure for eval")
    for cam in shaper.cameras:
        pe = shaper.pos_buffer_expert.get(cam)
        po = shaper.pos_buffer_online.get(cam)
        ng = shaper.neg_buffer.get(cam)
        print(f"  {cam}: pos_expert={tuple(pe.shape) if pe is not None else None}  "
              f"pos_online={tuple(po.shape) if po is not None else None}  "
              f"neg={tuple(ng.shape) if ng is not None else None}")

    # 3) For each eval episode, compute Φ and shaped reward
    rows = []
    for is_succ, path_list in [(True, eval_succ), (False, eval_fail)]:
        for p in path_list:
            z = np.load(p)
            images = z["images"]                  # (T, 6, H, W) float32 [0,1]
            T = images.shape[0]
            r_sparse = np.asarray(z["rewards"], dtype=np.float32)
            phi = shaper._compute_potential(images)              # (T,)
            r_shaped = shaper.shape_rewards(r_sparse, images)    # (T-1,)
            dense = phi[1:] * shaper.gamma_pbrs - phi[:-1]       # (T-1,)
            sparse_for_trans = r_sparse[1:]                       # (T-1,)
            cum = np.cumsum(r_shaped)
            # Crude undiscounted return estimate (for rough Q-target sense)
            ret = r_shaped.sum()
            rows.append(dict(path=os.path.basename(p), succ=is_succ,
                             phi=phi, dense=dense, sparse=sparse_for_trans,
                             shaped=r_shaped, cum=cum, ret=ret))

    # 4) Print stats
    print("\n=== Per-episode summary ===")
    print(f"{'episode':30s}  {'succ':4s}  {'Φ_max':>7s} {'Φ_min':>7s}  "
          f"{'r̃_max':>7s} {'r̃_min':>7s}  {'Σr̃':>7s}")
    for r in rows:
        print(f"{r['path']:30s}  {'S' if r['succ'] else 'F':>4s}  "
              f"{r['phi'].max():>7.3f} {r['phi'].min():>7.3f}  "
              f"{r['shaped'].max():>7.3f} {r['shaped'].min():>7.3f}  "
              f"{r['ret']:>7.3f}")

    succ_rets = [r["ret"] for r in rows if r["succ"]]
    fail_rets = [r["ret"] for r in rows if not r["succ"]]
    if succ_rets and fail_rets:
        print(f"\nMean Σr̃ — success {np.mean(succ_rets):+.3f} | failure {np.mean(fail_rets):+.3f}  "
              f"Δ={np.mean(succ_rets)-np.mean(fail_rets):+.3f}")

    # Rough Q-target magnitude (geometric series of mean per-step r̃ over τ ≈ episode length)
    if succ_rets:
        mean_T = float(np.mean([len(r["shaped"]) for r in rows if r["succ"]]))
        per_step = float(np.mean([r["ret"] / max(len(r["shaped"]), 1) for r in rows if r["succ"]]))
        γ = shaper.gamma_pbrs
        q_geom = per_step * (1 - γ**mean_T) / (1 - γ)
        print(f"\nRough Q-target estimate for success: per-step r̃ ≈ {per_step:+.4f}, "
              f"T ≈ {mean_T:.1f}, geometric sum ≈ {q_geom:+.3f}")

    # 5) Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    succ_color, fail_color = "tab:green", "tab:red"
    titles = ["Φ(s_t) along trajectory",
              "Dense PBRS term:  γ·Φ_{t+1} − Φ_t",
              "Shaped reward r̃_t = sparse[t+1] + dense",
              "Cumulative Σ_{≤t} r̃"]
    keys   = ["phi", "dense", "shaped", "cum"]
    for ax, key, title in zip(axes.flat, keys, titles):
        for r in rows:
            x = np.arange(len(r[key]))
            ax.plot(x, r[key],
                    color=(succ_color if r["succ"] else fail_color),
                    alpha=0.6, lw=1.5,
                    label=("success" if r["succ"] else "failure") + f" {r['path'][-12:-4]}")
        ax.set_title(title)
        ax.set_xlabel("transition step t")
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.3)
    # de-duplicate legend on first plot
    handles, labels = axes[0,0].get_legend_handles_labels()
    seen = set(); ulabels, uhandles = [], []
    for h, l in zip(handles, labels):
        prefix = l.split()[0]
        if prefix not in seen:
            seen.add(prefix); uhandles.append(h); ulabels.append(prefix)
    axes[0,0].legend(uhandles, ulabels, loc="best")

    fig.suptitle(
        f"HiRE reward curves  "
        f"(λ={shaper.contrastive_lambda}, β_pos={shaper.logsumexp_beta_pos}, "
        f"β_neg={shaper.logsumexp_beta_neg}, online_pos_ratio={shaper.online_pos_ratio})",
        y=1.02)
    plt.tight_layout()
    out = args.out
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot → {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
