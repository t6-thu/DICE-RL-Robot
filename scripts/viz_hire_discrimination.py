#!/usr/bin/env python3
"""Visualize the HiRE discrimination finding:

    Swapping the POSITIVE source (expert -> online-success) raises the
    success-vs-failure separation of sim_pos by ~5x.

Produces a multi-panel figure from REAL DINOv2 embeddings of the
yam_rl_rollouts_hire_v2 episodes + the curated expert demos.

Usage:
    python scripts/viz_hire_discrimination.py \
        --roll /home/bike/data/real_processed/yam_rl_rollouts_hire_v2 \
        --expert /home/bike/data/real_processed/yam_picknplace_arizonabottle_224/train.npz \
        --curation /home/bike/data/real_processed/yam_picknplace_arizonabottle_224/expert_curation.json \
        --out /home/bike/Documents/niu/hire_analysis/hire_discrimination.png
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/bike/Documents/niu/DICE-RL-Robot")
from dice_rl.reward.hire_shaper import DinoV2Encoder  # noqa: E402

BPOS, BNEG, LAM = 10.0, 1.0, 0.1


def l2(x):
    return torch.nn.functional.normalize(x, dim=-1)


def patch_cos_lse(cur_PD, tgt_KPD, beta):
    """Replicate HireRewardShaper._sim_to_targets for one current frame."""
    if tgt_KPD is None or len(tgt_KPD) == 0:
        return 0.0
    cur = l2(cur_PD.unsqueeze(0))
    tgt = l2(tgt_KPD)
    per_patch = torch.einsum("bpd,kpd->bkp", cur, tgt)
    per_pair = per_patch.mean(-1)
    return float((torch.logsumexp(beta * per_pair, dim=-1) / beta)[0].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", default="/home/bike/data/real_processed/yam_rl_rollouts_hire_v2")
    ap.add_argument("--expert", default="/home/bike/data/real_processed/yam_picknplace_arizonabottle_224/train.npz")
    ap.add_argument("--curation", default="/home/bike/data/real_processed/yam_picknplace_arizonabottle_224/expert_curation.json")
    ap.add_argument("--out", default="/home/bike/Documents/niu/hire_analysis/hire_discrimination_curated.png")
    ap.add_argument("--n_eval", type=int, default=20, help="held-out success/failure eval episodes")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    enc = DinoV2Encoder(device=args.device)

    def enc_base(f6):
        f = f6.astype(np.float32)
        if f6.dtype == np.uint8:
            f /= 255.0
        return enc.encode(torch.from_numpy(f[:3]).unsqueeze(0))[0]

    # ---- pools ----
    d_npz = np.load(args.expert)
    expert = d_npz["images"]

    # Apply curation JSON (same logic as HireRewardShaper.build_from_expert_npz)
    traj_lengths = d_npz["traj_lengths"].astype(int)
    include_eps = []
    if args.curation and os.path.isfile(args.curation):
        with open(args.curation) as f:
            cur_json = json.load(f)
        include_eps = sorted(set(int(x) for x in cur_json.get("include", [])))
        if include_eps:
            ep_starts = np.concatenate([[0], np.cumsum(traj_lengths)])
            idx_chunks = []
            for ep in include_eps:
                if 0 <= ep < len(traj_lengths):
                    idx_chunks.append(np.arange(int(ep_starts[ep]), int(ep_starts[ep + 1])))
            all_ex_idx = np.concatenate(idx_chunks)
            print(f"curation: using {len(include_eps)}/{len(traj_lengths)} expert episodes "
                  f"({len(all_ex_idx)} frames)")
        else:
            all_ex_idx = np.arange(len(expert))
            print("curation include list empty — using all expert frames")
    else:
        all_ex_idx = np.arange(len(expert))
        print("no curation file — using all expert frames")

    ex_sample = np.linspace(0, len(all_ex_idx) - 1, 80).astype(int)
    ex_idx = all_ex_idx[ex_sample]
    EX = torch.stack([enc_base(expert[i]) for i in ex_idx])

    paths = sorted(glob.glob(os.path.join(args.roll, "episode_*.npz")))
    succ_pos, neg, succ_eval, fail_eval = [], [], [], []
    ns = nf = 0
    for p in paths:
        d = np.load(p)
        img = d["images"]
        suc = d["rewards"][-1] > 0.5
        if suc:
            if ns < args.n_eval:
                succ_eval.append(enc_base(img[-1]))
                ns += 1
            else:
                for fi in range(len(img)):
                    succ_pos.append(enc_base(img[fi]))
        else:
            if nf < args.n_eval:
                fail_eval.append(enc_base(img[-1]))
                nf += 1
            else:
                neg.append(enc_base(img[-1]))
    SUCC = torch.stack(succ_pos)
    NEG = torch.stack(neg)
    print(f"pools: expert={len(EX)} online_succ_pos={len(SUCC)} neg={len(NEG)} "
          f"eval(succ/fail)={len(succ_eval)}/{len(fail_eval)}")

    # ---- compute sim_pos for each eval frame under each positive source ----
    def sims(pool):
        sp_s = np.array([patch_cos_lse(e, pool, BPOS) for e in succ_eval])
        sp_f = np.array([patch_cos_lse(e, pool, BPOS) for e in fail_eval])
        return sp_s, sp_f

    sp_s_ex, sp_f_ex = sims(EX)
    sp_s_on, sp_f_on = sims(SUCC)
    sn_s = np.array([patch_cos_lse(e, NEG, BNEG) for e in succ_eval])
    sn_f = np.array([patch_cos_lse(e, NEG, BNEG) for e in fail_eval])

    phi_s_ex = sp_s_ex - LAM * sn_s
    phi_f_ex = sp_f_ex - LAM * sn_f
    phi_s_on = sp_s_on - LAM * sn_s
    phi_f_on = sp_f_on - LAM * sn_f

    d_ex = sp_s_ex.mean() - sp_f_ex.mean()
    d_on = sp_s_on.mean() - sp_f_on.mean()
    print(f"sim_pos Δ: expert={d_ex:+.3f}  online-succ={d_on:+.3f}  ratio={d_on/max(d_ex,1e-6):.1f}x")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    C_S, C_F = "#2ca02c", "#d62728"

    def overlap_panel(a, s, f, title, dval):
        lo = min(s.min(), f.min())
        hi = max(s.max(), f.max())
        bins = np.linspace(lo, hi, 18)
        a.hist(f, bins=bins, alpha=0.55, color=C_F, label=f"failure (μ={f.mean():.3f})", density=True)
        a.hist(s, bins=bins, alpha=0.55, color=C_S, label=f"success (μ={s.mean():.3f})", density=True)
        a.axvline(f.mean(), color=C_F, ls="--", lw=2)
        a.axvline(s.mean(), color=C_S, ls="--", lw=2)
        a.set_title(f"{title}\nΔ(succ−fail) = {dval:+.3f}", fontweight="bold")
        a.set_xlabel("sim_pos  (logsumexp β=10 patch-cosine)")
        a.set_ylabel("density")
        a.legend(fontsize=9)

    overlap_panel(ax[0, 0], sp_s_ex, sp_f_ex, "Positives = EXPERT demos", d_ex)
    overlap_panel(ax[0, 1], sp_s_on, sp_f_on, "Positives = ONLINE-SUCCESS frames", d_on)

    # bar chart of Δ
    axb = ax[1, 0]
    bars = axb.bar(["expert", "online-success"], [d_ex, d_on],
                   color=["#7f7f7f", "#1f77b4"], width=0.55)
    axb.set_ylabel("Δ sim_pos (success − failure)")
    axb.set_title(f"Discrimination jumps {d_on/max(d_ex,1e-6):.1f}×  by swapping positive source",
                  fontweight="bold")
    for b, v in zip(bars, [d_ex, d_on]):
        axb.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:+.3f}",
                 ha="center", va="bottom", fontweight="bold")
    axb.axhline(0, color="k", lw=0.8)

    # per-frame paired scatter: sim_pos under expert vs online positives
    axs = ax[1, 1]
    axs.scatter(sp_f_ex, sp_f_on, c=C_F, label="failure", alpha=0.7, edgecolor="k", linewidth=0.3)
    axs.scatter(sp_s_ex, sp_s_on, c=C_S, label="success", alpha=0.7, edgecolor="k", linewidth=0.3)
    axs.set_xlabel("sim_pos vs EXPERT positives")
    axs.set_ylabel("sim_pos vs ONLINE-SUCCESS positives")
    axs.set_title("Same eval frames, two positive sources\n(vertical spread = added signal)",
                  fontweight="bold")
    axs.legend(fontsize=9)

    curation_note = (f"expert: {len(include_eps)}/{len(traj_lengths)} curated episodes"
                     if args.curation and os.path.isfile(args.curation) and include_eps
                     else "expert: all episodes (no curation)")
    fig.suptitle(
        "HiRE: why swapping the positive source recovers task signal "
        f"(hire_v2, real DINOv2)\nexpert↔online domain gap makes expert positives near-constant"
        f"  [{curation_note}]",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130)
    print(f"saved: {args.out}")

    # also dump raw numbers for reuse
    np.savez(os.path.splitext(args.out)[0] + "_data.npz",
             sp_s_ex=sp_s_ex, sp_f_ex=sp_f_ex, sp_s_on=sp_s_on, sp_f_on=sp_f_on,
             sn_s=sn_s, sn_f=sn_f, d_ex=d_ex, d_on=d_on)


if __name__ == "__main__":
    main()
