#!/usr/bin/env python3
"""Stricter control for the HiRE 5x-discrimination claim.

Addresses the methodological question:
  - Is "online-success" a within-trajectory comparison (trivially high sim)?
    NO: positive pool and eval frames are episode-level DISJOINT.
  - Is the success<->success closeness just "same-domain any-two-frames" noise?
    Tested via a 2x2 cross-class table + a within-domain RANDOM baseline.

Design (all episode-level disjoint):
  success eps:  first N -> succ_eval (last frame);   rest -> SUCC pool (all frames)
  failure eps:  first N -> fail_eval (last frame);   rest -> FAIL pool (last frames)
  RAND pool  :  random frames sampled across ALL non-eval eps (domain control)

We report sim_pos (logsumexp beta=10, patch-cosine) of each eval set against
each pool, exactly as HireRewardShaper._sim_to_targets does.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/bike/Documents/niu/DICE-RL-Robot")
from dice_rl.reward.hire_shaper import DinoV2Encoder  # noqa: E402

BPOS = 10.0


def l2(x):
    return torch.nn.functional.normalize(x, dim=-1)


def patch_cos_lse(cur_PD, tgt_KPD, beta=BPOS):
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
    ap.add_argument("--n_eval", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    enc = DinoV2Encoder(device=args.device)

    def enc_base(f6):
        f = f6.astype(np.float32)
        if f6.dtype == np.uint8:
            f /= 255.0
        return enc.encode(torch.from_numpy(f[:3]).unsqueeze(0))[0]

    paths = sorted(glob.glob(os.path.join(args.roll, "episode_*.npz")))
    succ_eval, fail_eval = [], []
    succ_pool, fail_pool, rand_pool = [], [], []
    ns = nf = 0
    succ_eval_eps, succ_pool_eps = [], []
    for p in paths:
        d = np.load(p)
        img = d["images"]
        suc = d["rewards"][-1] > 0.5
        if suc:
            if ns < args.n_eval:
                succ_eval.append(enc_base(img[-1]))
                succ_eval_eps.append(os.path.basename(p))
                ns += 1
            else:
                succ_pool_eps.append(os.path.basename(p))
                for fi in range(len(img)):
                    succ_pool.append(enc_base(img[fi]))
                # contribute a couple random frames to the domain control
                for fi in rng.choice(len(img), size=min(3, len(img)), replace=False):
                    rand_pool.append(enc_base(img[int(fi)]))
        else:
            if nf < args.n_eval:
                fail_eval.append(enc_base(img[-1]))
                nf += 1
            else:
                fail_pool.append(enc_base(img[-1]))
                for fi in rng.choice(len(img), size=min(3, len(img)), replace=False):
                    rand_pool.append(enc_base(img[int(fi)]))

    SUCC = torch.stack(succ_pool)
    FAIL = torch.stack(fail_pool)
    RAND = torch.stack(rand_pool)

    # sanity: no episode overlap between eval and pool
    overlap = set(succ_eval_eps) & set(succ_pool_eps)
    print(f"episode-disjoint check (succ eval vs succ pool overlap): {overlap or 'NONE'}")
    print(f"pools: succ={len(SUCC)} fail={len(FAIL)} rand={len(RAND)}  "
          f"eval(succ/fail)={len(succ_eval)}/{len(fail_eval)}\n")

    def avg(evals, pool):
        return np.array([patch_cos_lse(e, pool) for e in evals])

    s_to_S = avg(succ_eval, SUCC)
    s_to_F = avg(succ_eval, FAIL)
    s_to_R = avg(succ_eval, RAND)
    f_to_S = avg(fail_eval, SUCC)
    f_to_F = avg(fail_eval, FAIL)
    f_to_R = avg(fail_eval, RAND)

    print("=== 2x2 cross-class sim_pos (mean ± std) ===")
    hdr = "eval \\ pool"
    print(f"{hdr:<14}{'SUCC-pos':>14}{'FAIL-pos':>14}{'RAND(domain)':>16}")
    print(f"{'succ-eval':<14}{s_to_S.mean():>9.4f}±{s_to_S.std():.3f}"
          f"{s_to_F.mean():>9.4f}±{s_to_F.std():.3f}{s_to_R.mean():>11.4f}±{s_to_R.std():.3f}")
    print(f"{'fail-eval':<14}{f_to_S.mean():>9.4f}±{f_to_S.std():.3f}"
          f"{f_to_F.mean():>9.4f}±{f_to_F.std():.3f}{f_to_R.mean():>11.4f}±{f_to_R.std():.3f}")

    print("\n=== discrimination Δ = mean(succ-eval) − mean(fail-eval) ===")
    dS = s_to_S.mean() - f_to_S.mean()
    dR = s_to_R.mean() - f_to_R.mean()
    # within-condition noise scale: std of the per-frame sim_pos
    noise = 0.5 * (s_to_S.std() + f_to_S.std())
    print(f"Δ vs SUCC-pos (online-success positives): {dS:+.4f}   ({dS/ (noise+1e-9):.2f}σ)")
    print(f"Δ vs RAND-pos (same-domain control)     : {dR:+.4f}   <- if ~Δ_SUCC, claim is trivial")
    print(f"per-frame noise scale (std)             : {noise:.4f}")

    print("\n=== preference test: does each eval class prefer its OWN-class pool? ===")
    print(f"succ-eval prefers SUCC over FAIL by: {(s_to_S - s_to_F).mean():+.4f}")
    print(f"fail-eval prefers FAIL over SUCC by: {(f_to_F - f_to_S).mean():+.4f}")


if __name__ == "__main__":
    main()
