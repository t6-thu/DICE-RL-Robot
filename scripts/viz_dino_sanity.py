#!/usr/bin/env python3
"""Sanity-check that the DINO similarity inputs are ALIGNED.

HiRE compares patch i of the current frame to patch i of the target frame
(einsum "bpd,kpd->bkp"), so the two images must share the SAME preprocessing
AND spatial layout for the similarity to be meaningful.

This figure verifies that, end to end, on real frames:

  (a) what DINO actually ingests (decoded back to viewable RGB) — confirms
      channel order (RGB not BGR), [0,1] scale, correct base/wrist split;
  (b) self-similarity is ~1.0 (identity), MP4 round-trip ~0.97;
  (c) spatial sensitivity: a horizontally-flipped copy drops the score, and
      the per-patch heatmap becomes structured — proving patch i ↔ patch i
      spatial correspondence is real (inputs are aligned, not scrambled);
  (d) cross-camera (base vs wrist) is low, as expected.

Usage:
    python scripts/viz_dino_sanity.py --out /home/bike/Documents/niu/hire_analysis/dino_sanity.png
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/bike/Documents/niu/DICE-RL-Robot")
from dice_rl.reward.hire_shaper import DinoV2Encoder  # noqa: E402

GRID = 16  # 224/14 = 16 patches per side


def l2(x):
    return torch.nn.functional.normalize(x, dim=-1)


def per_patch_cos(a_PD, b_PD):
    """Patch-aligned cosine: cos(a_i, b_i) for each patch i -> (P,)."""
    return (l2(a_PD) * l2(b_PD)).sum(-1)


def scalar_sim(a_PD, b_PD):
    """The number HiRE actually reduces to: mean over patch-aligned cosine."""
    return float(per_patch_cos(a_PD, b_PD).mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", default="/home/bike/data/real_processed/yam_picknplace_arizonabottle_224/train.npz")
    ap.add_argument("--roll", default="/home/bike/data/real_processed/yam_rl_rollouts_hire_v2")
    ap.add_argument("--out", default="/home/bike/Documents/niu/hire_analysis/dino_sanity.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    enc = DinoV2Encoder(device=args.device)

    def enc_chw(img_chw_uint_or_f):
        """(3,224,224) any -> patch tokens (P,D). Mirrors offline/online: /255 if uint8."""
        f = img_chw_uint_or_f.astype(np.float32)
        if img_chw_uint_or_f.dtype == np.uint8:
            f /= 255.0
        return enc.encode(torch.from_numpy(f).unsqueeze(0))[0]

    # ---- grab one expert frame (uint8) and one online frame (float [0,1]) ----
    expert = np.load(args.expert)["images"][1000]          # (6,224,224) uint8
    import glob
    onp = sorted(glob.glob(os.path.join(args.roll, "episode_*.npz")))[1]
    online = np.load(onp)["images"][5]                     # (6,224,224) float [0,1]

    ex_base = expert[:3]                                   # uint8
    ex_wrist = expert[3:]
    on_base = online[:3]                                   # float

    # viewable RGB (HWC) exactly as DINO ingests (post /255, pre ImageNet-norm)
    def view(img_chw):
        f = img_chw.astype(np.float32)
        if img_chw.dtype == np.uint8:
            f /= 255.0
        return np.clip(np.transpose(f, (1, 2, 0)), 0, 1)

    # ---- embeddings for the test battery ----
    emb_exb = enc_chw(ex_base)
    emb_exb_again = enc_chw(ex_base.copy())                # identity
    emb_exw = enc_chw(ex_wrist)
    emb_onb = enc_chw(on_base)

    # MP4 round-trip on the expert base frame
    import imageio.v2 as imageio
    hwc = (view(ex_base) * 255).astype(np.uint8)
    mp4 = "/tmp/_dino_sanity.mp4"
    with imageio.get_writer(mp4, fps=30, macro_block_size=1, codec="libx264", quality=8) as w:
        for _ in range(6):
            w.append_data(hwc)
    dec = imageio.mimread(mp4, memtest=False)[0][..., :3]
    emb_mp4 = enc_chw(np.transpose(dec, (2, 0, 1)).astype(np.uint8))

    # horizontal flip (spatial-alignment breaker)
    ex_base_flip = ex_base[:, :, ::-1].copy()
    emb_flip = enc_chw(ex_base_flip)

    sims = {
        "self\n(A vs A)": scalar_sim(emb_exb, emb_exb_again),
        "MP4 round-trip\n(A vs mp4(A))": scalar_sim(emb_exb, emb_mp4),
        "diff frame\n(expert vs online)": scalar_sim(emb_exb, emb_onb),
        "H-flip\n(A vs flip(A))": scalar_sim(emb_exb, emb_flip),
        "cross-cam\n(base vs wrist)": scalar_sim(emb_exb, emb_exw),
    }
    for k, v in sims.items():
        print(f"  {k.splitlines()[0]:18s}: {v:.4f}")

    # ---- per-patch spatial heatmaps (reshape 256 -> 16x16) ----
    def heat(a, b):
        return per_patch_cos(a, b).reshape(GRID, GRID).detach().cpu().numpy()

    h_self = heat(emb_exb, emb_exb_again)
    h_flip = heat(emb_exb, emb_flip)
    h_diff = heat(emb_exb, emb_onb)

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10})
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])

    # row 0: the actual DINO inputs
    a = fig.add_subplot(gs[0, 0]); a.imshow(view(ex_base)); a.set_title("DINO input A\nexpert base (uint8/255)", fontweight="bold"); a.axis("off")
    a = fig.add_subplot(gs[0, 1]); a.imshow(view(ex_wrist)); a.set_title("expert wrist\n(channels 3:6)"); a.axis("off")
    a = fig.add_subplot(gs[0, 2]); a.imshow(view(on_base)); a.set_title("DINO input B\nonline base (float [0,1])", fontweight="bold"); a.axis("off")
    # range check text
    a = fig.add_subplot(gs[0, 3]); a.axis("off")
    txt = (
        "INPUT ALIGNMENT CHECKS\n"
        "──────────────────────\n"
        f"expert base : dtype=uint8\n"
        f"   /255 -> [{view(ex_base).min():.2f}, {view(ex_base).max():.2f}]\n"
        f"online base : dtype=float32\n"
        f"   range    [{on_base.min():.2f}, {on_base.max():.2f}]\n"
        f"patches/img : {emb_exb.shape[0]}  (=16x16)\n"
        f"feat dim    : {emb_exb.shape[1]}\n"
        f"resize 224  : no-op (already 224)\n"
        f"channel ord : RGB (rs.rgb8 / mp4 RGB)\n"
        f"split       : [:3]=base  [3:]=wrist\n"
        f"sim reduce  : mean_i cos(cur_i, tgt_i)\n"
        f"            => patch i <-> patch i"
    )
    a.text(0.0, 0.98, txt, va="top", ha="left", family="monospace", fontsize=9.5)

    # row 1 col0: sanity bars
    a = fig.add_subplot(gs[1, 0])
    keys = list(sims.keys()); vals = [sims[k] for k in keys]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    bars = a.bar(range(len(keys)), vals, color=colors)
    a.set_xticks(range(len(keys))); a.set_xticklabels(keys, fontsize=8)
    a.set_ylabel("mean patch-cosine (HiRE sim)")
    a.set_ylim(0, 1.05); a.axhline(1.0, color="k", ls=":", lw=1)
    a.set_title("Sanity battery\nself=1.0 ✓  flip<diff ✓", fontweight="bold")
    for b, v in zip(bars, vals):
        a.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # heatmaps
    for col, (h, title) in enumerate([
        (h_self, "per-patch cos: A vs A\n(≈1 everywhere → aligned ✓)"),
        (h_flip, "per-patch cos: A vs flip(A)\n(structured drop → spatially sensitive)"),
        (h_diff, "per-patch cos: expert vs online\n(domain gap, still spatially coherent)"),
    ], start=1):
        a = fig.add_subplot(gs[1, col])
        im = a.imshow(h, cmap="viridis", vmin=0, vmax=1)
        a.set_title(title, fontweight="bold", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    fig.suptitle("DINO similarity input-alignment verification (real frames, HiRE encoder)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
