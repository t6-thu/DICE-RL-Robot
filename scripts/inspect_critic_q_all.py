#!/usr/bin/env python3
"""Batch run inspect_critic_q diagnostic on ALL existing runs.

Loads BC policy ONCE, then for each run iterates through its latest ckpt and
its own episodes, computes Q diagnostics, and dumps a single comparison table.

Usage:
    . ./prepare.sh
    python scripts/inspect_critic_q_all.py
"""
import glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from dice_rl.config.yam_rl_config import BC_POLICY_CKPT, TRAINING, NETWORK
from dice_rl.model.distill_rl import DistilledActor, DistilledCritic
from utils.model_io import load_policy

ALL_RUNS = [
    # name shown in table             , reported peak
    ("v2",                              "70%"),       # pre-fix baseline (no HiRE)
    ("hire_v2",                         "60%"),       # pre-fix HiRE
    ("hire_lambda09",                   "30%"),       # pre-fix, early
    ("hire_lambda09_fixedsuccess",      "33%"),       # POST-fix
    ("hire_lambda09_fullhire",          "10%"),
    ("hire_v2_recover",                 "10%"),
    ("hire_noclamp_lambda01_onlinesuccesshire", "40%"),
    ("hire_noclamp_lambda01_onlinepos", "40%"),
    ("hire_curated_terminalstride",     "30%"),
    ("bc140_curated_terminalstride",    "30%"),
    ("pos_only_no_neg",                 "20%"),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading BC policy once on {device} …")
    bc_policy, _, _ = load_policy(BC_POLICY_CKPT, device)
    bc_policy.eval()
    obs_feature_dim = bc_policy.obs_feature_dim
    H = TRAINING["action_horizon"]
    A = TRAINING["action_dim"]
    ah, ch = NETWORK["actor_hidden_dims"], NETWORK["critic_hidden_dims"]

    rows = []
    for run_name, reported_peak in ALL_RUNS:
        ckpt_dir = os.path.expanduser(f"~/training_outputs/yam_rl_finetuning_{run_name}")
        eps_dir  = os.path.expanduser(f"~/data/real_processed/yam_rl_rollouts_{run_name}")
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pt")))
        if not ckpts:
            print(f"\n[{run_name}] (no checkpoints, skip)")
            continue
        ckpt_path = ckpts[-1]
        eps_paths = sorted(glob.glob(os.path.join(eps_dir, "episode_*.npz")))[:40]
        if not eps_paths:
            print(f"\n[{run_name}] (no episodes, skip)")
            continue

        print(f"\n[{run_name}] ckpt={os.path.basename(ckpt_path)}  eps={len(eps_paths)}")

        # Build & load actor + critics fresh
        actor = DistilledActor(obs_dim=obs_feature_dim, action_dim=A, cond_steps=1,
                               horizon_steps=H, hidden_dims=ah,
                               activation_type="GELU", use_layernorm=True).to(device).eval()
        critics = torch.nn.ModuleList([
            DistilledCritic(obs_dim=obs_feature_dim, action_dim=A,
                            horizon_steps=H, hidden_dims=ch).to(device).eval()
            for _ in range(TRAINING["critic_ensemble_size"])
        ])
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            actor.load_state_dict(ckpt["actor"])
            for c, sd in zip(critics, ckpt["critics"]):
                c.load_state_dict(sd)
        except Exception as e:
            print(f"  ❌ ckpt load failed: {e}")
            continue

        # Split transitions
        succ_obs, succ_act, fail_obs, fail_act = [], [], [], []
        oh = TRAINING["obs_horizon"]
        for p in eps_paths:
            d = np.load(p)
            images, states, actions, rewards = d["images"], d["states"], d["actions"], d["rewards"]
            is_succ = rewards[-1] > 0.5
            T = images.shape[0]
            for t in range(T - 1):
                frames, jnts = [], []
                for k in range(oh - 1, -1, -1):
                    idx = max(t - k, 0)
                    raw = images[idx].astype(np.float32)
                    frames.append(raw if raw.max() <= 1.0 else raw / 255.0)
                    jnts.append(states[idx])
                obs = {
                    "rgb_0":     np.stack([f[:3] for f in frames]),
                    "rgb_1":     np.stack([f[3:] for f in frames]),
                    "joint_pos": np.stack(jnts),
                }
                a = actions[t] if actions.ndim == 3 else np.tile(actions[t], (H, 1))
                (succ_obs if is_succ else fail_obs).append(obs)
                (succ_act if is_succ else fail_act).append(a)

        # Compute Q stats
        @torch.no_grad()
        def q_stats(obs_list, group_name):
            if not obs_list:
                return None
            bs = 64
            Q_bc, Q_actor, sens = [], [], []
            for i in range(0, len(obs_list), bs):
                chunk = obs_list[i:i+bs]
                rgb_0 = torch.from_numpy(np.stack([o["rgb_0"] for o in chunk])).to(device).float()
                rgb_1 = torch.from_numpy(np.stack([o["rgb_1"] for o in chunk])).to(device).float()
                jpos  = torch.from_numpy(np.stack([o["joint_pos"] for o in chunk])).to(device).float()
                obs_batch = {"rgb_0": rgb_0, "rgb_1": rgb_1, "joint_pos": jpos}
                B = rgb_0.shape[0]
                noise = torch.randn(B, H, A, device=device)
                nobs = {k: bc_policy.sparse_normalizer[k].normalize(v) for k, v in obs_batch.items()}
                feat = bc_policy.obs_encoder(nobs)
                bc_action = bc_policy.predict_action_from_features(
                    sparse_nobs_encode=feat, init_noise=noise, unnormalize=False
                )["sparse"]
                delta = actor(feat.unsqueeze(1), noise)
                actor_action = bc_action + delta
                q_bc_ens    = torch.stack([c(feat, noise, bc_action)    for c in critics], dim=-1)
                q_actor_ens = torch.stack([c(feat, noise, actor_action) for c in critics], dim=-1)
                q_bc_min     = q_bc_ens.min(dim=-1).values.squeeze(-1)
                q_actor_min  = q_actor_ens.min(dim=-1).values.squeeze(-1)
                Q_bc.append(q_bc_min.cpu())
                Q_actor.append(q_actor_min.cpu())
                sens.append((q_actor_min - q_bc_min).abs().cpu())
            return torch.cat(Q_bc), torch.cat(Q_actor), torch.cat(sens)

        succ_q = q_stats(succ_obs, "S")
        fail_q = q_stats(fail_obs, "F")

        if succ_q is None or fail_q is None:
            print("  (insufficient transitions)")
            continue

        s_qbc, s_qa, s_sens = succ_q
        f_qbc, f_qa, f_sens = fail_q

        row = dict(
            run=run_name, peak=reported_peak,
            n_succ=len(s_qbc), n_fail=len(f_qbc),
            Q_succ_actor_mean=float(s_qa.mean()),
            Q_succ_actor_max=float(s_qa.max()),
            Q_fail_actor_mean=float(f_qa.mean()),
            Q_fail_actor_max=float(f_qa.max()),
            Q_gap=float(s_qa.mean() - f_qa.mean()),
            sens_succ=float(s_sens.mean()),
            sens_fail=float(f_sens.mean()),
        )
        rows.append(row)
        print(f"  Q_succ(actor)={row['Q_succ_actor_mean']:+.3f} (max {row['Q_succ_actor_max']:+.3f})  "
              f"Q_fail(actor)={row['Q_fail_actor_mean']:+.3f} (max {row['Q_fail_actor_max']:+.3f})  "
              f"gap={row['Q_gap']:+.3f}  sens={row['sens_succ']:.3f}")

    # Print final table
    print("\n\n========== FINAL TABLE ==========\n")
    print(f"{'run':45s} | {'peak':>5s} | {'Q_succ(μ)':>10s} | {'Q_succ(max)':>12s} | "
          f"{'Q_fail(μ)':>10s} | {'Q_fail(max)':>12s} | {'gap':>6s} | {'sens_S':>6s} | {'sens_F':>6s}")
    print("-" * 140)
    for r in rows:
        print(f"{r['run']:45s} | {r['peak']:>5s} | "
              f"{r['Q_succ_actor_mean']:>+10.3f} | {r['Q_succ_actor_max']:>+12.3f} | "
              f"{r['Q_fail_actor_mean']:>+10.3f} | {r['Q_fail_actor_max']:>+12.3f} | "
              f"{r['Q_gap']:>+6.3f} | {r['sens_succ']:>6.3f} | {r['sens_fail']:>6.3f}")


if __name__ == "__main__":
    main()
