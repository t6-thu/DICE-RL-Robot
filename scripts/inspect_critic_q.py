#!/usr/bin/env python3
"""Diagnose whether the critic learned task-relevant Q values.

Loads:
  - The frozen BC policy (for obs feature encoding + BC reference action)
  - The latest RL checkpoint (actor + critic ensemble)
  - Online episodes from the current run dir

For each transition (sampled, not full enumeration), computes:
  Q_actor   = Q(s, BC(s) + actor(s, z))            # what critic gives actor's action
  Q_bc      = Q(s, BC(s))                          # what critic gives pure BC action
  Q_random  = Q(s, BC(s) + 0.5*random)              # baseline for action-sensitivity

Then splits transitions by their episode's outcome (success / failure) and reports:
  1. Q gap (success vs failure): if critic learned task signal, Q_succ ≫ Q_fail
  2. Action sensitivity:         if critic action-aware, |Q_actor - Q_bc| > 0
  3. Per-transition Q range across the ensemble's 5 critics

Usage:
    . ./prepare.sh
    python scripts/inspect_critic_q.py
    # Optional: --ckpt path  --device cpu  --max-eps 40
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from dice_rl.config.yam_rl_config import (
    BC_POLICY_CKPT, ONLINE_DATA_DIR, RL_CKPT_DIR, TRAINING, NETWORK,
)
from dice_rl.model.distill_rl import DistilledActor, DistilledCritic
from utils.model_io import load_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="path to checkpoint_XXXXXX.pt (default = latest of --ckpt-dir)")
    ap.add_argument("--ckpt-dir", default=None, help="dir containing checkpoint_*.pt (default = current RUN_NAME)")
    ap.add_argument("--eps-dir", default=None, help="dir containing episode_*.npz (default = current RUN_NAME)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-eps", type=int, default=40, help="how many online episodes to load")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Resolve paths (CLI overrides config)
    ckpt_dir = args.ckpt_dir or RL_CKPT_DIR
    eps_dir  = args.eps_dir  or ONLINE_DATA_DIR

    # 1) Resolve checkpoint
    if args.ckpt is None:
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pt")))
        if not ckpts:
            print(f"No checkpoint in {ckpt_dir}"); sys.exit(1)
        args.ckpt = ckpts[-1]
    print(f"[inspect_critic_q] device={device}")
    print(f"[inspect_critic_q] checkpoint = {args.ckpt}")
    print(f"[inspect_critic_q] episodes   = {eps_dir}")

    # 2) Load frozen BC policy
    print(f"[inspect_critic_q] loading BC policy …")
    bc_policy, _, _ = load_policy(BC_POLICY_CKPT, device)
    bc_policy.eval()
    obs_feature_dim = bc_policy.obs_feature_dim
    H = TRAINING["action_horizon"]
    A = TRAINING["action_dim"]

    # 3) Build actor + critic ensemble, load ckpt
    ah = NETWORK["actor_hidden_dims"]
    ch = NETWORK["critic_hidden_dims"]
    actor = DistilledActor(obs_dim=obs_feature_dim, action_dim=A, cond_steps=1,
                           horizon_steps=H, hidden_dims=ah,
                           activation_type="GELU", use_layernorm=True).to(device).eval()
    critics = torch.nn.ModuleList([
        DistilledCritic(obs_dim=obs_feature_dim, action_dim=A,
                        horizon_steps=H, hidden_dims=ch).to(device).eval()
        for _ in range(TRAINING["critic_ensemble_size"])
    ])
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    for c, sd in zip(critics, ckpt["critics"]):
        c.load_state_dict(sd)

    # 4) Load online episodes
    paths = sorted(glob.glob(os.path.join(eps_dir, "episode_*.npz")))[: args.max_eps]
    print(f"[inspect_critic_q] loading {len(paths)} online episodes …")

    succ_states, succ_actions = [], []
    fail_states, fail_actions = [], []
    for p in paths:
        d = np.load(p)
        images = d["images"]      # (T, 6, H, W) float32 [0,1]
        states = d["states"]      # (T, 7) normalized
        actions = d["actions"]    # (T, H, 7) normalized
        rewards = d["rewards"]
        is_succ = rewards[-1] > 0.5
        T = images.shape[0]
        # use obs_horizon = 2 (matches BC training)
        oh = TRAINING["obs_horizon"]
        # Build per-transition obs history (mirrors yam_replay_buffer._make_obs)
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
            target = succ_states if is_succ else fail_states
            target_a = succ_actions if is_succ else fail_actions
            target.append(obs)
            target_a.append(a)

    print(f"[inspect_critic_q] success transitions: {len(succ_states)}  | failure transitions: {len(fail_states)}")

    @torch.no_grad()
    def encode_and_q(obs_list, action_list, tag):
        if not obs_list:
            print(f"  {tag}: (none)")
            return None
        # Batch in mini-batches to avoid OOM
        bs = 64
        Q_bc_all, Q_actor_all, Q_random_all, Q_ensemble_range_all = [], [], [], []
        action_sensitivity = []
        for i in range(0, len(obs_list), bs):
            chunk = obs_list[i:i + bs]
            act_chunk = action_list[i:i + bs]
            rgb_0 = torch.from_numpy(np.stack([o["rgb_0"] for o in chunk])).to(device).float()
            rgb_1 = torch.from_numpy(np.stack([o["rgb_1"] for o in chunk])).to(device).float()
            jpos  = torch.from_numpy(np.stack([o["joint_pos"] for o in chunk])).to(device).float()
            obs_batch = {"rgb_0": rgb_0, "rgb_1": rgb_1, "joint_pos": jpos}
            B = rgb_0.shape[0]
            noise = torch.randn(B, H, A, device=device)
            # Encode obs through the BC obs_encoder (mirrors learner._encode_obs)
            nobs = {k: bc_policy.sparse_normalizer[k].normalize(v) for k, v in obs_batch.items()}
            feat = bc_policy.obs_encoder(nobs)        # (B, D)
            # Get BC action from features
            bc_action = bc_policy.predict_action_from_features(
                sparse_nobs_encode=feat, init_noise=noise, unnormalize=False
            )["sparse"]                                # (B, H, A)
            # Actor delta + total action
            delta = actor(feat.unsqueeze(1), noise)
            actor_action = bc_action + delta
            random_action = bc_action + 0.5 * torch.randn_like(bc_action)
            # Q for each critic in ensemble — feature, noise, action
            q_bc_ens     = torch.stack([c(feat, noise, bc_action)     for c in critics], dim=-1)  # (B,1,5)
            q_actor_ens  = torch.stack([c(feat, noise, actor_action)  for c in critics], dim=-1)
            q_random_ens = torch.stack([c(feat, noise, random_action) for c in critics], dim=-1)
            # min across ensemble (= what actor sees)
            q_bc     = q_bc_ens.min(dim=-1).values.squeeze(-1)        # (B,)
            q_actor  = q_actor_ens.min(dim=-1).values.squeeze(-1)
            q_random = q_random_ens.min(dim=-1).values.squeeze(-1)
            ens_range = (q_actor_ens.max(dim=-1).values - q_actor_ens.min(dim=-1).values).squeeze(-1)
            Q_bc_all.append(q_bc.cpu()); Q_actor_all.append(q_actor.cpu())
            Q_random_all.append(q_random.cpu()); Q_ensemble_range_all.append(ens_range.cpu())
            action_sensitivity.append((q_actor - q_bc).abs().cpu())

        Q_bc = torch.cat(Q_bc_all);     Q_actor = torch.cat(Q_actor_all)
        Q_random = torch.cat(Q_random_all); ens_rng = torch.cat(Q_ensemble_range_all)
        sens = torch.cat(action_sensitivity)
        print(f"\n  {tag}  ({len(Q_bc)} transitions)")
        print(f"    Q(s, BC):     mean={Q_bc.mean():+.4f}  std={Q_bc.std():.4f}  "
              f"min={Q_bc.min():+.4f}  max={Q_bc.max():+.4f}")
        print(f"    Q(s, actor):  mean={Q_actor.mean():+.4f}  std={Q_actor.std():.4f}  "
              f"min={Q_actor.min():+.4f}  max={Q_actor.max():+.4f}")
        print(f"    Q(s, random): mean={Q_random.mean():+.4f}  std={Q_random.std():.4f}")
        print(f"    ensemble range (max-min over 5 critics): mean={ens_rng.mean():.4f}")
        print(f"    action sensitivity |Q(s,actor) - Q(s,BC)|: mean={sens.mean():.4f}  "
              f"max={sens.max():.4f}")
        return Q_bc, Q_actor, Q_random

    succ_q = encode_and_q(succ_states, succ_actions, "[SUCCESS]")
    fail_q = encode_and_q(fail_states, fail_actions, "[FAILURE]")

    # 5) Final verdict
    if succ_q is not None and fail_q is not None:
        succ_Q_bc, succ_Q_actor, _ = succ_q
        fail_Q_bc, fail_Q_actor, _ = fail_q
        gap_actor = succ_Q_actor.mean() - fail_Q_actor.mean()
        gap_bc    = succ_Q_bc.mean()    - fail_Q_bc.mean()
        print("\n=== VERDICT ===")
        print(f"  Q gap (success − failure) under actor action: {gap_actor:+.4f}")
        print(f"  Q gap (success − failure) under BC action:    {gap_bc:+.4f}")
        print(f"  → If gap < 0.05: critic does NOT distinguish task outcomes (likely cheat)")
        print(f"  → If gap > 0.3:  critic learned meaningful Q signal")

        # Action sensitivity
        all_states = succ_states + fail_states
        all_actions = succ_actions + fail_actions
        _ = None  # already printed per-group above
        print(f"\n  Action-sensitivity (BC vs actor) printed per-group above.")
        print(f"  → If <0.02: critic ≈ action-agnostic (cheat solution likely)")
        print(f"  → If >0.10: critic action-aware (good for RL training)")


if __name__ == "__main__":
    main()
