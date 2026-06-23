#!/usr/bin/env python3
"""Interactive evaluation of any RL checkpoint in the current run.

Workflow:
    1. Lists every `checkpoint_*.pt` in RL_CKPT_DIR (plus a "BC-only" slot).
    2. You pick one → it's loaded into a fresh DistilledActor.
    3. Run as many evaluation episodes as you like with that ckpt:
         Enter   → start episode
         s/f/d   → label success / failure / discard
       Episodes are saved into:
         <ONLINE_DATA_DIR>/eval/<ckpt_label>/episode_NNNN.npz
       …in exactly the same format as env_runner, so they can be replayed with
       `python scripts/view_episode.py <that dir>` immediately.
    4. After each episode, choose:
         Enter   → next eval episode with the same ckpt
         c       → pick a different ckpt
         q       → quit
    5. Running tally of s/f for each ckpt is printed.

Refers to dice_rl/env_runner/yam_rl_env_runner.py for the rollout loop and to
scripts/view_episode.py for the .npz layout.

Usage:
    . ./prepare.sh
    python scripts/eval_ckpt.py                              # interactive picker
    python scripts/eval_ckpt.py --ckpt checkpoint_006000.pt  # auto-pick one ckpt
    python scripts/eval_ckpt.py --bc                         # pure BC, no actor
"""
from __future__ import annotations
import argparse
import glob
import logging
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from dice_rl.config.yam_rl_config import (
    BC_POLICY_CKPT, NORM_NPZ, RUN_NAME, ONLINE_DATA_DIR, RL_CKPT_DIR,
    TRAINING, NETWORK, HARDWARE, COMM,
)
from dice_rl.env_runner.yam_rl_env_runner import YAMRLEnvRunner
from dice_rl.model.distill_rl import DistilledActor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
)
log = logging.getLogger("eval_ckpt")


# ---- ckpt discovery + picker ----

def list_ckpts(ckpt_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pt")))


def pick_ckpt_interactive(ckpt_dir: str) -> Optional[str]:
    """Return ckpt path, "BC" for pure BC, or None to quit."""
    ckpts = list_ckpts(ckpt_dir)
    if not ckpts:
        print(f"\n(no checkpoint_*.pt files under {ckpt_dir})")
    print("\n=== checkpoints in", ckpt_dir, "===")
    print("  [0] BC-only (no residual actor)")
    for i, c in enumerate(ckpts, start=1):
        mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(c)))
        sz = os.path.getsize(c) / 1e6
        print(f"  [{i}] {os.path.basename(c)}   {sz:.1f} MB   {mt}")
    print("  [q] quit")
    while True:
        s = input("pick ckpt: ").strip().lower()
        if s == "q":
            return None
        if s == "0":
            return "BC"
        if s.isdigit() and 1 <= int(s) <= len(ckpts):
            return ckpts[int(s) - 1]
        print("  bad choice; try again")


# ---- load actor into the runner ----

def load_actor_from_ckpt(ckpt_path: str, obs_feature_dim: int, device: torch.device) -> DistilledActor:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor = DistilledActor(
        obs_dim=obs_feature_dim,
        action_dim=TRAINING["action_dim"],
        cond_steps=1,
        horizon_steps=TRAINING["action_horizon"],
        hidden_dims=NETWORK["actor_hidden_dims"],
        activation_type="GELU",
        use_layernorm=True,
    ).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


# ---- main ----

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None,
                   help="checkpoint filename (under RL_CKPT_DIR) or full path; skips picker")
    p.add_argument("--bc", action="store_true",
                   help="evaluate pure BC (no residual actor)")
    p.add_argument("--out-root", type=str, default=os.path.join(ONLINE_DATA_DIR, "eval"),
                   help="root dir for saved eval episodes")
    p.add_argument("--residual-scale", type=float, default=1.0,
                   help="scale on actor delta (matches env_runner default 1.0)")
    p.add_argument("--max-joint-step", type=float, default=0.04,
                   help="max absolute command change per 30 Hz step; keep this "
                        "conservative for real-robot ckpt eval")
    args = p.parse_args()

    # Build the env runner with NO weights-watch path → it never auto-loads
    # latest_actor.pt under our feet during eval.
    log.info("Constructing env runner (loading BC, opening cameras, homing) …")
    runner = YAMRLEnvRunner(
        pretrained_policy_ckpt=BC_POLICY_CKPT,
        norm_npz_path=NORM_NPZ,
        base_cam_serial=HARDWARE["base_cam_serial"],
        wrist_cam_serial=HARDWARE["wrist_cam_serial"],
        can_channel=HARDWARE["can_channel"],
        gripper_type=HARDWARE["gripper_type"],
        home_joint_pos=HARDWARE["home_joint_pos"],
        home_gripper_pos=HARDWARE["home_gripper_pos"],
        control_hz=HARDWARE["control_hz"],
        max_episode_steps=HARDWARE["max_episode_steps"],
        obs_horizon=TRAINING["obs_horizon"],
        action_horizon=TRAINING["action_horizon"],
        action_dim=TRAINING["action_dim"],
        actor_hidden_dims=NETWORK["actor_hidden_dims"],
        residual_scale=args.residual_scale,
        max_joint_step=args.max_joint_step,
        online_data_dir="/tmp/_eval_dummy_unused",   # we override save dir ourselves
        rl_checkpoint_dir=None,                       # disables auto-update from latest_actor.pt
        network_server_endpoint=COMM["network_server_endpoint"],
        network_weight_topic=COMM["network_weight_topic"],
        transitions_server_endpoint=COMM["transitions_server_endpoint"],
        transitions_topic=COMM["transitions_topic"],
    )
    runner._move_to_home()

    # Track results per ckpt across the session.
    tally: dict[str, dict] = {}

    def get_choice() -> Optional[str]:
        if args.bc:
            return "BC"
        if args.ckpt:
            path = args.ckpt
            if not os.path.isabs(path):
                path = os.path.join(RL_CKPT_DIR, path)
            if not os.path.exists(path):
                log.error("ckpt not found: %s", path); return None
            return path
        return pick_ckpt_interactive(RL_CKPT_DIR)

    choice = get_choice()
    # Only auto-pick once; subsequent rounds always use the picker.
    args.ckpt = None
    args.bc = False

    while choice is not None:
        # Configure runner for this choice.
        if choice == "BC":
            runner.actor = None
            runner._actor_step = -1
            label = "bc_only"
            log.info("=== Eval mode: pure BC (no residual actor) ===")
        else:
            log.info("Loading ckpt: %s", choice)
            try:
                runner.actor = load_actor_from_ckpt(choice, runner._obs_feature_dim, runner.device)
            except Exception as e:
                log.error("Failed to load actor from %s: %s", choice, e)
                choice = pick_ckpt_interactive(RL_CKPT_DIR); continue
            ckpt_data = torch.load(choice, map_location="cpu", weights_only=False)
            runner._actor_step = int(ckpt_data.get("total_gradient_steps", -1))
            label = os.path.basename(choice).replace(".pt", "")
            log.info("=== Eval mode: %s (training step=%d) ===", label, runner._actor_step)

        save_dir = os.path.join(args.out_root, label)
        os.makedirs(save_dir, exist_ok=True)
        existing = sorted(glob.glob(os.path.join(save_dir, "episode_*.npz")))
        ep_idx = len(existing)
        if ep_idx > 0:
            log.info("Resuming under %s (%d eval eps already saved)", save_dir, ep_idx)

        if label not in tally:
            tally[label] = {"s": 0, "f": 0, "d": 0}
            # Pre-count s/f from any resumed files so the tally is honest.
            for f in existing:
                try:
                    r = np.load(f)["rewards"]
                    if len(r) and r[-1] > 0.5: tally[label]["s"] += 1
                    else:                       tally[label]["f"] += 1
                except Exception: pass

        # Inner episode loop with the chosen ckpt.
        while True:
            actor_info = (f"actor step={runner._actor_step}"
                          if runner.actor is not None else "pure BC")
            t = tally[label]
            print(f"\n[{label} | {actor_info} | s={t['s']} f={t['f']} d={t['d']}] "
                  f"Press Enter to start eval ep {ep_idx+1}, "
                  f"c=change ckpt, q=quit.")
            try:
                sel = input().strip().lower()
            except EOFError:
                sel = "q"
            if sel == "q":
                choice = None; break
            if sel == "c":
                choice = pick_ckpt_interactive(RL_CKPT_DIR); break

            try:
                ep_data = runner.run_episode()
            except Exception as e:
                log.error("Episode crashed: %s", e)
                runner._move_to_home()
                continue

            if ep_data.get("discard"):
                tally[label]["d"] += 1
                log.info("Episode discarded. (d=%d)", tally[label]["d"])
                runner._move_to_home()
                continue

            ok = bool(ep_data["success"])
            tally[label]["s" if ok else "f"] += 1
            ep_path = os.path.join(save_dir, f"episode_{ep_idx:04d}.npz")
            np.savez_compressed(
                ep_path,
                images=(ep_data["images"] * 255.0).clip(0, 255).astype(np.uint8),
                states=ep_data["states"],
                actions=ep_data["actions"],
                rewards=ep_data["rewards"],
                dones=ep_data["dones"],
            )
            tot = tally[label]["s"] + tally[label]["f"]
            sr  = tally[label]["s"] / max(tot, 1) * 100.0
            log.info("Saved %s  (success=%s)  → %s/%s = %.1f%%",
                     os.path.basename(ep_path), ok, tally[label]["s"], tot, sr)
            ep_idx += 1
            runner._move_to_home()

    # ---- final summary ----
    print("\n=========== EVAL SUMMARY ===========")
    for label, t in tally.items():
        tot = t["s"] + t["f"]
        sr = (t["s"] / tot * 100.0) if tot else 0.0
        print(f"  {label:30s}  s={t['s']:3d}  f={t['f']:3d}  d={t['d']:3d}  → {sr:5.1f}%")
    print(f"Replay any of them with:")
    print(f"  python scripts/view_episode.py {args.out_root}/<label>/")


if __name__ == "__main__":
    main()
