#!/usr/bin/env python3
"""Offline replay sanity check for the flow-matching YAM policy.

Replays one (or more) **training-set** episode through the policy, comparing
the predicted action chunk against the ground-truth actions stored in the
dataset. Use this BEFORE the real-hardware dry_run to confirm that:
  * the checkpoint loads with the right config / EMA weights
  * obs preprocessing matches what training saw (states + 6-channel 128x128 images)
  * normalization is applied correctly on both ends
  * the model outputs are sane in shape and magnitude

A useful pass looks like: mean absolute error (MAE) between predicted and
recorded actions at <0.05 in normalized [-1, 1] space (~5% of full range).
Larger errors are still expected in regions where the recorded behavior is
multimodal --- this is a smoke check, not a tight numerical equivalence.

Usage:

    source /home/bike/Documents/niu/DICE-RL-Robot/.venv/bin/activate
    python /home/bike/Documents/niu/DICE-RL-Robot/scripts/eval_flow_matching_offline_replay.py \\
        --config <ckpt_dir>/.hydra/config.yaml \\
        --ckpt   <ckpt_dir>/checkpoint/state_700.pt \\
        --norm   <ckpt_dir>/data_meta/normalization.npz \\
        --replay <ckpt_dir>/data_meta/yam_replay_episode0.npz
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Make dice-rl's `model/` package importable.
_DICE_RL_REPO = os.environ.get("DICE_RL_REPO", str(Path.home() / "Documents/niu/dice-rl"))
if _DICE_RL_REPO not in sys.path:
    sys.path.insert(0, _DICE_RL_REPO)

import numpy as np
import torch
from omegaconf import OmegaConf

log = logging.getLogger("offline_replay")
logging.basicConfig(level=logging.INFO, format="[%(name)s %(levelname)s] %(message)s")


def load_flow_policy(config_path: str, ckpt_path: str, device: torch.device, use_ema: bool):
    OmegaConf.register_new_resolver("now", lambda fmt: "n/a", replace=True)
    cfg = OmegaConf.load(config_path)
    import hydra

    model = hydra.utils.instantiate(cfg.model).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    which = "ema" if (use_ema and isinstance(state, dict) and "ema" in state) else "model"
    sd = state[which] if isinstance(state, dict) and which in state else state
    log.info("using %r weights (epoch=%s)", which, state.get("epoch") if isinstance(state, dict) else "n/a")
    m, u = model.load_state_dict(sd, strict=False)
    log.info("load: %d missing, %d unexpected", len(m), len(u))
    model.eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--norm", required=True)
    parser.add_argument("--replay", required=True,
                        help="Path to extracted episode npz (states, actions, images, traj_lengths).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="How many episodes from the replay to evaluate (default 1).")
    parser.add_argument("--query_stride", type=int, default=8,
                        help="Spacing of query frames inside an episode. Default 8 matches "
                        "cfg.act_steps (1 inference per chunk).")
    parser.add_argument("--dump_dir", default=None,
                        help="Optional dir to dump per-step pred vs target arrays for inspection.")
    args = parser.parse_args()

    device = torch.device(args.device)
    log.info("device=%s", device)

    model, cfg = load_flow_policy(args.config, args.ckpt, device, use_ema=not args.no_ema)
    cond_steps = int(cfg.cond_steps)
    img_cond_steps = int(cfg.img_cond_steps)
    horizon_steps = int(cfg.horizon_steps)
    act_steps = int(cfg.act_steps)
    log.info("policy cfg: cond=%d img_cond=%d horizon=%d act_steps=%d flow_steps=%d",
             cond_steps, img_cond_steps, horizon_steps, act_steps, int(cfg.flow_steps))

    # The replay file holds already-normalized states/actions and uint8 images.
    rep = np.load(args.replay)
    states = rep["states"]                # (T, 7) float32 in [-1, 1]
    actions_gt = rep["actions"]           # (T, 7) float32 in [-1, 1]
    images = rep["images"]                # (T, 6, 128, 128) uint8
    traj_lengths = rep["traj_lengths"]    # (E,) int64
    log.info("replay: %d episodes, %d frames total, image dtype=%s",
             len(traj_lengths), states.shape[0], images.dtype)

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    # Aggregated stats across the eval.
    per_chunk_mae = []
    per_step_mae = []
    per_joint_sum = np.zeros(7, dtype=np.float64)
    n_rows = 0

    cur = 0
    for ep_idx, ep_len in enumerate(traj_lengths[: args.num_episodes]):
        ep_start = cur
        ep_end = cur + int(ep_len)
        cur = ep_end
        log.info("=== episode %d  frames=[%d, %d)  len=%d ===", ep_idx, ep_start, ep_end, ep_len)

        for t in range(ep_start, ep_end - 1, args.query_stride):
            # Build cond with most-recent at end, replicating sequence.py pad behaviour.
            def stack(arr, n):
                idxs = [max(t - k, ep_start) for k in reversed(range(n))]
                return np.stack([arr[i] for i in idxs], axis=0)

            state_hist = stack(states, cond_steps)
            img_hist = stack(images, img_cond_steps)

            state_t = torch.from_numpy(state_hist)[None].to(device).float()
            img_t = torch.from_numpy(img_hist)[None].to(device).float()
            cond = {"state": state_t, "rgb": img_t}

            t0 = time.perf_counter()
            with torch.no_grad():
                sample = model(cond=cond, deterministic=True)
            actions_pred = sample.trajectories[0].cpu().numpy()           # (16, 7) in [-1, 1]
            target = actions_gt[t : t + horizon_steps]                    # (<=16, 7)

            n = min(actions_pred.shape[0], target.shape[0])
            err = np.abs(actions_pred[:n] - target[:n])                   # (n, 7)
            chunk_mae = float(err.mean())
            step_mae = float(err[: min(act_steps, n)].mean())
            per_chunk_mae.append(chunk_mae)
            per_step_mae.append(step_mae)
            per_joint_sum += err.sum(axis=0)
            n_rows += n

            infer_ms = (time.perf_counter() - t0) * 1000.0
            log.info("  [t=%4d] infer=%.1fms  chunk_mae=%.4f  exec_mae=%.4f  pred[0]=%s  gt[0]=%s",
                     t - ep_start, infer_ms, chunk_mae, step_mae,
                     np.round(actions_pred[0], 3).tolist(),
                     np.round(target[0], 3).tolist())

            if args.dump_dir:
                np.savez_compressed(
                    os.path.join(args.dump_dir, f"ep{ep_idx:03d}_t{t-ep_start:04d}.npz"),
                    state_hist=state_hist, img_hist=img_hist,
                    actions_pred=actions_pred, actions_gt=target,
                )

    chunk_mae_mean = float(np.mean(per_chunk_mae)) if per_chunk_mae else float("nan")
    step_mae_mean = float(np.mean(per_step_mae)) if per_step_mae else float("nan")
    per_joint = per_joint_sum / max(n_rows, 1)
    log.info("======================================================================")
    log.info("OFFLINE REPLAY SUMMARY (normalized [-1, 1] action space):")
    log.info("  num chunks queried       : %d", len(per_chunk_mae))
    log.info("  full-chunk MAE (mean)    : %.4f", chunk_mae_mean)
    log.info("  exec-only MAE (mean)     : %.4f  (first %d steps per chunk)",
             step_mae_mean, act_steps)
    log.info("  per-joint MAE            : %s", np.round(per_joint, 4).tolist())
    log.info("  rule of thumb            : <0.05 healthy; >0.15 indicates misalignment")
    log.info("======================================================================")


if __name__ == "__main__":
    main()
