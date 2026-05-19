#!/usr/bin/env python3
"""Offline replay sanity check for the DICE-RL-Robot diffusion policy.

Loads a checkpoint, feeds ground-truth training observations one chunk at a
time, and measures MAE vs. the recorded actions. A healthy BC checkpoint has
exec-only MAE < 0.05 (normalized [-1,1] action space).

Usage:
    source /home/bike/Documents/niu/DICE-RL-Robot/.venv/bin/activate
    . /home/bike/Documents/niu/DICE-RL-Robot/prepare.sh

    python /home/bike/Documents/niu/DICE-RL-Robot/scripts/eval_dp_offline_replay.py \
        --ckpt ~/training_outputs/2026.05.19/<run>/checkpoints/latest.ckpt \
        --replay ~/data/real_processed/yam_picknplace_arizonabottle_224/train.npz \
        --num_episodes 1 \
        --query_stride 16
"""

from __future__ import annotations
import argparse, time, os, sys
import numpy as np
import torch
import hydra

# Dataset sits next to this repo; import it to load obs properly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_policy(ckpt_path: str, device: torch.device):
    """Load the diffusion policy from a DICE-RL-Robot checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['cfg']
    # Reconstruct the policy from the saved Hydra config.
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(ckpt['state_dicts']['model'])
    policy.to(device)
    policy.eval()
    return policy, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--replay', required=True,
                        help='Path to train.npz from process_yam_dataset.py '
                             '(must match the image_size the model was trained on).')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--query_stride', type=int, default=16,
                        help='Frames between queries (matches act_steps from cfg).')
    parser.add_argument('--num_inference_steps', type=int, default=None,
                        help='Override DDIM steps (None=use training default).')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[offline_replay] loading checkpoint: {args.ckpt}")
    policy, cfg = load_policy(args.ckpt, device)
    print(f"[offline_replay] epoch={cfg.get('epoch','?')}  "
          f"train_loss={cfg.get('train_loss','?')}")

    if args.num_inference_steps is not None:
        policy.num_inference_steps = args.num_inference_steps
        print(f"[offline_replay] overriding num_inference_steps → {args.num_inference_steps}")

    rep = np.load(args.replay)
    states  = rep['states']    # (T, 7) float32, normalized
    actions_gt = rep['actions'] # (T, 7) float32, normalized
    images  = rep['images']    # (T, 6, H, W) uint8  base:0:3  wrist:3:6
    traj_lengths = rep['traj_lengths']

    cond_steps     = cfg.task.obs_horizon       # e.g. 2
    img_cond_steps = cfg.task.obs_horizon       # same
    action_horizon = cfg.task.action_horizon    # e.g. 16
    act_steps      = args.query_stride

    print(f"[offline_replay] data: {len(traj_lengths)} episodes, "
          f"{states.shape[0]} frames  |  image {images.shape[1:]}")
    print(f"[offline_replay] obs_horizon={cond_steps}  action_horizon={action_horizon}  "
          f"query_stride={act_steps}")
    print()
    print(f"{'t':>4} | {'infer_ms':>8} | {'exec_mae':>8} | {'chunk_mae':>9}")
    print('-' * 45)

    per_chunk_mae = []
    per_step_mae  = []
    per_joint     = np.zeros(7, dtype=np.float64)
    n_rows        = 0

    ep_starts = np.concatenate([[0], np.cumsum(traj_lengths[:-1])])

    for ep_idx, ep_len in enumerate(traj_lengths[:args.num_episodes]):
        s0 = int(ep_starts[ep_idx])
        s1 = s0 + int(ep_len)

        for t in range(s0, s1 - action_horizon, act_steps):
            # ---- build cond ---- #
            def _stack(arr, n):
                return np.stack([arr[max(t - k, s0)] for k in reversed(range(n))])

            s_hist   = _stack(states, cond_steps)         # (To, 7)
            img_hist = _stack(images, img_cond_steps)     # (To, 6, H, W)

            rgb0 = img_hist[:, :3].astype(np.float32) / 255.0  # (To, 3, H, W)
            rgb1 = img_hist[:, 3:].astype(np.float32) / 255.0

            obs_dict = {
                'rgb_0':    torch.from_numpy(rgb0)[None].to(device),    # (1, To, 3, H, W)
                'rgb_1':    torch.from_numpy(rgb1)[None].to(device),
                'joint_pos': torch.from_numpy(s_hist)[None].to(device).float(),
            }

            # ---- inference ---- #
            t0 = time.perf_counter()
            with torch.no_grad():
                result = policy.predict_action({'obs': {'sparse': obs_dict}})
            infer_ms = (time.perf_counter() - t0) * 1000.0

            # predict_action returns a dict; extract action chunk
            if isinstance(result, dict):
                pred_chunk = result.get('action', result.get('action_pred'))
                if pred_chunk is None:
                    pred_chunk = next(iter(result.values()))
                if isinstance(pred_chunk, dict):
                    pred_chunk = pred_chunk.get('sparse', next(iter(pred_chunk.values())))
            else:
                pred_chunk = result
            pred_np = pred_chunk[0].cpu().numpy()       # (action_horizon, 7)

            gt = actions_gt[t : t + action_horizon]     # (≤action_horizon, 7)
            n  = min(pred_np.shape[0], gt.shape[0])
            err = np.abs(pred_np[:n] - gt[:n])

            chunk_mae = float(err.mean())
            step_mae  = float(err[:act_steps].mean())
            per_chunk_mae.append(chunk_mae)
            per_step_mae.append(step_mae)
            per_joint += err.sum(axis=0)
            n_rows    += n

            print(f"{t - s0:4d} | {infer_ms:8.1f} | {step_mae:8.4f} | {chunk_mae:9.4f}")

    print()
    print('=' * 60)
    print(f"OFFLINE REPLAY SUMMARY  ({n_rows} prediction rows, normalized [-1,1])")
    print(f"  num chunks queried    : {len(per_chunk_mae)}")
    print(f"  full-chunk MAE (mean) : {np.mean(per_chunk_mae):.4f}")
    print(f"  exec-only MAE (mean)  : {np.mean(per_step_mae):.4f}  "
          f"(first {act_steps} steps per chunk)")
    print(f"  per-joint MAE         : {(per_joint / max(n_rows, 1)).round(4).tolist()}")
    print(f"  rule of thumb         : <0.05 healthy; >0.15 indicates misalignment")
    print('=' * 60)


if __name__ == '__main__':
    main()
