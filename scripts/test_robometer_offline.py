#!/usr/bin/env python3
"""Offline smoke test for Robometer reward shaping (no robot, no env_runner).

Usage (from repo root, same venv as RL learner):
    . ./prepare.sh
    python scripts/test_robometer_offline.py
    python scripts/test_robometer_offline.py --episode ~/data/real_processed/yam_warmup_pool_finestride/episode_0000.npz

    # Imports only (no HTTP to eval server):
    python scripts/test_robometer_offline.py --dry-run

Requires:
    - Existing .venv (prepare.sh) — numpy, torch, etc. already there.
    - pip package ``requests`` (usually already installed; see --dry-run).
    - For live reward test: Robometer eval_server on robometer_server_url (default :8000).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _REPO)


def _find_episode(explicit: str | None) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    roots = [
        os.path.expanduser("~/data/real_processed/yam_warmup_pool_finestride"),
        os.path.expanduser("~/data/real_processed/yam_rl_rollouts_*"),
    ]
    for root in roots:
        for pattern in (root, os.path.join(root, "episode_*.npz")):
            if "*" in pattern:
                hits = sorted(glob.glob(pattern))
            else:
                hits = sorted(glob.glob(os.path.join(pattern, "episode_*.npz")))
            if hits:
                return hits[0]
    raise FileNotFoundError(
        "No episode_*.npz found; pass --episode /path/to/episode_0000.npz"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true", help="imports + config only")
    ap.add_argument("--horizon", type=int, default=16)
    args = ap.parse_args()

    print("=== 1) venv imports ===")
    try:
        import requests  # noqa: F401
    except ImportError:
        print("FAIL: pip install requests")
        sys.exit(1)
    print(f"  requests OK")

    from dice_rl.config.yam_env_overrides import apply_robometer_env_overrides
    from dice_rl.config.yam_rl_config import TRAINING
    from dice_rl.reward.robometer_client import health_check
    from dice_rl.reward.robometer_episode_shaper import (
        RobometerEpisodeRewardShaper,
        resolve_yam_robometer_camera,
    )

    cfg = apply_robometer_env_overrides(dict(TRAINING))
    cfg["use_hire_reward"] = False
    cfg["use_robometer_reward"] = True
    print(f"  robometer_use_relative_rewards = {cfg.get('robometer_use_relative_rewards')}")
    print(f"  robometer_camera = {cfg.get('robometer_camera')}")
    print(f"  alias sideview_image -> {resolve_yam_robometer_camera('sideview_image')}")

    if args.dry_run:
        print("\n=== dry-run OK (no HTTP, no robot) ===")
        return

    url = str(cfg.get("robometer_server_url", "http://127.0.0.1:8000"))
    print(f"\n=== 2) eval server health ({url}) ===")
    if not health_check(url):
        print("FAIL: server not reachable. Start with:")
        print("  bash scripts/robometer/start_eval_server.sh")
        sys.exit(1)
    print("  server healthy")

    ep_path = _find_episode(args.episode)
    d = np.load(ep_path)
    images = d["images"]
    T = images.shape[0]
    print(f"\n=== 3) shape_rewards on disk episode ===")
    print(f"  file: {ep_path}")
    print(f"  images shape: {images.shape}, T={T}")

    shaper = RobometerEpisodeRewardShaper(
        server_url=url,
        task_instruction=str(cfg["robometer_task_instruction"]),
        reward_weight=float(cfg["robometer_reward_weight"]),
        camera=str(cfg["robometer_camera"]),
        use_frame_steps=bool(cfg.get("robometer_use_frame_steps", False)),
        max_frames=int(cfg.get("robometer_max_frames", 16)),
        request_timeout_s=float(cfg.get("robometer_request_timeout_s", 120.0)),
        bgr_to_rgb=bool(cfg.get("robometer_bgr_to_rgb", False)),
        use_relative_rewards=bool(cfg.get("robometer_use_relative_rewards", True)),
        gamma_pbrs=float(cfg.get("robometer_gamma_pbrs", 0.99)),
        query_every_n_chunks=int(cfg.get("robometer_query_every_n_chunks", 4)),
        query_fill_mode=str(cfg.get("robometer_query_fill_mode", "hold")),
        max_batch_size=int(cfg.get("robometer_max_batch_size", 4)),
    )
    r = shaper.shape_rewards(images, horizon=args.horizon)
    print(f"  shaped rewards: len={len(r)}, min={r.min():.4f}, max={r.max():.4f}, mean={r.mean():.4f}")
    print(f"  first 5: {np.round(r[:5], 4)}")
    print("\n=== OK: Robometer path works in this venv (robot not used) ===")


if __name__ == "__main__":
    main()
