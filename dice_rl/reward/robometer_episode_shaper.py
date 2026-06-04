"""Robometer-4B dense reward for YAM offline episodes (learner replay buffer).

Aligned with HiRE-Dice_RL ``util/robometer_reward_shaper.py`` / LIBERO wrapper:
  - Progress head: scalar = clip(progress(prefix)[-1], 0, 1)
  - Online transitions use **Robometer-only** relative progress (default):
        r_t = reward_weight · (progress_{t+H} − progress_t)
    Optional absolute PBRS when ``use_relative_rewards=False``.
  - Expert demos in the replay buffer stay sparse (env label only), matching
    ``expert_dataset.use_env_rewards_only=true`` in HiRE-Dice robometer RLPD runs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dice_rl.reward.robometer_client import (
    extract_progress_from_outputs,
    health_check,
    make_progress_sample,
    post_evaluate_batch_npy,
    subsample_trajectory_frames,
)

log = logging.getLogger(__name__)

# Logical names → slice of the 6-channel YAM episode tensor (T, 6, H, W).
YAM_CAMERA_ALIASES = {
    "base": "base",
    "wrist": "wrist",
    "rgb_0": "base",
    "rgb_1": "wrist",
    # HiRE-Dice robomimic keys (fixed third-person ≈ base on YAM).
    "agentview_image": "base",
    "sideview_image": "base",
}


def resolve_yam_robometer_camera(name: str) -> str:
    """Normalize camera selector to ``base`` or ``wrist``."""
    key = str(name).strip()
    resolved = YAM_CAMERA_ALIASES.get(key)
    if resolved is None:
        raise ValueError(
            f"Unknown robometer camera {name!r}; use one of "
            f"{sorted(YAM_CAMERA_ALIASES)}"
        )
    return resolved


def clamp_progress(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return np.clip(values.astype(np.float64), 0.0, 1.0)


class RobometerEpisodeRewardShaper:
    """Shape per-transition rewards for a saved YAM episode via Robometer HTTP server."""

    def __init__(
        self,
        *,
        server_url: str = "http://127.0.0.1:8000",
        task_instruction: str,
        reward_weight: float = 1.0,
        camera: str = "base",
        use_frame_steps: bool = False,
        max_frames: int = 16,
        request_timeout_s: float = 120.0,
        bgr_to_rgb: bool = True,
        use_relative_rewards: bool = True,
        gamma_pbrs: float = 0.99,
        query_every_n_chunks: int = 1,
        query_fill_mode: str = "hold",
        max_batch_size: Optional[int] = 4,
    ) -> None:
        self.server_url = str(server_url)
        self.task_instruction = str(task_instruction)
        self.reward_weight = float(reward_weight)
        self.camera = resolve_yam_robometer_camera(camera)
        self.use_frame_steps = bool(use_frame_steps)
        self.max_frames = int(max_frames)
        self.request_timeout_s = float(request_timeout_s)
        self.bgr_to_rgb = bool(bgr_to_rgb)
        self.use_relative_rewards = bool(use_relative_rewards)
        self.gamma_pbrs = float(gamma_pbrs)
        self.query_every_n_chunks = max(1, int(query_every_n_chunks))
        fill_mode = str(query_fill_mode).lower()
        if fill_mode not in ("hold", "linear"):
            raise ValueError(
                f"robometer_query_fill_mode must be 'hold' or 'linear', got {fill_mode!r}"
            )
        self.query_fill_mode = fill_mode
        self.max_batch_size = None if max_batch_size is None else max(1, int(max_batch_size))

        if self.reward_weight > 0.0:
            if health_check(self.server_url, timeout_s=5.0):
                log.info("RobometerEpisodeRewardShaper: server healthy at %s", self.server_url)
            else:
                log.warning(
                    "RobometerEpisodeRewardShaper: server not reachable at %s "
                    "(episode shaping will fail until server starts)",
                    self.server_url,
                )
            log.info(
                "Robometer YAM shaper: task=%r camera=%s weight=%.3f "
                "relative=%s max_frames=%d query_every_n_chunks=%d fill=%s gamma_pbrs=%.3f",
                self.task_instruction,
                self.camera,
                self.reward_weight,
                self.use_relative_rewards,
                self.max_frames,
                self.query_every_n_chunks,
                self.query_fill_mode,
                self.gamma_pbrs,
            )

    def is_ready(self) -> bool:
        return self.reward_weight > 0.0

    def _extract_hwc_frames(self, images_T6HW: np.ndarray) -> np.ndarray:
        """(T, 6, H, W) → (T, H, W, 3) uint8 for the selected camera."""
        imgs = np.asarray(images_T6HW)
        if imgs.ndim != 4 or imgs.shape[1] != 6:
            raise ValueError(f"Expected images (T, 6, H, W), got {imgs.shape}")
        if self.camera == "base":
            raw = imgs[:, :3]
        elif self.camera == "wrist":
            raw = imgs[:, 3:]
        else:
            raise ValueError(f"camera must be 'base' or 'wrist', got {self.camera!r}")

        if raw.dtype == np.uint8:
            chw = raw
        else:
            chw = np.clip(raw.astype(np.float32), 0.0, 1.0)
            if chw.max() <= 1.0 + 1e-6:
                chw = (chw * 255.0).astype(np.uint8)
            else:
                chw = chw.astype(np.uint8)

        # CHW → HWC
        hwc = np.transpose(chw, (0, 2, 3, 1))
        out = np.ascontiguousarray(hwc)
        if self.bgr_to_rgb and out.shape[-1] == 3:
            out = out[..., ::-1]
        return out

    def _frames_for_server(self, frames_thwc: np.ndarray) -> np.ndarray:
        return subsample_trajectory_frames(frames_thwc, self.max_frames)

    @staticmethod
    def _progress_scalar(progress_curve: np.ndarray) -> float:
        if progress_curve.size == 0:
            return 0.0
        return float(clamp_progress(progress_curve)[-1])

    def _query_progress_batch(
        self, prefixes: Sequence[Tuple[int, np.ndarray]]
    ) -> Dict[int, float]:
        """Query progress scalars for multiple trajectory prefixes (batched HTTP)."""
        if not prefixes:
            return {}
        out: Dict[int, float] = {}
        batch: List[Tuple[int, np.ndarray, str]] = []
        for end_idx, frames in prefixes:
            batch.append((end_idx, frames, f"yam_ep_{end_idx}"))

        max_bs = self.max_batch_size or len(batch)
        for offset in range(0, len(batch), max_bs):
            chunk = batch[offset : offset + max_bs]
            samples = [
                make_progress_sample(
                    frames=self._frames_for_server(frames),
                    task=self.task_instruction,
                    sample_id=sample_id,
                    subsequence_length=int(frames.shape[0]),
                )
                for _, frames, sample_id in chunk
            ]
            outputs = post_evaluate_batch_npy(
                self.server_url,
                samples,
                timeout_s=self.request_timeout_s,
                use_frame_steps=self.use_frame_steps,
            )
            for i, (end_idx, _, _) in enumerate(chunk):
                curve = extract_progress_from_outputs(outputs, sample_index=i)
                out[end_idx] = self._progress_scalar(curve)
        return out

    def _checkpoint_indices(self, T: int, horizon: int) -> List[int]:
        """Frame indices (inclusive) at which to query Robometer progress."""
        step = horizon * self.query_every_n_chunks
        indices = list(range(0, T, step))
        if indices[-1] != T - 1:
            indices.append(T - 1)
        return sorted(set(indices))

    def _interpolate_progress(
        self,
        frame_idx: int,
        progress_at: Dict[int, float],
        checkpoints: List[int],
    ) -> float:
        if frame_idx in progress_at:
            return float(progress_at[frame_idx])
        if not checkpoints:
            return 0.0
        if frame_idx <= checkpoints[0]:
            return float(progress_at[checkpoints[0]])
        if frame_idx >= checkpoints[-1]:
            return float(progress_at[checkpoints[-1]])

        lo = checkpoints[0]
        hi = checkpoints[-1]
        for i in range(len(checkpoints) - 1):
            if checkpoints[i] <= frame_idx <= checkpoints[i + 1]:
                lo, hi = checkpoints[i], checkpoints[i + 1]
                break

        p_lo = float(progress_at[lo])
        p_hi = float(progress_at[hi])
        if self.query_fill_mode == "hold" or hi == lo:
            return p_lo
        alpha = (frame_idx - lo) / float(hi - lo)
        return p_lo + alpha * (p_hi - p_lo)

    def shape_rewards(
        self,
        images_T6HW: np.ndarray,
        horizon: int = 1,
    ) -> np.ndarray:
        """Return Robometer-only shaped rewards of length T - horizon.

        Default (LIBERO ``use_relative_rewards``): ``weight * (progress_{t+H} - progress_t)``.
        If ``use_relative_rewards=False``: PBRS ``γ·Φ_{t+H} − Φ_t`` with Φ = weight·progress.
        No sparse success/failure term is mixed in.
        """
        frames = self._extract_hwc_frames(images_T6HW)
        T = int(frames.shape[0])
        H = int(horizon)
        n = T - H
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        if not self.is_ready():
            return np.zeros(n, dtype=np.float32)

        checkpoints = self._checkpoint_indices(T, H)
        prefixes = [
            (idx, frames[: idx + 1])
            for idx in checkpoints
        ]
        progress_raw = self._query_progress_batch(prefixes)
        # Raw progress in [0, 1] at checkpoint frames (interpolated between queries).
        progress_at = {idx: float(p) for idx, p in progress_raw.items()}

        out = np.empty(n, dtype=np.float32)
        for t in range(n):
            p_t = self._interpolate_progress(t, progress_at, checkpoints)
            p_next = self._interpolate_progress(t + H, progress_at, checkpoints)
            if self.use_relative_rewards:
                out[t] = np.float32(
                    self.reward_weight * (p_next - p_t)
                )
            else:
                phi_t = self.reward_weight * p_t
                phi_next = self.reward_weight * p_next
                out[t] = np.float32(self.gamma_pbrs * phi_next - phi_t)
        return out
