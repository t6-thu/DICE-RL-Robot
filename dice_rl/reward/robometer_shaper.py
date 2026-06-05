"""Minimal Robometer reward source for YAM RL.

Sends each episode's BASE-camera video to a running Robometer-4B eval server
and uses the raw per-frame progress prediction as the per-transition reward:

    r_t = reward_weight * progress(s_{t+H})

`shape_rewards(images, horizon)` is called once per episode by the replay
buffer's `add_episode`. We POST the full episode (subsampled to
`max_frames`) to the server, receive a per-frame progress curve, linearly
upsample back to T, and return a length-(T-H) array.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from dice_rl.reward.robometer_client import (
    extract_progress_from_outputs,
    health_check,
    make_progress_sample,
    post_evaluate_batch_npy,
    subsample_trajectory_frames,
)

log = logging.getLogger(__name__)


class RobometerRawProgressShaper:
    """Raw Robometer progress as the online per-transition reward (base camera)."""

    def __init__(
        self,
        server_url: str,
        task_instruction: str,
        reward_weight: float = 1.0,
        max_frames: int = 16,
        use_frame_steps: bool = False,
        request_timeout_s: float = 120.0,
    ) -> None:
        self.server_url = server_url
        self.task_instruction = task_instruction
        self.reward_weight = float(reward_weight)
        self.max_frames = int(max_frames)
        self.use_frame_steps = bool(use_frame_steps)
        self.request_timeout_s = float(request_timeout_s)
        self._ready = health_check(server_url)
        if self._ready:
            log.info("Robometer ready @ %s  task=%r  weight=%.3f  (base camera)",
                     server_url, task_instruction, reward_weight)
        else:
            log.error("Robometer NOT reachable @ %s — falling back to sparse",
                      server_url)
        self._episode_counter = 0

    def is_ready(self) -> bool:
        return self._ready

    def _base_frames_uint8(self, images_TCxHxW: np.ndarray) -> np.ndarray:
        """(T, 6, H, W) {float32[0,1] | uint8[0,255]} → (T, H, W, 3) uint8 base cam."""
        x = images_TCxHxW[:, :3]
        x = np.transpose(x, (0, 2, 3, 1))
        if x.dtype != np.uint8:
            x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
        return np.ascontiguousarray(x)

    def shape_rewards(
        self, images_TCxHxW: np.ndarray, horizon: int = 1,
        sparse_rewards: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """One HTTP round-trip per episode.

        Returns array of length T - horizon with r_t = w * progress[t + horizon]
        (reward-on-arrival semantics, matching the buffer's transition shape)."""
        T = int(images_TCxHxW.shape[0])
        H = int(horizon)
        if T <= H:
            return np.zeros(0, dtype=np.float32)
        if not self._ready:
            return np.zeros(T - H, dtype=np.float32)

        frames = self._base_frames_uint8(images_TCxHxW)
        sent   = subsample_trajectory_frames(frames, self.max_frames)
        sample = make_progress_sample(
            frames=sent,
            task=self.task_instruction,
            sample_id=f"yam_ep_{self._episode_counter}",
            subsequence_length=int(sent.shape[0]),
        )
        try:
            outputs = post_evaluate_batch_npy(
                self.server_url, [sample],
                timeout_s=self.request_timeout_s,
                use_frame_steps=self.use_frame_steps,
            )
            progress_sent = extract_progress_from_outputs(outputs, sample_index=0)
        except Exception as e:
            log.warning("Robometer call failed (%s) — using zeros this episode", e)
            return np.zeros(T - H, dtype=np.float32)

        if len(progress_sent) == 0:
            return np.zeros(T - H, dtype=np.float32)
        if len(progress_sent) == T:
            progress_T = np.asarray(progress_sent, dtype=np.float32)
        else:
            n = len(progress_sent)
            x_sent = np.linspace(0.0, T - 1.0, n)
            x_full = np.arange(T, dtype=np.float32)
            progress_T = np.interp(x_full, x_sent, progress_sent).astype(np.float32)

        log.info("Robometer ep#%d  T=%d  progress[0]=%.3f  progress[-1]=%.3f  "
                 "mean=%.3f", self._episode_counter, T,
                 float(progress_T[0]), float(progress_T[-1]),
                 float(progress_T.mean()))
        self._episode_counter += 1
        return (self.reward_weight * progress_T[H : H + (T - H)]).astype(np.float32)
