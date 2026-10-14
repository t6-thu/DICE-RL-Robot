"""Real-Time Chunking (RTC) scheduler for chunked policy deployment.

Hardware-free bookkeeping used by `YAMRLEnvRunner.run_episode`. The control
loop ticks once per frame at `control_hz`; every frame must send exactly one
joint command and record exactly one transition. The scheduler decides, per
frame:

  * which action to send (from the active chunk, or a hold of the previous
    command when no chunk covers this frame),
  * when to launch the next asynchronous inference and which already-committed
    actions to freeze as the inpainting prefix of the new chunk,
  * whether a finished inference result is still usable.

Timeline conventions (match training data: `A[t]` is the command issued at
the same frame the observation `t` was taken):

  * A chunk produced from the observation at frame `k_obs` covers frames
    `k_obs .. k_obs + H - 1`, index `i` ↔ frame `k_obs + i`.
  * Inference is launched at frame `k_obs` right after that frame's command was
    sent and its observation recorded. While it runs, the old chunk keeps
    executing. The actions of the old chunk for frames `k_obs .. k_obs + d`
    are frozen as the prefix (`d` = `delay_steps`, the inference budget in
    frames), so the new chunk agrees with what is being executed and takes over
    seamlessly from frame `k_obs + d + 1`.
  * If inference finishes early, the new chunk may replace the old one at any
    frame inside the prefix (both contain identical actions there).
  * If inference finishes late, frames past the old chunk are served by holding
    the last command and flagged `action_valid=False`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Chunk:
    start_frame: int
    actions: np.ndarray  # (H, D) in the space the executor consumes

    @property
    def end_frame(self) -> int:  # inclusive
        return self.start_frame + len(self.actions) - 1

    def covers(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame

    def action_at(self, frame: int) -> np.ndarray:
        return self.actions[frame - self.start_frame]


@dataclass
class InferenceRequest:
    obs_frame: int
    prefix: np.ndarray  # (n, D), n >= 1, prefix[0] is the command of obs_frame


@dataclass
class RTCStats:
    frames: int = 0
    hold_frames: int = 0
    launches: int = 0
    late_results: int = 0  # result arrived after its whole chunk expired
    discarded_results: int = 0
    early_switches: int = 0  # new chunk adopted before old chunk ran out
    prefix_lengths: list = field(default_factory=list)


class RTCScheduler:
    """Per-frame chunk bookkeeping. All frame indices are integers at control_hz."""

    def __init__(self, action_horizon: int, delay_steps: int) -> None:
        if action_horizon < 2:
            raise ValueError("action_horizon must be >= 2")
        if not (0 <= delay_steps <= action_horizon - 2):
            raise ValueError(
                f"delay_steps must be in [0, action_horizon-2]; got {delay_steps} "
                f"for action_horizon={action_horizon}"
            )
        self.H = int(action_horizon)
        self.delay_steps = int(delay_steps)
        self.chunk: Optional[Chunk] = None
        self.in_flight: Optional[InferenceRequest] = None
        self.stats = RTCStats()

    # ---- per-frame API ----------------------------------------------------

    def reset(self, first_chunk: np.ndarray) -> None:
        self.chunk = Chunk(0, np.asarray(first_chunk))
        self.in_flight = None
        self.stats = RTCStats()

    def action_for_frame(self, frame: int, last_cmd: np.ndarray):
        """Return (action, valid). `valid` is False when holding `last_cmd`."""
        self.stats.frames += 1
        if self.chunk is not None and self.chunk.covers(frame):
            return self.chunk.action_at(frame), True
        self.stats.hold_frames += 1
        return np.asarray(last_cmd), False

    def maybe_launch(self, frame: int, cmd_at_frame: np.ndarray) -> Optional[InferenceRequest]:
        """Call after the command for `frame` was sent and its obs recorded.

        Returns an InferenceRequest when a new inference should start now.
        `cmd_at_frame` is the command actually issued at `frame` (used as the
        1-step prefix when no chunk covers the future).
        """
        if self.in_flight is not None:
            return None
        if self.chunk is not None and self.chunk.covers(frame):
            remaining = self.chunk.end_frame - frame  # frames left after this one
            if remaining > self.delay_steps:
                return None
            # Freeze everything the old chunk still has to execute, starting
            # with the command issued at the obs frame itself.
            prefix = self.chunk.actions[frame - self.chunk.start_frame:].copy()
        else:
            # No chunk covers this frame: we are holding. Condition only on the
            # command actually issued at the obs frame.
            prefix = np.asarray(cmd_at_frame)[None].copy()
        n = len(prefix)
        if n > self.H - 1:
            prefix = prefix[: self.H - 1]
        self.in_flight = InferenceRequest(obs_frame=frame, prefix=prefix)
        self.stats.launches += 1
        self.stats.prefix_lengths.append(len(prefix))
        return self.in_flight

    def on_result(self, frame: int, request: InferenceRequest, actions: np.ndarray) -> bool:
        """Deliver a finished inference at the end of `frame`.

        Returns True if the chunk was adopted (it serves frame+1 onwards).
        """
        if self.in_flight is not request:
            self.stats.discarded_results += 1
            return False
        self.in_flight = None
        new = Chunk(request.obs_frame, np.asarray(actions))
        if new.end_frame < frame + 1:
            # Arrived after every frame it could have served.
            self.stats.late_results += 1
            return False
        if self.chunk is not None and self.chunk.covers(frame + 1):
            self.stats.early_switches += 1
        self.chunk = new
        return True

    def cancel_in_flight(self) -> None:
        self.in_flight = None
