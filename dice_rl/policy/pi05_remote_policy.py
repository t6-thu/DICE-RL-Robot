"""Client for the pi0.5 YAM inference server (`scripts/pi05_yam_server.py`).

The server runs in RLinf's venv (it owns the openpi + torch-2.7 stack); this
client runs in the DICE venv (torch-2.11 for the 5090). Both sit on the same
machine and talk over a local IPC socket.

Typical usage from `dice_rl.env_runner.rl_finetuning_env_runner`:

    from dice_rl.policy.pi05_remote_policy import Pi05RemotePolicy

    policy = Pi05RemotePolicy(endpoint="ipc:///tmp/pi05_yam", prompt="pick up the cup")
    action_chunk = policy.predict_action(base_rgb, wrist_rgb, state_7d)

The returned chunk has shape `(action_horizon, 7)` and is already de-normalized
to YAM joint-delta space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Pi05RemotePolicy:
    endpoint: str = "ipc:///tmp/pi05_yam"
    prompt: str = ""
    timeout_ms: int = 30_000

    def __post_init__(self) -> None:
        import msgpack
        import msgpack_numpy
        import zmq

        msgpack_numpy.patch()
        self._msgpack = msgpack
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._sock.connect(self.endpoint)

    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

    def predict_action(
        self,
        base_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
        prompt: Optional[str] = None,
    ) -> np.ndarray:
        """Send one observation; receive an (action_horizon, 7) chunk."""
        if base_rgb.dtype != np.uint8 or wrist_rgb.dtype != np.uint8:
            raise ValueError("base_rgb and wrist_rgb must be uint8 HxWx3")
        req = {
            "base_rgb": base_rgb,
            "wrist_rgb": wrist_rgb,
            "state": np.asarray(state, dtype=np.float32),
            "prompt": prompt if prompt is not None else self.prompt,
        }
        self._sock.send(self._msgpack.packb(req, use_bin_type=True))
        payload = self._sock.recv()
        resp = self._msgpack.unpackb(payload, raw=False)
        if "error" in resp:
            raise RuntimeError(f"pi05 server error: {resp['error']}")
        return np.asarray(resp["actions"], dtype=np.float32)

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:
            pass
