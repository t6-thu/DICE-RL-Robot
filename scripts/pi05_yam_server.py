#!/usr/bin/env python
"""Pi0.5 (openpi) inference server for the YAM SFT checkpoint.

WHY THIS LIVES IN ITS OWN PROCESS
---------------------------------
openpi pins `torch==2.7.1` and `numpy<2`, while this repo's DICE venv runs
`torch==2.11.0+cu128` (required for the RTX 5090 / sm_120) and `numpy>=2.2`
(required by `i2rt`). Installing openpi into the DICE venv would downgrade
torch off the 5090-capable wheel. So we run pi0.5 inference inside RLinf's
own venv as a small ZMQ server and let the DICE env_runner reach it over a
local IPC socket.

USAGE
-----
    /home/bike/Documents/niu/RLinf/.venv/bin/python \
        /home/bike/Documents/niu/DICE-RL-Robot/scripts/pi05_yam_server.py \
        --ckpt /home/bike/Documents/niu/RLinf/models/sft_logs/pi05_yam-two_cam-picknplace+discrete_state_TRUE-20260429-15:02:06/pi05_yam-two_cam-picknplace+discrete_state_TRUE/checkpoints/global_step_4500/actor/model_state_dict/full_weights.pt \
        --norm-stats /home/bike/Documents/niu/RLinf/models/sft_logs/norm_stats.json \
        --endpoint ipc:///tmp/pi05_yam

Request shape (msgpack):
    {
        "base_rgb":  uint8  HxWx3,   # base camera (any resolution; resized internally)
        "wrist_rgb": uint8  HxWx3,   # wrist-mounted camera
        "state":     float32 [7],     # action_{t-1}-approximated proprio
        "prompt":    str,             # natural-language task description
    }

Response:
    {
        "actions": float32 [action_horizon, 7],  # de-normalized YAM actions
        "training_step": int,
    }

The server is intentionally single-threaded; pi0.5 inference is GPU-heavy
and the env_runner only requests one chunk every sparse_execution_horizon
steps, so contention is negligible.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# RLinf's openpi expects these to be importable; this script is meant to be
# invoked with RLinf's venv interpreter.
try:
    from openpi.models import model as _model  # noqa: F401  (probe import)
except ImportError as e:  # pragma: no cover - environment-specific
    sys.stderr.write(
        "[pi05_yam_server] openpi import failed. Run this script with RLinf's "
        "venv: `/home/bike/Documents/niu/RLinf/.venv/bin/python`.\n"
        f"Underlying error: {e}\n"
    )
    raise

# RLinf's pi0 wrapper (adds value head + RL-friendly sampling).
sys.path.insert(0, "/home/bike/Documents/niu/RLinf")
from rlinf.models.embodiment.openpi.openpi_action_model import (  # type: ignore
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)
from rlinf.models.embodiment.openpi.policies import yam_policy  # type: ignore

log = logging.getLogger("pi05_yam_server")
logging.basicConfig(level=logging.INFO, format="[%(name)s %(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _strip_actor_prefix(state_dict: dict) -> dict:
    """RLinf SFT checkpoints sometimes ship with an `actor.` prefix; strip it."""
    if all(not k.startswith("actor.") for k in state_dict):
        return state_dict
    return {k.removeprefix("actor."): v for k, v in state_dict.items()}


def build_model(device: torch.device) -> OpenPi0ForRLActionPrediction:
    """Build the SFT-trained pi05_yam model.

    The flags here are inferred from the SFT run name
    `pi05_yam-two_cam-picknplace+discrete_state_TRUE`:
        - pi05 = True
        - num_images_in_input = 2 (base + wrist)
        - discrete_state_input = True (state tokenised into prompt)
        - add_value_head = True, value_after_vlm = True (we see value_head.mlp
          with input dim 2048 in the checkpoint).
    """
    config = OpenPi0Config(
        config_name="pi05_yam",
        pi05=True,
        num_images_in_input=2,
        discrete_state_input=True,
        action_horizon=10,
        action_dim=32,         # internal padded action dim used by pi0/pi05
        action_env_dim=7,      # YAM 7D action (6 joint deltas + 1 absolute gripper)
        add_value_head=True,
        value_after_vlm=True,
    )
    model = OpenPi0ForRLActionPrediction(config)
    model.to(device)
    model.eval()
    return model


def load_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    log.info("loading checkpoint: %s (%.2f GB)", ckpt_path, Path(ckpt_path).stat().st_size / 1e9)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint root is {type(state).__name__}, expected dict")
    state = _strip_actor_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    log.info("loaded weights: %d missing, %d unexpected", len(missing), len(unexpected))
    if missing:
        log.warning("missing keys (first 10): %s", missing[:10])
    if unexpected:
        log.warning("unexpected keys (first 10): %s", unexpected[:10])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class YamPi05Inference:
    """Wraps OpenPi0ForRLActionPrediction with the YAM input/output transforms
    and the action-normalization stats used during SFT."""

    def __init__(
        self,
        model: OpenPi0ForRLActionPrediction,
        norm_stats_path: str,
        device: torch.device,
    ) -> None:
        self.model = model
        self.device = device
        with open(norm_stats_path, "r") as f:
            norm = json.load(f)["norm_stats"]
        self._state_mean = np.asarray(norm["state"]["mean"], dtype=np.float32)
        self._state_std = np.asarray(norm["state"]["std"], dtype=np.float32)
        self._action_mean = np.asarray(norm["actions"]["mean"], dtype=np.float32)
        self._action_std = np.asarray(norm["actions"]["std"], dtype=np.float32)
        self._yam_inputs = yam_policy.YamInputs(
            model_type=_model.ModelType.PI0, state_dim=7
        )
        self._yam_outputs = yam_policy.YamOutputs(action_dim=7)

    def __call__(
        self,
        base_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        # Normalize state with SFT stats (action_{t-1} approximation).
        state_norm = (state.astype(np.float32) - self._state_mean) / (self._state_std + 1e-8)

        # Run the data transform pipeline that SFT used.
        record = {
            "observation/image": base_rgb,
            "observation/wrist_image": wrist_rgb,
            "observation/state": state_norm,
            "prompt": prompt,
        }
        record = self._yam_inputs(record)

        # Build batched torch tensors expected by openpi.
        def _stack_image(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(np.asarray(arr, dtype=np.uint8))
            return t.unsqueeze(0)  # (1, H, W, C); from_dict handles uint8 -> [-1, 1].

        batched = {
            "image": {k: _stack_image(v) for k, v in record["image"].items()},
            "image_mask": {
                k: torch.tensor([bool(v)]) for k, v in record["image_mask"].items()
            },
            "state": torch.from_numpy(record["state"]).unsqueeze(0).to(torch.float32),
        }
        if "prompt" in record:
            # openpi tokenises prompts internally during preprocessing; we
            # pass the raw string here for tokenisation by the model. The
            # actual tokenisation is done by _preprocess_observation in
            # pi0_pytorch.PI0Pytorch.
            batched["prompt"] = [record["prompt"]]

        # Move to device.
        def _to_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(self.device)
            if isinstance(obj, dict):
                return {k: _to_device(v) for k, v in obj.items()}
            return obj

        batched = _to_device(batched)

        # Build the Observation dataclass and sample actions.
        from openpi.models.model import Observation

        observation = Observation.from_dict(batched)
        with torch.inference_mode():
            actions = self.model.sample_actions(self.device, observation, num_steps=10)
        actions = actions.detach().to(torch.float32).cpu().numpy()  # (1, T, 32)

        # Trim padded dims to the YAM 7-D action.
        out = self._yam_outputs({"actions": actions[0]})  # (T, 7)
        actions_yam_norm = np.asarray(out["actions"], dtype=np.float32)

        # De-normalize.
        actions_yam = actions_yam_norm * (self._action_std + 1e-8) + self._action_mean
        return actions_yam.astype(np.float32)


# ---------------------------------------------------------------------------
# ZMQ server
# ---------------------------------------------------------------------------


def serve(infer: YamPi05Inference, endpoint: str, training_step: int) -> None:
    import msgpack
    import msgpack_numpy
    import zmq

    msgpack_numpy.patch()
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(endpoint)
    log.info("listening on %s", endpoint)

    while True:
        try:
            payload = sock.recv()
        except KeyboardInterrupt:
            log.info("shutting down")
            return
        try:
            req = msgpack.unpackb(payload, raw=False)
            t0 = time.monotonic()
            actions = infer(
                base_rgb=req["base_rgb"],
                wrist_rgb=req["wrist_rgb"],
                state=req["state"],
                prompt=req["prompt"],
            )
            dt_ms = (time.monotonic() - t0) * 1000.0
            log.info("served chunk shape=%s in %.1f ms", actions.shape, dt_ms)
            response = {"actions": actions, "training_step": training_step, "latency_ms": dt_ms}
        except Exception as e:
            log.exception("inference failed")
            response = {"error": str(e)}
        sock.send(msgpack.packb(response, use_bin_type=True))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path to full_weights.pt")
    parser.add_argument(
        "--norm-stats",
        required=True,
        help="Path to norm_stats.json with action + state mean/std.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--endpoint", default="ipc:///tmp/pi05_yam")
    parser.add_argument(
        "--training-step",
        type=int,
        default=4500,
        help="SFT step number to report back in responses.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build the model, load weights, optionally run a single forward "
        "pass on a synthetic observation, then exit. Use to validate "
        "checkpoint + venv before launching the server proper.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    log.info("device=%s", device)

    model = build_model(device)
    load_weights(model, args.ckpt)
    infer = YamPi05Inference(model, args.norm_stats, device)

    if args.smoke_test:
        log.info("smoke-test: running a single forward pass on dummy obs")
        base = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        wrist = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        state = np.zeros(7, dtype=np.float32)
        actions = infer(base, wrist, state, "pick up the cup")
        log.info("smoke-test OK; actions shape=%s mean=%.4f std=%.4f",
                 actions.shape, actions.mean(), actions.std())
        return

    serve(infer, args.endpoint, args.training_step)


if __name__ == "__main__":
    main()
