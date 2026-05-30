"""HiRE (Hindsight Reward Editing) reward shaper for YAM real-robot DICE-RL.

Ports the contrastive-prompt PBRS recipe from the simulation codebase
`dice-rl` (commit `nhy_fixedbuffer`):

    Φ(s) = reward_weight · ( sim_pos(s) − contrastive_lambda · sim_neg(s) )
    r_dense_t = γ_pbrs · Φ(s_{t+1}) − Φ(s_t)            (standard PBRS)
    r_final   = r_sparse + r_dense

`sim_X(s)` is a logsumexp-smooth-max over `K` cosine-similarities between
DINOv2 patch embeddings of the current observation and a buffer of
positive / negative reference embeddings. Similarities are computed
independently for the base and wrist cameras and summed.

Buffers
-------
* **Positive**  (sharp logsumexp, β_pos ≈ 10): *all* frames from
    - offline expert demos (`train.npz`, sub-sampled by `expert_frame_stride`)
    - online *success* episodes (every frame)
* **Negative**  (smooth logsumexp, β_neg ≈ 1): *last frame only* of online
    *failure* episodes.

Both buffers are populated once at startup from
(offline expert npz)  +  (a "history" directory of past online episodes),
and the negative buffer keeps growing as new online failures arrive.
"""

from __future__ import annotations
import glob
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

log = logging.getLogger(__name__)

# Pin the DINOv2 hub commit so behaviour matches the dice-rl reference.
_DINOV2_COMMIT = "b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8"


class DinoV2Encoder(nn.Module):
    """DINOv2 ViT-S/14 patch-token encoder (frozen).

    Accepts (B, 3, H, W) float images in [0, 1] (will be resized + ImageNet-
    normalised). Returns (B, P, D) patch tokens.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        # ImageNet stats on 0-1 scale (YAM stores images in [0, 1]).
        self.transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])
        self.encoder = self._load()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval().to(self.device)

    def _load(self):
        local = os.environ.get("DINO_REPO_LOCAL", "").strip()
        if local and os.path.isdir(local):
            log.info("Loading DINOv2 from local repo %s", local)
            return torch.hub.load(local, "dinov2_vits14", source="local")
        pinned = f"facebookresearch/dinov2:{_DINOV2_COMMIT}"
        log.info("Loading DINOv2 from %s (one-time download)", pinned)
        try:
            return torch.hub.load(pinned, "dinov2_vits14", trust_repo=True)
        except Exception:
            return torch.hub.load(pinned, "dinov2_vits14",
                                  trust_repo=True, skip_validation=True)

    @torch.no_grad()
    def encode(self, images_b3hw: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) float in [0,1] → (B, P, D) patch tokens."""
        if images_b3hw.dim() == 3:
            images_b3hw = images_b3hw.unsqueeze(0)
        images_b3hw = images_b3hw.to(self.device, non_blocking=True).float()
        x = self.transform(images_b3hw)
        feats = self.encoder.forward_features(x)
        return feats["x_norm_patchtokens"]


class HireRewardShaper:
    """Contrastive-prompt PBRS dense reward shaping for YAM episodes.

    Notes
    -----
    * Per-camera buffers (`pos_buffer[cam]`, `neg_buffer[cam]`) hold
      patch-token embeddings of shape (N, P, D).
    * `shape_rewards(R, images)` returns a new reward array that adds the
      PBRS dense term on top of the sparse `R`.
    * Buffers can be grown online via `add_episode_to_buffer`.
    """

    def __init__(
        self,
        encoder: DinoV2Encoder,
        cameras: List[str] = ("base", "wrist"),
        reward_weight: float = 1.0,
        contrastive_lambda: float = 0.1,
        logsumexp_beta_pos: float = 10.0,   # sharp max for positives
        logsumexp_beta_neg: float = 1.0,    # smooth max ≈ mean for negatives
        gamma_pbrs: float = 0.99,
        sample_K: int = 64,
        # Frame-selection rules (replace the old `frames_per_episode_for_buffer`):
        online_success_frames="all",        # "all" or an int — frames per online success
        online_failure_frames: int = 1,     # last N frames per online failure (paper: 1)
        expert_frame_stride: int = 5,       # sub-sample offline expert (every Nth frame)
        max_buffer_size: int = 4096,
        encode_batch_size: int = 32,
    ) -> None:
        self.encoder = encoder
        self.device  = encoder.device
        self.cameras = list(cameras)
        self.reward_weight      = float(reward_weight)
        self.contrastive_lambda = float(contrastive_lambda)
        self.logsumexp_beta_pos = float(logsumexp_beta_pos)
        self.logsumexp_beta_neg = float(logsumexp_beta_neg)
        self.gamma_pbrs         = float(gamma_pbrs)
        self.sample_K           = int(sample_K)
        self.online_success_frames = online_success_frames   # "all" or int
        self.online_failure_frames = int(online_failure_frames)
        self.expert_frame_stride   = max(1, int(expert_frame_stride))
        self.max_buffer_size    = int(max_buffer_size)
        self.encode_batch_size  = int(encode_batch_size)

        self.pos_buffer: Dict[str, torch.Tensor] = {}
        self.neg_buffer: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """True iff at least one camera has at least one of pos or neg embeddings.

        With the new buffer recipe, the positive buffer is filled from offline
        expert (always available) while the negative buffer only fills as online
        failures accumulate. We allow HiRE to operate as soon as either side
        has something — the missing side contributes 0 in `_sim_to_targets`.
        """
        for cam in self.cameras:
            pos = self.pos_buffer.get(cam)
            neg = self.neg_buffer.get(cam)
            if (pos is not None and pos.numel() > 0) or \
               (neg is not None and neg.numel() > 0):
                return True
        return False

    @torch.no_grad()
    def _append_to_buffer(self, buf: Dict[str, torch.Tensor],
                          camera: str, feats_NPD: torch.Tensor) -> None:
        feats_NPD = feats_NPD.detach().to(self.device)
        if camera in buf and buf[camera].numel() > 0:
            buf[camera] = torch.cat([buf[camera], feats_NPD], dim=0)
        else:
            buf[camera] = feats_NPD
        # Cap buffer size by keeping the most recent embeddings.
        if buf[camera].shape[0] > self.max_buffer_size:
            buf[camera] = buf[camera][-self.max_buffer_size:]

    @torch.no_grad()
    def _encode_episode_frames(self, images_T6HW_f01: np.ndarray,
                               frame_indices: np.ndarray):
        """Encode selected frames from one episode for both cameras.

        Returns (feats_base_NPD, feats_wrist_NPD).
        """
        sel = images_T6HW_f01[frame_indices]            # (n, 6, H, W)
        base  = torch.from_numpy(sel[:, :3]).float()     # (n, 3, H, W)
        wrist = torch.from_numpy(sel[:, 3:]).float()

        # Encode in mini-batches to keep memory bounded
        def _batched(x):
            outs = []
            for i in range(0, x.shape[0], self.encode_batch_size):
                outs.append(self.encoder.encode(x[i:i + self.encode_batch_size]))
            return torch.cat(outs, dim=0) if outs else torch.empty(0, device=self.device)

        return _batched(base), _batched(wrist)

    @torch.no_grad()
    def add_episode_to_buffer(self, images_T6HW_f01: np.ndarray, success: bool) -> int:
        """Add frames from one online episode to pos (if success) or neg (if failure).

        * success  → uses `online_success_frames` ("all" or int)  → positive buffer
        * failure  → uses `online_failure_frames` (default 1)     → negative buffer
        Returns the number of frames added.
        """
        T = int(images_T6HW_f01.shape[0])
        if T == 0:
            return 0
        if success:
            if self.online_success_frames == "all":
                n = T
            else:
                n = min(int(self.online_success_frames), T)
        else:
            n = min(int(self.online_failure_frames), T)
        frame_idx = np.arange(T - n, T, dtype=np.int64)
        feats_base, feats_wrist = self._encode_episode_frames(
            images_T6HW_f01, frame_idx)
        buf = self.pos_buffer if success else self.neg_buffer
        self._append_to_buffer(buf, "base",  feats_base)
        self._append_to_buffer(buf, "wrist", feats_wrist)
        return n

    @torch.no_grad()
    def build_from_expert_npz(self, expert_npz_path: str) -> int:
        """Encode (strided) frames from offline expert and add to positive buffer.

        Expert npz contains `images` of shape (T_total, 6, H, W) uint8 — all
        treated as successful (positive). Sub-sampled by `expert_frame_stride`.
        Prefers a sidecar `<stem>_images.npy` if present (fast mmap).
        """
        if not os.path.isfile(expert_npz_path):
            log.warning("HiRE: expert npz %s missing — skipping offline positives",
                        expert_npz_path)
            return 0

        sidecar = os.path.splitext(expert_npz_path)[0] + "_images.npy"
        if os.path.isfile(sidecar):
            images = np.load(sidecar, mmap_mode="r")
            log.info("HiRE: using mmap'd sidecar %s for offline expert positives",
                     os.path.basename(sidecar))
        else:
            d = np.load(expert_npz_path)
            images = d["images"]

        T_total = int(images.shape[0])
        indices = np.arange(0, T_total, self.expert_frame_stride)
        if len(indices) > self.max_buffer_size:
            rng = np.random.default_rng(0)
            indices = np.sort(rng.choice(indices, self.max_buffer_size, replace=False))
        log.info("HiRE: encoding %d offline-expert frames (of %d total, stride=%d)…",
                 len(indices), T_total, self.expert_frame_stride)

        added = 0
        bs = self.encode_batch_size
        for i in range(0, len(indices), bs):
            sel_idx = indices[i:i + bs]
            sel = np.asarray(images[sel_idx])         # (b, 6, H, W) uint8
            if sel.dtype == np.uint8:
                sel = sel.astype(np.float32) / 255.0
            else:
                sel = sel.astype(np.float32)
            base  = torch.from_numpy(sel[:, :3])
            wrist = torch.from_numpy(sel[:, 3:])
            self._append_to_buffer(self.pos_buffer, "base",  self.encoder.encode(base))
            self._append_to_buffer(self.pos_buffer, "wrist", self.encoder.encode(wrist))
            added += sel.shape[0]
        log.info("HiRE: positive buffer after expert: %s",
                 {c: tuple(self.pos_buffer[c].shape) for c in self.cameras
                  if c in self.pos_buffer})
        return added

    @torch.no_grad()
    def build_initial_buffers_from_dir(self, episode_dir: str) -> None:
        """Scan all `episode_*.npz` under `episode_dir`, populate both buffers."""
        paths = sorted(glob.glob(os.path.join(episode_dir, "episode_*.npz")))
        if not paths:
            log.warning("HiRE: no episodes under %s — buffers stay empty", episode_dir)
            return
        log.info("HiRE: building positive/negative buffers from %d episodes in %s …",
                 len(paths), episode_dir)
        n_pos_ep, n_neg_ep, n_pos_f, n_neg_f = 0, 0, 0, 0
        for p in paths:
            d = np.load(p)
            r = d.get("rewards", np.zeros(1, dtype=np.float32))
            success = bool(r[-1] > 0.5) if len(r) > 0 else False
            images  = d["images"]   # (T, 6, H, W) float32 [0,1]
            added = self.add_episode_to_buffer(images, success)
            if success:
                n_pos_ep += 1; n_pos_f += added
            else:
                n_neg_ep += 1; n_neg_f += added
        log.info("HiRE: positive buffer  = %d frames from %d success episodes",
                 n_pos_f, n_pos_ep)
        log.info("HiRE: negative buffer  = %d frames from %d failure episodes",
                 n_neg_f, n_neg_ep)
        for cam in self.cameras:
            pos = self.pos_buffer.get(cam)
            neg = self.neg_buffer.get(cam)
            log.info("HiRE: camera=%s  pos=%s  neg=%s", cam,
                     tuple(pos.shape) if pos is not None else None,
                     tuple(neg.shape) if neg is not None else None)

    # ------------------------------------------------------------------
    # Similarity & potential
    # ------------------------------------------------------------------

    def _l2(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, dim=-1)

    @staticmethod
    def _logsumexp_smooth_max(x: torch.Tensor, dim: int, beta: float) -> torch.Tensor:
        return torch.logsumexp(beta * x, dim=dim) / beta

    def _sample_buffer(self, buf: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if buf is None or buf.shape[0] == 0:
            return None
        K = self.sample_K
        if buf.shape[0] <= K:
            return buf
        idx = torch.randperm(buf.shape[0], device=self.device)[:K]
        return buf[idx]

    def _sim_to_targets(self, cur_BPD: torch.Tensor,
                        tgt_KPD: Optional[torch.Tensor],
                        beta: float) -> torch.Tensor:
        """cur (B, P, D) vs tgt (K, P, D) → (B,) logsumexp-smooth-max sim at given beta."""
        if tgt_KPD is None or tgt_KPD.numel() == 0:
            return torch.zeros(cur_BPD.shape[0], device=self.device)
        cur = self._l2(cur_BPD)
        tgt = self._l2(tgt_KPD)
        per_patch = torch.einsum("bpd,kpd->bkp", cur, tgt)
        per_pair  = per_patch.mean(dim=-1)              # (B, K)
        return self._logsumexp_smooth_max(per_pair, dim=-1, beta=beta)

    @torch.no_grad()
    def _compute_potential(self, images_T6HW_f01: np.ndarray) -> np.ndarray:
        """For an episode's image sequence, return Φ as (T,) numpy."""
        T = int(images_T6HW_f01.shape[0])
        if T == 0:
            return np.zeros(0, dtype=np.float32)
        imgs = torch.from_numpy(images_T6HW_f01.astype(np.float32))
        base  = imgs[:, :3]
        wrist = imgs[:, 3:]
        f_base  = self.encoder.encode(base)
        f_wrist = self.encoder.encode(wrist)

        sim_total = torch.zeros(T, device=self.device)
        # Re-sample K from buffer once per episode (paper does so per step but
        # per-episode sampling is much faster and statistically similar).
        pos_b = self._sample_buffer(self.pos_buffer.get("base"))
        neg_b = self._sample_buffer(self.neg_buffer.get("base"))
        pos_w = self._sample_buffer(self.pos_buffer.get("wrist"))
        neg_w = self._sample_buffer(self.neg_buffer.get("wrist"))

        if "base" in self.cameras:
            sp = self._sim_to_targets(f_base, pos_b, beta=self.logsumexp_beta_pos)
            sn = self._sim_to_targets(f_base, neg_b, beta=self.logsumexp_beta_neg)
            sim_total = sim_total + (sp - self.contrastive_lambda * sn)
        if "wrist" in self.cameras:
            sp = self._sim_to_targets(f_wrist, pos_w, beta=self.logsumexp_beta_pos)
            sn = self._sim_to_targets(f_wrist, neg_w, beta=self.logsumexp_beta_neg)
            sim_total = sim_total + (sp - self.contrastive_lambda * sn)

        phi = self.reward_weight * sim_total
        return phi.detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Public: shape an episode's rewards (PBRS)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def shape_rewards(self, sparse_rewards: np.ndarray,
                      images_T6HW_f01: np.ndarray) -> np.ndarray:
        """Return r_sparse + γ·Φ(s_{t+1}) − Φ(s_t)  for t = 0…T−2 (last unchanged)."""
        T = int(images_T6HW_f01.shape[0])
        if T == 0:
            return sparse_rewards.astype(np.float32).copy()
        if not self.is_ready():
            return sparse_rewards.astype(np.float32).copy()

        phi = self._compute_potential(images_T6HW_f01)   # (T,)
        new = sparse_rewards.astype(np.float32).copy()
        for t in range(T - 1):
            r_dense = self.gamma_pbrs * phi[t + 1] - phi[t]
            new[t] = sparse_rewards[t] + r_dense
        # The final transition is not stored in the buffer; leave new[-1] as-is.
        return new
