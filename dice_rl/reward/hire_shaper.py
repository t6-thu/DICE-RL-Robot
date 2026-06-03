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
        # Frame-selection rules:
        online_success_frames="all",        # "all" or an int — frames per online success
        online_failure_frames: int = 1,     # last N frames per online failure (paper: 1)
        expert_frame_stride: int = 5,       # sub-sample offline expert (every Nth frame)
        # Two independent FIFO caps per camera:
        max_pos_buffer_size: int = 4096,    # positive buffer: offline expert + online success FIFO
        max_neg_buffer_size: int = 10,      # negative buffer: recent online failures (small FIFO)
        # NEW: Split positive buffer into expert and online; control sampling
        # proportion between them at HiRE-reward time.
        # online_pos_ratio = 1.0 → 100 % of sampled positives from online success
        #                          (matches the empirical "3.2× sharper Δ" finding —
        #                           offline expert demos suffer a visual domain gap
        #                           with online frames and contribute near-constant
        #                           sim_pos, masking the success/failure signal).
        # online_pos_ratio = 0.0 → 100 % from offline expert (original behavior).
        # online_pos_ratio = r   → ⌊K*r⌋ from online, K-⌊K*r⌋ from expert.
        # If one pool is empty, all K samples come from the other.
        online_pos_ratio: float = 1.0,
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
        self.max_pos_buffer_size   = int(max_pos_buffer_size)
        self.max_neg_buffer_size   = int(max_neg_buffer_size)
        self.online_pos_ratio      = max(0.0, min(1.0, float(online_pos_ratio)))
        self.encode_batch_size     = int(encode_batch_size)

        # Per-camera FIFO buffers.
        # SPLIT positive buffer: expert vs online-success kept separate so that
        # sampling can be drawn from either pool independently. Both still cap
        # at `max_pos_buffer_size` per pool (so the total positive capacity is
        # 2 × max_pos_buffer_size, but the original single-pool semantics are
        # preserved when online_pos_ratio=0.0).
        self.pos_buffer_expert: Dict[str, torch.Tensor] = {}
        self.pos_buffer_online: Dict[str, torch.Tensor] = {}
        self.neg_buffer:        Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """True iff at least one camera has any embeddings in any positive/negative buffer."""
        for cam in self.cameras:
            for buf in (self.pos_buffer_expert, self.pos_buffer_online, self.neg_buffer):
                t = buf.get(cam)
                if t is not None and t.numel() > 0:
                    return True
        return False

    def _pos_buffer_for_sampling(self, cam: str) -> Optional[torch.Tensor]:
        """Sample-time positive pool for one camera, respecting online_pos_ratio.

        Returns up to K vectors stacked, drawn ratio-wise from online vs expert.
        Falls back to whichever pool has data if the other is empty. Returns
        None only if BOTH pools are empty for this camera.
        """
        exp = self.pos_buffer_expert.get(cam)
        onl = self.pos_buffer_online.get(cam)
        has_exp = exp is not None and exp.numel() > 0
        has_onl = onl is not None and onl.numel() > 0
        if not has_exp and not has_onl:
            return None
        if not has_exp:
            return self._sample_buffer(onl)
        if not has_onl:
            return self._sample_buffer(exp)
        K = self.sample_K
        K_onl = int(round(K * self.online_pos_ratio))
        K_exp = K - K_onl
        parts = []
        if K_onl > 0:
            n_onl = min(K_onl, onl.shape[0])
            idx = torch.randperm(onl.shape[0], device=self.device)[:n_onl]
            parts.append(onl[idx])
        if K_exp > 0:
            n_exp = min(K_exp, exp.shape[0])
            idx = torch.randperm(exp.shape[0], device=self.device)[:n_exp]
            parts.append(exp[idx])
        return torch.cat(parts, dim=0) if parts else None

    @torch.no_grad()
    def _append_to_buffer(self, buf: Dict[str, torch.Tensor],
                          camera: str, feats_NPD: torch.Tensor,
                          cap: int) -> None:
        """FIFO append into `buf[camera]` and truncate to keep at most `cap` entries."""
        feats_NPD = feats_NPD.detach().to(self.device)
        if camera in buf and buf[camera].numel() > 0:
            buf[camera] = torch.cat([buf[camera], feats_NPD], dim=0)
        else:
            buf[camera] = feats_NPD
        if buf[camera].shape[0] > cap:
            buf[camera] = buf[camera][-cap:]

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
        """Add frames from one online episode.

        * success → `online_success_frames` frames appended to POSITIVE buffer
                    (FIFO with cap=`max_pos_buffer_size`; will gradually push out
                    the oldest entries, including offline expert seeds).
        * failure → `online_failure_frames` frames appended to NEGATIVE buffer
                    (FIFO with cap=`max_neg_buffer_size`, small by design).
        """
        T = int(images_T6HW_f01.shape[0])
        if T == 0:
            return 0
        if success:
            n = T if self.online_success_frames == "all" \
                  else min(int(self.online_success_frames), T)
            target_buf = self.pos_buffer_online   # ← online success goes to ONLINE pool only
            cap        = self.max_pos_buffer_size
        else:
            n = min(int(self.online_failure_frames), T)
            target_buf = self.neg_buffer
            cap        = self.max_neg_buffer_size
        frame_idx = np.arange(T - n, T, dtype=np.int64)
        feats_base, feats_wrist = self._encode_episode_frames(
            images_T6HW_f01, frame_idx)
        self._append_to_buffer(target_buf, "base",  feats_base,  cap=cap)
        self._append_to_buffer(target_buf, "wrist", feats_wrist, cap=cap)
        return n

    @torch.no_grad()
    def build_from_expert_npz(self, expert_npz_path: str,
                              curation_path: Optional[str] = None) -> int:
        """Encode strided frames from offline expert and add to positive buffer.

        Sampling pattern (per episode):
          Start at the LAST frame (terminal = the "success state") and step
          BACKWARD by `expert_frame_stride` toward the beginning. The resulting
          per-episode indices are then sorted ascending for batch-encoding.

          Example with stride=5 on a 200-frame episode:
              picked = {199, 194, 189, ..., 4}   ← anchored at terminal

          This guarantees:
          * the terminal "goal state" is always sampled (was previously
            absent when stride didn't divide L-1)
          * sampled frames are phase-aligned to the success endpoint across
            episodes of different lengths, which gives Φ a sharper "near-goal"
            signal at training time.

        If `curation_path` points to a JSON produced by `scripts/curate_expert.py`
        with an "include" list of episode indices, only those trajectories are
        used. Otherwise ALL trajectories are used.
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

        # Need traj_lengths for the backward-from-terminal stride below.
        d_npz = np.load(expert_npz_path)
        traj_lengths = d_npz["traj_lengths"].astype(int)
        ep_starts    = np.concatenate([[0], np.cumsum(traj_lengths)])
        n_eps        = int(len(traj_lengths))

        # Determine which episodes to include.
        include_eps: List[int]
        if curation_path and os.path.isfile(curation_path):
            import json
            with open(curation_path) as f:
                cur = json.load(f)
            cur_ids = sorted(set(int(x) for x in cur.get("include", [])))
            if not cur_ids:
                log.warning("HiRE: curation %s has empty `include` list — "
                            "falling back to ALL trajectories", curation_path)
                include_eps = list(range(n_eps))
            else:
                include_eps = [ep for ep in cur_ids if 0 <= ep < n_eps]
                log.info("HiRE: curation %s → using %d/%d expert episodes",
                         os.path.basename(curation_path),
                         len(include_eps), n_eps)
        else:
            include_eps = list(range(n_eps))

        # Per-episode backward-stride from terminal (e - 1), spanning back to s.
        stride = self.expert_frame_stride
        idx_chunks = []
        for ep in include_eps:
            s = int(ep_starts[ep])
            e = int(ep_starts[ep + 1])     # exclusive end
            if e <= s:
                continue
            idx = np.arange(e - 1, s - 1, -stride)   # e-1, e-1-stride, ...
            idx = idx[::-1]                          # ascending order
            idx_chunks.append(idx)
        indices = (np.concatenate(idx_chunks).astype(np.int64) if idx_chunks
                   else np.arange(0, T_total, stride, dtype=np.int64))

        cap = self.max_pos_buffer_size
        if len(indices) > cap:
            rng = np.random.default_rng(0)
            indices = np.sort(rng.choice(indices, cap, replace=False))
        log.info("HiRE: encoding %d offline-expert frames (of %d total, stride=%d) → pos_buffer_expert (cap=%d)…",
                 len(indices), T_total, self.expert_frame_stride, cap)

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
            self._append_to_buffer(self.pos_buffer_expert, "base",
                                   self.encoder.encode(base),  cap=cap)
            self._append_to_buffer(self.pos_buffer_expert, "wrist",
                                   self.encoder.encode(wrist), cap=cap)
            added += sel.shape[0]
        log.info("HiRE: pos_buffer_expert after expert: %s",
                 {c: tuple(self.pos_buffer_expert[c].shape)
                  for c in self.cameras if c in self.pos_buffer_expert})
        return added

    @torch.no_grad()
    def build_initial_buffers_from_dir(self, episode_dir: str) -> None:
        """Scan all `episode_*.npz` under `episode_dir`, append to pos/neg buffers."""
        if not episode_dir or not os.path.isdir(episode_dir):
            log.info("HiRE: no online-episode seed dir at %s — skipping", episode_dir)
            return
        paths = sorted(glob.glob(os.path.join(episode_dir, "episode_*.npz")))
        if not paths:
            log.info("HiRE: no episodes under %s — buffers stay empty for now", episode_dir)
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
        log.info("HiRE: pos_buffer_online += %d frames from %d success episodes (FIFO cap=%d)",
                 n_pos_f, n_pos_ep, self.max_pos_buffer_size)
        log.info("HiRE: neg_buffer += %d frames from %d failure episodes (FIFO cap=%d)",
                 n_neg_f, n_neg_ep, self.max_neg_buffer_size)
        for cam in self.cameras:
            pe  = self.pos_buffer_expert.get(cam)
            po  = self.pos_buffer_online.get(cam)
            neg = self.neg_buffer.get(cam)
            log.info("HiRE: camera=%s  pos_expert=%s  pos_online=%s  neg=%s  (online_pos_ratio=%.2f)",
                     cam,
                     tuple(pe.shape)  if pe  is not None else None,
                     tuple(po.shape)  if po  is not None else None,
                     tuple(neg.shape) if neg is not None else None,
                     self.online_pos_ratio)

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
        # Re-sample K from each buffer once per episode (paper does so per step
        # but per-episode sampling is much faster and statistically similar).
        # Positives are drawn ratio-wise from online-success vs offline-expert
        # (see _pos_buffer_for_sampling — online_pos_ratio controls the mix).
        pos_b = self._pos_buffer_for_sampling("base")
        neg_b = self._sample_buffer(self.neg_buffer.get("base"))
        pos_w = self._pos_buffer_for_sampling("wrist")
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
        """Return per-transition shaped rewards of length T-1.

        For transition t = (s_t, a_t → s_{t+1}), the shaped reward is
            r̃_t = r_t + γ·Φ(s_{t+1}) − Φ(s_t)
        where r_t is the env-emitted reward upon arriving at s_{t+1}
        (i.e. `sparse_rewards[t+1]` in the env runner's state-aligned array).

        Terminal boundary (t = T-2, the last stored transition):
            Φ(s_{T-1}) is forced to 0 per standard PBRS (Ng et al. 1999).
            Without this, the critic sees an extra γ·Φ(s_terminal) in the
            last-step reward that is never cancelled by a next-state Q value
            (done=True zeros out the bootstrap), breaking policy invariance.
        """
        T = int(images_T6HW_f01.shape[0])
        sparse = np.asarray(sparse_rewards, dtype=np.float32)
        if T <= 1:
            return sparse[1:T].copy()
        if not self.is_ready():
            return sparse[1:T].copy()

        phi = self._compute_potential(images_T6HW_f01)             # (T,)
        out = np.empty(T - 1, dtype=np.float32)
        for t in range(T - 1):
            phi_next = 0.0 if t == T - 2 else phi[t + 1]          # Φ(terminal) = 0
            r_dense = self.gamma_pbrs * phi_next - phi[t]
            out[t] = sparse[t + 1] + r_dense
        return out
