"""DICE-RL Learner for YAM joint-space diffusion policy.

Architecture summary
--------------------
The learner implements RLPD (RL with Prior Data):

  Every update_every_x_episode new rollouts:
    for _ in range(gradient_steps):
        batch = replay_buffer.sample(batch_size, expert_ratio)
        # batch mixes 50% expert (BC training data) + 50% online rollouts
        critic_loss = TD3-like ensemble soft Q-learning on the batch
        actor_loss  = DDPO: maximise Q(s, f(s)) where f is the residual actor
        update target critics (Polyak average, tau=0.01)

Key hyperparams (from rl_finetuning_config.py)
----------------------------------------------
  num_episodes_before_first_training: 20
      → collect 20 pure-BC rollouts before any gradient steps
  gradient_steps: 2000
      → 2000 actor+critic updates per training round
  update_every_x_episode: 10
      → one training round per 10 new rollouts
  batch_size: 256
      → 128 expert + 128 online per update
  adaptive_expert_ratio: 0.7 → 0.2 over 30000 steps
      → start with more expert data (stable early), anneal towards more online
"""

from __future__ import annotations
import glob
import logging
import os
import pickle
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dice_rl.communication.learner_node import Learner
from dice_rl.model.distill_rl import DistilledActor, DistilledCritic
from dice_rl.replay_buffer.yam_replay_buffer import YAMReplayBuffer
from utils.model_io import load_policy

log = logging.getLogger(__name__)


class YAMRLLearner:
    """Learns a residual RL actor on top of a frozen BC diffusion policy."""

    def __init__(
        self,
        # Policy
        pretrained_policy_ckpt: str,
        # Replay buffer
        expert_npz_path: str,
        online_data_dir: str,
        # Network dims (inferred from policy if None)
        actor_hidden_dims: list = None,
        critic_hidden_dims: list = None,
        # Training
        num_episodes_before_first_training: int = 20,
        gradient_steps: int = 2000,
        update_every_x_episode: int = 10,
        batch_size: int = 256,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        action_dim: int = 7,
        # RLPD
        use_rlpd: bool = True,
        expert_ratio: float = 0.5,
        use_adaptive_expert_ratio: bool = True,
        adaptive_expert_ratio_start: float = 0.7,
        adaptive_expert_ratio_end: float = 0.2,
        adaptive_expert_ratio_steps: int = 30000,
        # RL algorithm
        gamma: float = 0.99,
        tau: float = 0.01,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        bc_loss_weight: float = 140.0,
        critic_ensemble_size: int = 5,
        max_grad_norm: float = 1.0,
        # Multi-sample stabilisers (from original DICE-RL-Robot)
        num_next_noise_samples: int = 4,        # K for target Q averaging
        num_multi_z_for_actor_loss: int = 8,    # K for actor loss averaging
        use_q_normalization: bool = True,       # divide q_loss by mean(|Q|)
        disable_q_loss_for_expert_data: bool = True,  # mask Q-loss on expert samples
        # BC loss filter (matches original distill_rl.py actor_loss post-warmup branch)
        use_soft_q_filtering: bool = False,
        q_filtering_warmup_steps: int = 25000,
        # HiRE — Hindsight Reward Editing (contrastive + PBRS dense reward)
        use_hire_reward: bool = False,
        hire_init_dir: str = None,                # past online episodes to seed pos/neg
        hire_expert_curation_path: str = None,    # JSON listing which expert eps to include
        hire_reward_weight: float = 1.0,
        hire_contrastive_lambda: float = 0.1,
        hire_logsumexp_beta_pos: float = 10.0,    # sharp max for positive (goal-like)
        hire_logsumexp_beta_neg: float = 1.0,     # smooth max ≈ mean for negative
        hire_gamma_pbrs: float = 0.99,
        hire_sample_K: int = 64,
        hire_online_success_frames="all",         # paper: all frames of online success
        hire_online_failure_frames: int = 1,      # paper: last frame of online failure
        hire_expert_frame_stride: int = 5,        # subsample offline expert
        hire_max_pos_buffer_size: int = 4096,          # positive FIFO (offline + online share)
        hire_max_neg_buffer_size: int = 10,            # most-recent online failures (FIFO)
        # Reward-recipe switch:
        #   False (default) — online success uses HiRE shaped reward (full method)
        #   True            — online success reverts to sparse reward (offline-style)
        use_sparse_for_online_success: bool = False,
        # ZMQ
        network_server_endpoint: str = "ipc:///tmp/feeds/rl_weights",
        network_weight_topic: str = "rl_network_weights_topic",
        transitions_server_endpoint: str = "ipc:///tmp/feeds/rl_transitions",
        transitions_topic: str = "rl_transitions_topic",
        # Misc
        device: str = "cuda",
        rl_checkpoint_dir: str = None,
    ) -> None:
        self.device = torch.device(device)
        self.gradient_steps = gradient_steps
        self.update_every_x_episode = update_every_x_episode
        self.num_episodes_before_first_training = num_episodes_before_first_training
        self.batch_size = batch_size
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.bc_loss_weight = bc_loss_weight
        self.max_grad_norm = max_grad_norm
        self.num_next_noise_samples = num_next_noise_samples
        self.num_multi_z_for_actor_loss = num_multi_z_for_actor_loss
        self.use_q_normalization = use_q_normalization
        self.disable_q_loss_for_expert_data = disable_q_loss_for_expert_data
        self.use_soft_q_filtering = use_soft_q_filtering
        self.q_filtering_warmup_steps = q_filtering_warmup_steps
        self.use_rlpd = use_rlpd
        self.expert_ratio = expert_ratio
        self.use_adaptive = use_adaptive_expert_ratio
        self.ratio_start = adaptive_expert_ratio_start
        self.ratio_end = adaptive_expert_ratio_end
        self.ratio_steps = adaptive_expert_ratio_steps
        self.rl_checkpoint_dir = rl_checkpoint_dir
        if rl_checkpoint_dir:
            os.makedirs(rl_checkpoint_dir, exist_ok=True)

        # ---- load frozen BC policy ----
        log.info("Loading pretrained BC policy from %s", pretrained_policy_ckpt)
        self.bc_policy, shape_meta, cfg = load_policy(pretrained_policy_ckpt, device)
        self.bc_policy.eval()
        for p in self.bc_policy.parameters():
            p.requires_grad = False
        obs_feature_dim = self.bc_policy.obs_feature_dim
        log.info("BC policy obs_feature_dim=%d", obs_feature_dim)

        # ---- residual actor ----
        ah = actor_hidden_dims or [1024, 1024, 1024]
        ch = critic_hidden_dims or [1024, 1024, 1024]
        self._actor_hidden_dims = ah
        self.actor = DistilledActor(
            obs_dim=obs_feature_dim,
            action_dim=action_dim,
            cond_steps=1,
            horizon_steps=action_horizon,
            hidden_dims=ah,
            activation_type="GELU",
            use_layernorm=True,
        ).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # ---- critic ensemble ----
        self.critics = nn.ModuleList([
            DistilledCritic(
                obs_dim=obs_feature_dim,
                action_dim=action_dim,
                horizon_steps=action_horizon,
                hidden_dims=ch,
            ).to(self.device)
            for _ in range(critic_ensemble_size)
        ])
        self.critic_targets = nn.ModuleList([
            DistilledCritic(
                obs_dim=obs_feature_dim,
                action_dim=action_dim,
                horizon_steps=action_horizon,
                hidden_dims=ch,
            ).to(self.device)
            for _ in range(critic_ensemble_size)
        ])
        for ct, c in zip(self.critic_targets, self.critics):
            ct.load_state_dict(c.state_dict())
            for p in ct.parameters():
                p.requires_grad = False
        self.critic_optim = torch.optim.Adam(
            [p for c in self.critics for p in c.parameters()], lr=critic_lr
        )

        # ---- HiRE reward shaper (optional) ----
        self.use_hire_reward = use_hire_reward
        self.use_sparse_for_online_success = use_sparse_for_online_success
        self.hire_shaper = None
        if self.use_hire_reward:
            from dice_rl.reward.hire_shaper import DinoV2Encoder, HireRewardShaper
            log.info("HiRE enabled — building DINOv2 encoder + contrastive PBRS shaper")
            _dino = DinoV2Encoder(device=device)
            self.hire_shaper = HireRewardShaper(
                encoder=_dino,
                cameras=("base", "wrist"),
                reward_weight=hire_reward_weight,
                contrastive_lambda=hire_contrastive_lambda,
                logsumexp_beta_pos=hire_logsumexp_beta_pos,
                logsumexp_beta_neg=hire_logsumexp_beta_neg,
                gamma_pbrs=hire_gamma_pbrs,
                sample_K=hire_sample_K,
                online_success_frames=hire_online_success_frames,
                online_failure_frames=hire_online_failure_frames,
                expert_frame_stride=hire_expert_frame_stride,
                max_pos_buffer_size=hire_max_pos_buffer_size,
                max_neg_buffer_size=hire_max_neg_buffer_size,
            )
            # 1) Positive buffer ← ALL (strided) frames of offline expert demos.
            #    This is the only seeding HiRE always does — it gives the
            #    shaper "ideal goal-state" reference embeddings before any
            #    online interaction has happened.
            self.hire_shaper.build_from_expert_npz(
                expert_npz_path,
                curation_path=hire_expert_curation_path,
            )
            # 2) Optionally seed pos/neg from a past-run directory of online
            #    episodes (e.g. a previous HiRE checkpoint to resume from).
            #    Skipped when `hire_init_dir is None` so HiRE training starts
            #    from a clean slate, matching the baseline's fresh start.
            if hire_init_dir is not None:
                self.hire_shaper.build_initial_buffers_from_dir(hire_init_dir)
            # 3) Always also scan the current ONLINE_DATA_DIR so a resumed run
            #    picks up its own previously-collected episodes (no-op if empty).
            if (hire_init_dir is None) or (online_data_dir != hire_init_dir):
                self.hire_shaper.build_initial_buffers_from_dir(online_data_dir)
            log.info("HiRE reward-recipe switch: use_sparse_for_online_success=%s",
                     self.use_sparse_for_online_success)

        # ---- replay buffer ----
        self.replay_buffer = YAMReplayBuffer(
            expert_npz_path=expert_npz_path,
            online_data_dir=online_data_dir,
            obs_horizon=obs_horizon,
            action_dim=action_dim,
            action_horizon=action_horizon,
            device=device,
            hire_shaper=self.hire_shaper,
            use_sparse_for_online_success=self.use_sparse_for_online_success,
        )

        # ---- ZMQ communication ----
        self.learner_node = Learner(
            network_server_endpoint=network_server_endpoint,
            network_weight_topic=network_weight_topic,
            transitions_server_endpoint=transitions_server_endpoint,
            transitions_topic=transitions_topic,
            network_weight_expire_time_s=3600,
        )

        self.total_episodes = 0
        self.total_gradient_steps = 0
        self._recent_outcomes: list = []  # track success/failure for reporting
        # Track which on-disk episode files have already been loaded — used by
        # the disk-polling loop in run() to pick up new files without dupes.
        # Reuse the buffer's own loaded-path list to avoid any glob race.
        self._online_data_dir = online_data_dir
        self._loaded_disk_files: set = set(getattr(self.replay_buffer, "loaded_paths", []))
        resumed = self._maybe_resume_checkpoint()
        # Sync total_episodes from disk-restored episodes so the training trigger
        # fires at the right episode count after a restart.
        disk_eps = self.replay_buffer._num_online_episodes
        if disk_eps > self.total_episodes:
            self.total_episodes = disk_eps
            log.info("Synced total_episodes=%d from disk replay buffer", self.total_episodes)
        # Push current weights immediately so env runner can load them on startup.
        if resumed or self.total_gradient_steps > 0:
            self._push_actor_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main learner loop: poll disk for new episode files, train, push weights.

        Uses disk polling (not ZMQ) as the authoritative episode source.
        The env runner saves every episode as `episode_NNNN.npz` to
        `online_data_dir`. The learner picks them up here.
        """
        log.info("Learner running. Polling %s every 2 s for new episodes…",
                 self._online_data_dir)
        log.info("  num_episodes_before_first_training = %d",
                 self.num_episodes_before_first_training)
        log.info("  update_every_x_episode             = %d",
                 self.update_every_x_episode)
        log.info("  gradient_steps per round           = %d", self.gradient_steps)
        log.info("  batch_size (expert+online)         = %d  (expert_ratio=%.1f%%)",
                 self.batch_size, self.expert_ratio * 100)
        log.info("  starting at total_episodes=%d, total_gradient_steps=%d",
                 self.total_episodes, self.total_gradient_steps)

        while True:
            new_count = self._scan_new_disk_episodes()
            if new_count > 0 and self.total_gradient_steps > 0:
                # Re-push current weights so env runner always has the latest actor.
                self._push_actor_weights()

            # ---- decide whether to train ----
            # Cumulative "rounds done vs expected" check — runs every iteration
            # so we catch up after a restart even if no new episodes arrive.
            if self.total_episodes >= self.num_episodes_before_first_training:
                expected_rounds = (
                    (self.total_episodes - self.num_episodes_before_first_training)
                    // self.update_every_x_episode + 1
                )
                done_rounds = self.total_gradient_steps // self.gradient_steps
                if expected_rounds > done_rounds:
                    self._log_success_rate()
                    log.info("Training round (episode %d): expected=%d done=%d → training…",
                             self.total_episodes, expected_rounds, done_rounds)
                    self._train_round()
                    self._push_actor_weights()
                    continue  # immediately re-check (no sleep) after training

            if new_count == 0:
                time.sleep(2.0)

    def _scan_new_disk_episodes(self) -> int:
        """Load any new episode_*.npz files from disk that aren't in the buffer yet.

        Returns the number of episodes ACTUALLY loaded (not just discovered).
        Files that look in-progress (mtime <2 s) are silently retried next pass.
        """
        paths = sorted(glob.glob(os.path.join(self._online_data_dir, "episode_*.npz")))
        new_paths = [p for p in paths if p not in self._loaded_disk_files]
        if not new_paths:
            return 0
        loaded_count = 0
        for p in new_paths:
            # Skip files that look like they're still being written (modified <2 s ago).
            try:
                if time.time() - os.path.getmtime(p) < 2.0:
                    continue  # not added to _loaded_disk_files → retry next pass
                d = np.load(p)
                ep_data = {k: d[k] for k in d.files}
            except Exception as e:
                log.warning("Episode %s not yet readable (%s); will retry", p, e)
                continue
            self.replay_buffer.add_episode(ep_data)
            self.total_episodes += 1
            rewards = ep_data.get("rewards", np.zeros(1, dtype=np.float32))
            success = bool(rewards[-1] > 0.5) if len(rewards) > 0 else False
            self._recent_outcomes.append(success)
            log.info("New episode from disk (#%d) [%s] %s — online transitions: %d",
                     self.total_episodes,
                     "SUCCESS" if success else "FAILURE",
                     os.path.basename(p),
                     self.replay_buffer.num_online_transitions)
            # Grow the HiRE buffer with the *new* episode's last frames
            # (positive if successful, negative if failed). Done AFTER
            # add_episode so the new episode's own shaping uses the
            # buffer state from before this addition.
            if self.hire_shaper is not None and "images" in ep_data:
                self.hire_shaper.add_episode_to_buffer(ep_data["images"], success)
            self._loaded_disk_files.add(p)
            loaded_count += 1
        return loaded_count

    def _log_success_rate(self) -> None:
        n   = len(self._recent_outcomes)
        win = self._recent_outcomes[-self.update_every_x_episode:]
        win_rate  = sum(win) / len(win) * 100 if win else 0.0
        all_rate  = sum(self._recent_outcomes) / n * 100 if n else 0.0
        bar = "█" * sum(win) + "░" * (len(win) - sum(win))
        log.info("=" * 55)
        log.info("  SUCCESS RATE — last %d episodes: %d/%d  (%.0f%%)  [%s]",
                 len(win), sum(win), len(win), win_rate, bar)
        log.info("  SUCCESS RATE — all  %d episodes: %d/%d  (%.0f%%)",
                 n, sum(self._recent_outcomes), n, all_rate)
        log.info("=" * 55)
        self._recent_outcomes.clear()

    def _train_round(self) -> None:
        """Run gradient_steps updates with pre-encoded feature + multi-K BC pool.

        Pre-computes ONCE per round (matches original loss() exactly but vectorised):
          - ViT features for obs / next_obs
          - K_actor=8 (noise, BC) pairs for current_obs  → multi-z actor loss
          - K_critic=4 (noise, BC) pairs for next_obs    → multi-sample target Q
        """
        er = self._current_expert_ratio()

        pool_size  = min(self.gradient_steps * self.batch_size, 10_000)
        encode_bs  = 128                                   # smaller because K-expansion below
        K_actor    = self.num_multi_z_for_actor_loss       # 8
        K_critic   = self.num_next_noise_samples           # 4

        log.info("Pre-encoding pool (N=%d, K_actor=%d, K_critic=%d, mini-batch=%d)…",
                 pool_size, K_actor, K_critic, encode_bs)
        feat_list, fnext_list = [], []
        noise_K_list, noise_next_K_list = [], []
        bc_K_list, bc_next_K_list = [], []
        act_list, rew_list, done_list, is_expert_list = [], [], [], []

        # Reduce BC diffusion steps during RL pool building to match original
        # rl_num_inference_steps=8 (vs 16 used at deployment). 2× faster pool.
        _orig_inf_steps = getattr(self.bc_policy, "num_inference_steps", None)
        if _orig_inf_steps is not None:
            self.bc_policy.num_inference_steps = 8

        n = 0
        while n < pool_size:
            bs = min(encode_bs, pool_size - n)
            raw = self.replay_buffer.sample(bs, expert_ratio=er)
            if not raw:
                break
            with torch.no_grad():
                feat      = self._encode_obs(raw["obs"])          # (bs, D)
                feat_next = self._encode_obs(raw["next_obs"])     # (bs, D)
                # K-expanded noise and BC for current state
                noise_K = torch.randn(bs, K_actor, self.action_horizon, self.action_dim, device=self.device)
                feat_K  = feat.unsqueeze(1).expand(-1, K_actor, -1).reshape(bs * K_actor, -1)
                bc_K    = self.bc_policy.predict_action_from_features(
                    sparse_nobs_encode=feat_K,
                    init_noise=noise_K.reshape(bs * K_actor, self.action_horizon, self.action_dim),
                    unnormalize=False,
                )["sparse"].reshape(bs, K_actor, self.action_horizon, self.action_dim)
                # K-expanded noise and BC for next state
                noise_next_K = torch.randn(bs, K_critic, self.action_horizon,
                                           self.action_dim, device=self.device)
                feat_next_K  = feat_next.unsqueeze(1).expand(-1, K_critic, -1).reshape(bs * K_critic, -1)
                bc_next_K    = self.bc_policy.predict_action_from_features(
                    sparse_nobs_encode=feat_next_K,
                    init_noise=noise_next_K.reshape(bs * K_critic, self.action_horizon, self.action_dim),
                    unnormalize=False,
                )["sparse"].reshape(bs, K_critic, self.action_horizon, self.action_dim)
            feat_list.append(feat)
            fnext_list.append(feat_next)
            noise_K_list.append(noise_K)
            noise_next_K_list.append(noise_next_K)
            bc_K_list.append(bc_K)
            bc_next_K_list.append(bc_next_K)
            act_list.append(raw["action"])
            rew_list.append(raw["reward"])
            done_list.append(raw["done"])
            is_expert_list.append(raw["is_expert"])
            n += bs

        # Restore BC's inference steps for any downstream use
        if _orig_inf_steps is not None:
            self.bc_policy.num_inference_steps = _orig_inf_steps

        if not feat_list:
            log.warning("Empty replay buffer — skipping training round")
            return

        pool_feat         = torch.cat(feat_list)            # (N, D)
        pool_feat_next    = torch.cat(fnext_list)           # (N, D)
        pool_noise_K      = torch.cat(noise_K_list)         # (N, K_actor, H, D)
        pool_noise_next_K = torch.cat(noise_next_K_list)    # (N, K_critic, H, D)
        pool_bc_K         = torch.cat(bc_K_list)            # (N, K_actor, H, D)
        pool_bc_next_K    = torch.cat(bc_next_K_list)       # (N, K_critic, H, D)
        pool_act          = torch.cat(act_list)             # (N, H, D)
        pool_rew          = torch.cat(rew_list)             # (N, 1)
        pool_done         = torch.cat(done_list)            # (N, 1)
        pool_is_expert    = torch.cat(is_expert_list)       # (N, 1)
        N = pool_feat.shape[0]
        log.info("Pool ready. Running %d MLP-only gradient steps…", self.gradient_steps)

        for step in range(self.gradient_steps):
            idx = torch.randint(0, N, (self.batch_size,), device=self.device)
            feat         = pool_feat[idx]
            feat_next    = pool_feat_next[idx]
            noise_K      = pool_noise_K[idx]
            noise_next_K = pool_noise_next_K[idx]
            bc_K         = pool_bc_K[idx]
            bc_next_K    = pool_bc_next_K[idx]
            act          = pool_act[idx]
            rew          = pool_rew[idx]
            done         = pool_done[idx]
            is_expert    = pool_is_expert[idx]

            c_loss = self._critic_update_feat(
                feat, act, rew, feat_next, done, noise_next_K, bc_next_K)
            a_loss = self._actor_update_feat(feat, noise_K, bc_K, is_expert)
            self._update_target_critics()
            self.total_gradient_steps += 1

            if step % 200 == 0:
                log.info("  [step %5d/%5d] critic_loss=%.4f  actor_loss=%.4f  "
                         "expert_ratio=%.2f",
                         step, self.gradient_steps, c_loss, a_loss, er)

        if self.rl_checkpoint_dir:
            self._save_checkpoint()

    # ------------------------------------------------------------------
    # Actor/critic updates
    # ------------------------------------------------------------------

    def _encode_obs(self, obs: dict) -> torch.Tensor:
        """Run frozen BC policy encoder on obs, return (B, feature_dim)."""
        nobs = {k: self.bc_policy.sparse_normalizer[k].normalize(v)
                for k, v in obs.items()}
        with torch.no_grad():
            return self.bc_policy.obs_encoder(nobs)

    def _critic_update_feat(self, feat, act, rew, feat_next, done,
                            noise_next_K, bc_next_K) -> float:
        """Critic TD update — averages target Q over K_critic next-noise samples.

        Faithful to original distill_rl.loss() with multi_sample_next_noise=True,
        clip_action=False (no clamp on next_act).
        """
        B, K = noise_next_K.shape[0], noise_next_K.shape[1]
        H, D = self.action_horizon, self.action_dim

        with torch.no_grad():
            feat_next_K_flat  = feat_next.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
            noise_next_K_flat = noise_next_K.reshape(B * K, H, D)
            bc_next_K_flat    = bc_next_K.reshape(B * K, H, D)
            delta_next_flat   = self.actor(feat_next_K_flat.unsqueeze(1), noise_next_K_flat)
            next_act_flat     = bc_next_K_flat + delta_next_flat   # NO clamp (original clip_action=False)

            q_targets = [ct(feat_next_K_flat, noise_next_K_flat, next_act_flat)
                         for ct in self.critic_targets]
            q_target_flat = torch.min(torch.stack(q_targets, dim=-1), dim=-1).values  # (B*K, 1)
            target_next_q = q_target_flat.reshape(B, K, 1).mean(dim=1)                # (B, 1)
            td_target     = rew + self.gamma * (1 - done) * target_next_q

        # Critic prediction on actual transition action. Use a single fresh noise
        # for the critic's noise input (the critic learns Q(s, z, a); z here is a
        # conditioning variable that doesn't need to match BC noise for data actions).
        noise_data = torch.randn(B, H, D, device=self.device)
        q_preds = [c(feat, noise_data, act) for c in self.critics]
        critic_loss = sum(F.mse_loss(q, td_target) for q in q_preds) / len(self.critics)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            [p for c in self.critics for p in c.parameters()], self.max_grad_norm)
        self.critic_optim.step()
        return critic_loss.item()

    def _actor_update_feat(self, feat, noise_K, bc_K, is_expert) -> float:
        """Actor update — faithful port of original distill_rl.actor_loss().

        Warmup phase (training_step <= q_filtering_warmup_steps):
          - simple Q + BC loss, no filter
        Post-warmup phase:
          - Compute Q(s, a_actor) and Q(s, a_BC) per K-sample
          - bc_filter = 1 - I[Q_actor > Q_BC]   (codebase fallback, q_overestimation=None)
          - BC loss = mean over K of bc_filter * ||residual||^2
          - This is the "DICE-RL" mechanism: stop pulling actor back to BC
            once its action is clearly value-improving.
        Both phases apply:
          - q_normalisation (q_loss /= mean(|Q|))
          - disable_q_loss_for_expert_data mask
        """
        B, K = noise_K.shape[0], noise_K.shape[1]
        H, D = self.action_horizon, self.action_dim

        feat_K_flat  = feat.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        noise_K_flat = noise_K.reshape(B * K, H, D)
        bc_K_flat    = bc_K.reshape(B * K, H, D)

        # actor action (with grads)
        delta_flat     = self.actor(feat_K_flat.unsqueeze(1), noise_K_flat)
        final_act_flat = bc_K_flat + delta_flat                # NO clamp (matches codebase clip_action=False)

        # Q(s, a_actor) with grads (for the Q-maximisation loss)
        q_actor_flat = torch.stack(
            [c(feat_K_flat, noise_K_flat, final_act_flat) for c in self.critics], dim=-1
        )                                                      # (B*K, 1, ensemble)
        q_actor_min_flat = q_actor_flat.min(dim=-1).values     # (B*K, 1)
        q_actor_per_K    = q_actor_min_flat.reshape(B, K)      # (B, K)

        # ---- Q-loss ----
        q_per_b = q_actor_per_K.mean(dim=1)                    # (B,)
        online_mask = None
        if self.disable_q_loss_for_expert_data:
            online_mask = (1.0 - is_expert.squeeze(-1)).float()  # (B,)
            q_per_b = q_per_b * online_mask
        q_loss = -q_per_b.mean()
        if self.use_q_normalization:
            with torch.no_grad():
                if online_mask is not None:
                    om_K = online_mask.unsqueeze(-1).expand(-1, K)
                    cnt  = om_K.sum().clamp_min(1.0)
                    q_abs_mean = (q_actor_per_K.abs() * om_K).sum() / cnt
                else:
                    q_abs_mean = q_actor_per_K.abs().mean()
            if q_abs_mean > 1e-8:
                q_loss = q_loss / q_abs_mean

        # ---- BC loss (with optional soft Q filter) ----
        # MSE-per-sample over (horizon, action_dim): (B, K)
        residual = delta_flat.reshape(B, K, H, D)
        mse_per_sample = (residual ** 2).mean(dim=(2, 3))        # (B, K)

        in_warmup = self.total_gradient_steps <= self.q_filtering_warmup_steps
        if (not in_warmup) and self.use_soft_q_filtering:
            with torch.no_grad():
                # Q(s, a_BC) — Q evaluated at the unmodified BC action
                q_bc_flat = torch.stack(
                    [c(feat_K_flat, noise_K_flat, bc_K_flat) for c in self.critics], dim=-1
                )
                q_bc_min_flat = q_bc_flat.min(dim=-1).values    # (B*K, 1)
                q_bc_per_K = q_bc_min_flat.reshape(B, K)        # (B, K)
                better_than_bc = (q_actor_per_K > q_bc_per_K).float()  # (B, K)
                bc_filter_expanded = 1.0 - better_than_bc       # (B, K)
            bc_loss = (bc_filter_expanded * mse_per_sample).mean()
        else:
            bc_loss = mse_per_sample.mean()                     # plain mean (warmup behaviour)

        actor_loss = q_loss + self.bc_loss_weight * bc_loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()
        return actor_loss.item()

    def _update_target_critics(self) -> None:
        for ct, c in zip(self.critic_targets, self.critics):
            for pt, p in zip(ct.parameters(), c.parameters()):
                pt.data.mul_(1 - self.tau).add_(p.data * self.tau)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_expert_ratio(self) -> float:
        if not self.use_adaptive:
            return self.expert_ratio
        frac = min(1.0, self.total_gradient_steps / max(self.ratio_steps, 1))
        return self.ratio_start + (self.ratio_end - self.ratio_start) * frac

    def _receive_episode(self) -> Optional[dict]:
        """Non-blocking check for episode from env runner."""
        try:
            data, _ = self.learner_node.transitions_client.pop_data(
                topic=self.learner_node.transitions_topic,
                order="latest", n=1,
            )
            if data:
                return pickle.loads(data[0])
        except Exception as e:
            log.debug("Receive error (non-fatal): %s", e)
        return None

    def _push_actor_weights(self) -> None:
        """Save actor weights to disk for env runner to pick up.

        Uses atomic write (tmp+rename) so env runner never reads a half-written
        file. Replaces the previous ZMQ-based push which was unreliable.
        """
        if not self.rl_checkpoint_dir:
            return
        path = os.path.join(self.rl_checkpoint_dir, "latest_actor.pt")
        payload = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_config": {
                "obs_dim": self.bc_policy.obs_feature_dim,
                "action_dim": self.action_dim,
                "cond_steps": 1,
                "horizon_steps": self.action_horizon,
                "hidden_dims": self._actor_hidden_dims,
            },
            "training_step": self.total_gradient_steps,
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        log.info("★ Saved latest actor weights to %s (step=%d)",
                 os.path.basename(path), self.total_gradient_steps)

    def _maybe_resume_checkpoint(self) -> bool:
        if not self.rl_checkpoint_dir:
            return False
        ckpts = sorted(glob.glob(os.path.join(self.rl_checkpoint_dir, "checkpoint_*.pt")))
        if not ckpts:
            return False
        path = ckpts[-1]
        log.info("Resuming from checkpoint: %s", path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        for c, sd in zip(self.critics, ckpt["critics"]):
            c.load_state_dict(sd)
        for ct, sd in zip(self.critic_targets, ckpt["critics"]):
            ct.load_state_dict(sd)
        self.total_gradient_steps = ckpt["total_gradient_steps"]
        self.total_episodes       = ckpt["total_episodes"]
        log.info("Resumed: total_episodes=%d  total_gradient_steps=%d",
                 self.total_episodes, self.total_gradient_steps)
        return True

    def _save_checkpoint(self) -> None:
        path = os.path.join(
            self.rl_checkpoint_dir,
            f"checkpoint_{self.total_gradient_steps:06d}.pt",
        )
        torch.save({
            "actor": self.actor.state_dict(),
            "critics": [c.state_dict() for c in self.critics],
            "total_gradient_steps": self.total_gradient_steps,
            "total_episodes": self.total_episodes,
        }, path)
        log.info("Saved RL checkpoint: %s", path)
