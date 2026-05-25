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
        bc_loss_weight: float = 100.0,
        critic_ensemble_size: int = 5,
        max_grad_norm: float = 1.0,
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

        # ---- replay buffer ----
        self.replay_buffer = YAMReplayBuffer(
            expert_npz_path=expert_npz_path,
            online_data_dir=online_data_dir,
            obs_horizon=obs_horizon,
            action_dim=action_dim,
            action_horizon=action_horizon,
            device=device,
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
        """Run gradient_steps actor+critic updates with pre-encoded feature pool.

        ViT runs only during pool encoding (2 passes total), not inside the
        gradient loop — this gives a ~40× speedup over per-step encoding.
        """
        er = self._current_expert_ratio()

        # Build feature pool in small mini-batches to stay within GPU memory.
        pool_size = min(self.gradient_steps * self.batch_size, 10_000)
        encode_bs = 256
        log.info("Pre-encoding feature pool (%d transitions, mini-batch=%d)…",
                 pool_size, encode_bs)
        feat_list, fnext_list, act_list, rew_list, done_list = [], [], [], [], []
        n = 0
        while n < pool_size:
            bs = min(encode_bs, pool_size - n)
            raw = self.replay_buffer.sample(bs, expert_ratio=er)
            if not raw:
                break
            with torch.no_grad():
                feat_list.append(self._encode_obs(raw["obs"]))
                fnext_list.append(self._encode_obs(raw["next_obs"]))
            act_list.append(raw["action"])
            rew_list.append(raw["reward"])
            done_list.append(raw["done"])
            n += bs

        if not feat_list:
            log.warning("Empty replay buffer — skipping training round")
            return

        pool_feat      = torch.cat(feat_list)   # (N, feat_dim)
        pool_feat_next = torch.cat(fnext_list)  # (N, feat_dim)
        pool_act       = torch.cat(act_list)    # (N, H, D)
        pool_rew       = torch.cat(rew_list)    # (N, 1)
        pool_done      = torch.cat(done_list)   # (N, 1)
        N = pool_feat.shape[0]
        log.info("Pool ready (%d features). Running %d MLP-only gradient steps…",
                 N, self.gradient_steps)

        for step in range(self.gradient_steps):
            idx = torch.randint(0, N, (self.batch_size,), device=self.device)
            feat      = pool_feat[idx]
            feat_next = pool_feat_next[idx]
            act       = pool_act[idx]
            rew       = pool_rew[idx]
            done      = pool_done[idx]

            c_loss = self._critic_update_feat(feat, act, rew, feat_next, done)
            a_loss = self._actor_update_feat(feat)
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

    def _critic_update_feat(self, feat, act, rew, feat_next, done) -> float:
        """Critic TD update using pre-encoded features (no ViT call).

        Residual setup: target action = clamp(BC(s') + actor(s')) so the critic
        is trained on the same action distribution it will see at inference.
        """
        with torch.no_grad():
            noise = torch.randn(feat_next.shape[0], self.action_horizon,
                                self.action_dim, device=self.device)
            delta_next = self.actor(feat_next.unsqueeze(1), noise)
            bc_act_next = self.bc_policy.predict_action_from_features(
                sparse_nobs_encode=feat_next, init_noise=noise, unnormalize=False,
            )["sparse"]
            next_act = (bc_act_next + delta_next).clamp(-1.0, 1.0)

            q_targets = [ct(feat_next, noise, next_act) for ct in self.critic_targets]
            q_target  = torch.min(torch.stack(q_targets, dim=-1), dim=-1).values
            td_target = rew + self.gamma * (1 - done) * q_target

        noise_full = torch.randn(feat.shape[0], self.action_horizon,
                                 self.action_dim, device=self.device)
        q_preds = [c(feat, noise_full, act) for c in self.critics]
        critic_loss = sum(F.mse_loss(q, td_target) for q in q_preds) / len(self.critics)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            [p for c in self.critics for p in c.parameters()], self.max_grad_norm)
        self.critic_optim.step()
        return critic_loss.item()

    def _actor_update_feat(self, feat) -> float:
        """Actor update using pre-encoded features (no ViT call).

        Residual policy: actor outputs delta. Final action = clamp(BC + delta).
        Critic evaluates the final action (matches inference).
        BC loss pulls delta toward zero so the actor stays close to BC unless
        Q-learning gives a strong signal to deviate.
        """
        noise = torch.randn(feat.shape[0], self.action_horizon,
                            self.action_dim, device=self.device)
        delta = self.actor(feat.unsqueeze(1), noise)  # (B, H, D) residual
        with torch.no_grad():
            bc_act_n = self.bc_policy.predict_action_from_features(
                sparse_nobs_encode=feat, init_noise=noise, unnormalize=False,
            )["sparse"]
        final_act = (bc_act_n + delta).clamp(-1.0, 1.0)

        # Q-maximisation on the FULL (clipped) action — same as inference.
        q_vals = torch.stack([c(feat, noise, final_act) for c in self.critics], dim=-1)
        q_loss = -q_vals.min(dim=-1).values.mean()

        # Keep the residual small (regularise toward BC by construction).
        bc_loss = (delta ** 2).mean()

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
