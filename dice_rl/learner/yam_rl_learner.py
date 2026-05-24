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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main learner loop: listen for episodes, train, push weights."""
        log.info("Learner running. Waiting for episodes…")
        log.info("  num_episodes_before_first_training = %d",
                 self.num_episodes_before_first_training)
        log.info("  update_every_x_episode             = %d",
                 self.update_every_x_episode)
        log.info("  gradient_steps per round           = %d", self.gradient_steps)
        log.info("  batch_size (expert+online)         = %d  (expert_ratio=%.1f%%)",
                 self.batch_size, self.expert_ratio * 100)

        while True:
            # ---- receive an episode ----
            episode_data = self._receive_episode()
            if episode_data is None:
                time.sleep(0.1)
                continue

            self.replay_buffer.add_episode(episode_data)
            self.total_episodes += 1
            log.info("Episode received (#%d). Online transitions: %d",
                     self.total_episodes, self.replay_buffer.num_online_transitions)

            # ---- decide whether to train ----
            past_warmup = self.total_episodes >= self.num_episodes_before_first_training
            trigger = (self.total_episodes - self.num_episodes_before_first_training) \
                      % self.update_every_x_episode == 0
            if past_warmup and trigger:
                log.info("Training round %d  (%d gradient steps)…",
                         self.total_episodes, self.gradient_steps)
                self._train_round()
                self._push_actor_weights()

    def _train_round(self) -> None:
        """Run gradient_steps actor+critic updates."""
        for step in range(self.gradient_steps):
            er = self._current_expert_ratio()
            batch = self.replay_buffer.sample(self.batch_size, expert_ratio=er)
            if not batch:
                log.warning("Empty batch — skipping")
                continue

            c_loss = self._critic_update(batch)
            a_loss = self._actor_update(batch)
            self._update_target_critics()
            self.total_gradient_steps += 1

            if step % 200 == 0:
                log.info("  [step %5d/%5d] critic_loss=%.4f  actor_loss=%.4f  "
                         "expert_ratio=%.2f",
                         step, self.gradient_steps, c_loss, a_loss, er)

        # Save checkpoint every round.
        if self.rl_checkpoint_dir:
            self._save_checkpoint()

    # ------------------------------------------------------------------
    # Actor/critic updates
    # ------------------------------------------------------------------

    def _encode_obs(self, obs: dict) -> torch.Tensor:
        """Run frozen BC policy encoder on obs, return (B, feature_dim)."""
        # obs keys: rgb_0 (B,To,3,H,W), rgb_1, joint_pos (B,To,7)
        nobs = {k: self.bc_policy.sparse_normalizer[k].normalize(v)
                for k, v in obs.items()}
        with torch.no_grad():
            return self.bc_policy.obs_encoder(nobs)

    def _critic_update(self, batch: dict) -> float:
        obs, act, rew, next_obs, done = (
            batch["obs"], batch["action"], batch["reward"],
            batch["next_obs"], batch["done"],
        )
        with torch.no_grad():
            feat_next = self._encode_obs(next_obs)
            noise = torch.randn(
                feat_next.shape[0], self.action_horizon, self.action_dim,
                device=self.device,
            )
            next_act_n = self.actor(feat_next.unsqueeze(1), noise)  # normalized
            # target Q
            q_targets = [ct(feat_next, noise, next_act_n) for ct in self.critic_targets]
            q_target = torch.min(torch.stack(q_targets, dim=-1), dim=-1).values
            td_target = rew + self.gamma * (1 - done) * q_target

        feat = self._encode_obs(obs)
        noise = torch.randn_like(act.unsqueeze(1).expand(-1, 1, -1))
        noise = noise[:, 0, :]  # keep (B, action_dim)
        noise_full = torch.randn(feat.shape[0], self.action_horizon, self.action_dim,
                                 device=self.device)
        q_preds = [c(feat, noise_full, act) for c in self.critics]
        critic_loss = sum(F.mse_loss(q, td_target) for q in q_preds) / len(self.critics)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            [p for c in self.critics for p in c.parameters()], self.max_grad_norm
        )
        self.critic_optim.step()
        return critic_loss.item()

    def _actor_update(self, batch: dict) -> float:
        obs, act_expert = batch["obs"], batch["action"]
        feat = self._encode_obs(obs)
        noise = torch.randn(
            feat.shape[0], self.action_horizon, self.action_dim, device=self.device
        )
        pred_act = self.actor(feat.unsqueeze(1), noise)  # (B, H, D) normalized

        # Q-maximisation loss
        q_vals = torch.stack([c(feat, noise, pred_act) for c in self.critics], dim=-1)
        q_min = q_vals.min(dim=-1).values
        q_loss = -q_min.mean()

        # BC regularisation: stay close to BC policy's recommended action
        with torch.no_grad():
            bc_act_n = self.bc_policy.predict_action_from_features(
                sparse_nobs_encode=feat,
                init_noise=noise,
                unnormalize=False,
            )["sparse"]
        bc_loss = F.mse_loss(pred_act, bc_act_n)

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
        """Broadcast updated actor weights to env runner via ZMQ."""
        payload = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_config": {
                "obs_dim": self.bc_policy.obs_feature_dim,
                "action_dim": self.action_dim,
                "cond_steps": 1,
                "horizon_steps": self.action_horizon,
                "hidden_dims": list(self.actor.net[0].in_features
                                    for _ in range(1)),  # approximate
            },
            "training_step": self.total_gradient_steps,
        }
        self.learner_node.network_weight_server.push_data(
            data=pickle.dumps(payload),
            topic=self.learner_node.network_weight_topic,
        )
        log.info("Pushed actor weights (step %d)", self.total_gradient_steps)

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
