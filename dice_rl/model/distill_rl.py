"""
Distilled RL Model for online finetuning of flow matching policies.

This module implements the core components for online RL finetuning:
- DistilledActor: One-step distilled actor network
- Critic: Q-function network
- DynamicsModel: Environment dynamics model
- ConfidenceEstimator: Confidence estimation for pretrained policy
- DistillRLModel: Main model that orchestrates all components

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import hydra
import os
from omegaconf import OmegaConf

from utils.model_io import load_policy

log = logging.getLogger(__name__)

from dice_rl.model.common.mlp import MLP


class DistilledActor(nn.Module):
    """
    One-step distilled actor network.
    
    Takes state s and noise z as input, outputs action a.
    This is a simple MLP that learns to map (s, z) -> a.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cond_steps: int = 1,
        horizon_steps: int = 8,
        hidden_dims: List[int] = [256, 256, 256],
        activation_type: str = "Mish",
        use_layernorm: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        
        # Input: flattened state + noise
        # state: (B, cond_steps, obs_dim) -> flattened to (B, cond_steps * obs_dim)
        # noise: (B, horizon_steps, action_dim) -> flattened to (B, horizon_steps * action_dim)
        input_dim = cond_steps * obs_dim + horizon_steps * action_dim
        output_dim = horizon_steps * action_dim
        
        mlp_dims = [input_dim] + hidden_dims + [output_dim]
        
        self.mlp = MLP(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )
    
    def forward(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (B, cond_steps, obs_dim) - chunked state
            noise: (B, horizon_steps, action_dim) - chunked noise
            
        Returns:
            action: (B, horizon_steps, action_dim) - predicted action chunk
        """
        B = noise.shape[0]  # batch size
        if isinstance(state, dict):
            state = state["state"]
        # Flatten inputs for processing
        state_flat = state.view(B, -1)  # (B, cond_steps * obs_dim)
        noise_flat = noise.view(B, -1)  # (B, horizon_steps * action_dim)
        
        # Concatenate flattened inputs
        # BUG: use zero noise to debug
        # zero_noise_flat = torch.zeros_like(noise_flat)
        x = torch.cat([state_flat, noise_flat], dim=-1)  # (B, cond_steps * obs_dim + horizon_steps * action_dim)
        action_flat = self.mlp(x)  # (B, horizon_steps * action_dim)
        
        # Reshape back to chunked format
        action = action_flat.view(B, self.horizon_steps, self.action_dim)  # (B, horizon_steps, action_dim)
        
        return action


class DistilledTransformerActor(nn.Module):
    """
    Transformer-based distilled actor network.
    
    Takes state s and noise z as input, outputs action a.
    Uses self-attention to process the sequence of state and noise tokens.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cond_steps: int = 1,
        horizon_steps: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation_type: str = "GELU",
        use_layernorm: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        self.hidden_dim = hidden_dim
        
        # Token embeddings: project state and noise to hidden dimension
        self.state_embed = nn.Linear(obs_dim, hidden_dim)
        self.noise_embed = nn.Linear(action_dim, hidden_dim)
        
        # Positional embeddings
        # Total sequence length: cond_steps (for states) + horizon_steps (for noise)
        self.total_seq_len = cond_steps + horizon_steps
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_seq_len, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Token type embeddings (to distinguish state from noise tokens)
        self.state_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.noise_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.state_type_embed, std=0.02)
        nn.init.normal_(self.noise_type_embed, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation=activation_type.lower() if activation_type.lower() in ['relu', 'gelu'] else 'gelu',
            batch_first=True,
            norm_first=use_layernorm
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection: from hidden_dim to action_dim for each horizon step
        # We'll use the last horizon_steps tokens to generate actions
        self.action_proj = nn.Linear(hidden_dim, action_dim)
        
        # Optional final layer norm
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using transformer architecture.
        
        Args:
            state: (B, cond_steps, obs_dim) - chunked state
            noise: (B, horizon_steps, action_dim) - chunked noise
            
        Returns:
            action: (B, horizon_steps, action_dim) - predicted action chunk
        """
        B = state.shape[0]  # batch size
        
        if isinstance(state, dict):
            state = state["state"]
        
        # Embed state and noise tokens
        state_tokens = self.state_embed(state)  # (B, cond_steps, hidden_dim)
        noise_tokens = self.noise_embed(noise)  # (B, horizon_steps, hidden_dim)
        
        # Add type embeddings
        state_tokens = state_tokens + self.state_type_embed  # (B, cond_steps, hidden_dim)
        noise_tokens = noise_tokens + self.noise_type_embed  # (B, horizon_steps, hidden_dim)
        
        # Concatenate state and noise tokens
        tokens = torch.cat([state_tokens, noise_tokens], dim=1)  # (B, cond_steps + horizon_steps, hidden_dim)
        
        # Add positional embeddings
        tokens = tokens + self.pos_embed  # (B, total_seq_len, hidden_dim)
        
        # Apply transformer
        encoded = self.transformer(tokens)  # (B, total_seq_len, hidden_dim)
        
        # Extract the last horizon_steps tokens (corresponding to noise positions)
        # These will be used to generate actions
        action_tokens = encoded[:, self.cond_steps:, :]  # (B, horizon_steps, hidden_dim)
        
        # Apply final layer norm if specified
        if self.use_layernorm:
            action_tokens = self.final_norm(action_tokens)
        
        # Project to action dimension
        actions = self.action_proj(action_tokens)  # (B, horizon_steps, action_dim)
        
        return actions


class DistilledCritic(nn.Module):
    """
    Critic network that takes state, noise, and action as input.
    
    Q(s, z, a) -> scalar value or Q(s, a) -> scalar value
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cond_steps: int = 1,
        horizon_steps: int = 8,
        hidden_dims: List[int] = [256, 256, 256],
        activation_type: str = "Mish",
        use_layernorm: bool = False,
        q_depends_on_noise: bool = True,  # If False, Q(s,a) instead of Q(s,z,a)
        critic_ensemble_size: int = 2,  # Number of Q-networks in ensemble
        conservative_q_method: str = "min",  # "min" or "lcb" (lower confidence bound)
        lcb_kappa: float = 0.1,  # κ parameter for LCB: Q = μ - κσ
        td_loss: str = "mse",  # TD loss type: "mse", "huber", "bce"
        **kwargs
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        self.q_depends_on_noise = q_depends_on_noise
        self.critic_ensemble_size = critic_ensemble_size
        self.conservative_q_method = conservative_q_method
        self.lcb_kappa = lcb_kappa
        self.td_loss = td_loss
        
        # Input: flattened state + noise + action OR state + action
        # state: (B, cond_steps, obs_dim) -> flattened to (B, cond_steps * obs_dim)
        # noise: (B, horizon_steps, action_dim) -> flattened to (B, horizon_steps * action_dim) [optional]
        # action: (B, horizon_steps, action_dim) -> flattened to (B, horizon_steps * action_dim)
        if q_depends_on_noise:
            input_dim = cond_steps * obs_dim + 2 * horizon_steps * action_dim
        else:
            input_dim = cond_steps * obs_dim + horizon_steps * action_dim
        
        mlp_dims = [input_dim] + hidden_dims + [1]
        
        # Always use Identity activation - for BCE we'll use BCEWithLogitsLoss
        out_activation = "Identity"
        
        # Create ensemble of Q-networks
        self.Q_ensemble = nn.ModuleList([
            MLP(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type=out_activation,
                use_layernorm=use_layernorm,
            )
            for _ in range(critic_ensemble_size)
        ])
    
    def forward(self, state: torch.Tensor, noise: torch.Tensor, action: torch.Tensor, return_all=False, return_mean=False) -> torch.Tensor:
        """
        Forward pass through ensemble of Q-networks.
        
        Args:
            state: (B, cond_steps, obs_dim) - chunked state
            noise: (B, horizon_steps, action_dim) - chunked noise (ignored if q_depends_on_noise=False)
            action: (B, horizon_steps, action_dim) - chunked action
            return_all: bool - if True, return all Q-values, else return conservative estimate
            
        Returns:
            If return_all=True: List of Q-values from all networks [(B, 1), ...]
            If return_all=False: Conservative Q-value estimate (B, 1) using specified method
        """
        return_mean = False # here i forced return min, i found using mean for actor loss comparable to min
        B = state.shape[0]  # batch size
        
        # Flatten all inputs for processing
        state_flat = state.view(B, -1)  # (B, cond_steps * obs_dim)
        action_flat = action.view(B, -1)  # (B, horizon_steps * action_dim)
        
        # Concatenate inputs based on whether Q depends on noise
        if self.q_depends_on_noise:
            noise_flat = noise.view(B, -1)  # (B, horizon_steps * action_dim)
            x = torch.cat([state_flat, noise_flat, action_flat], dim=-1)  # (B, cond_steps * obs_dim + 2 * horizon_steps * action_dim)
        else:
            x = torch.cat([state_flat, action_flat], dim=-1)  # (B, cond_steps * obs_dim + horizon_steps * action_dim)
        
        # Get Q-values from all networks in ensemble
        q_values = []
        for q_network in self.Q_ensemble:
            q_val = q_network(x)  # (B, 1) - raw logits if td_loss=="bce"
            
            # Apply sigmoid for BCE loss to get Q-values in [0,1] for everything except loss computation
            # (critic_loss method will use raw logits with BCEWithLogitsLoss)
            if self.td_loss == "bce" and not return_all:
                # Apply sigmoid when returning single Q-value for actor loss or evaluation
                q_val = torch.sigmoid(q_val)
                
            q_values.append(q_val)
        
        if return_all:
            # Return raw logits for critic_loss when td_loss=="bce"
            return q_values  # List of (B, 1) tensors
        else:
            # Stack Q-values for easier manipulation
            q_stacked = torch.stack(q_values, dim=0)  # (ensemble_size, B, 1)
            if return_mean:
                return q_stacked.mean(dim=0)  # (B, 1)
            if self.conservative_q_method == "min":
                # Return minimum Q-value for conservative estimate (this works)
                return torch.min(q_stacked, dim=0)[0]  # (B, 1)
            elif self.conservative_q_method == "lcb":
                # Lower Confidence Bound: Q = μ - κσ (this doesn't work, has to be min)
                q_mean = q_stacked.mean(dim=0)  # (B, 1)
                q_std = q_stacked.std(dim=0)  # (B, 1)
                return q_mean - self.lcb_kappa * q_std  # (B, 1)
            else:
                raise ValueError(f"Unknown conservative_q_method: {self.conservative_q_method}")
    
    def return_both(self, *args, **kwargs):
        """Backward compatibility method - now returns all Q-values"""
        return self.forward(*args, return_all=True, **kwargs)


class DynamicsModel(nn.Module):
    """
    Dynamics model that predicts next state given current state and action.
    
    f(s, a) -> s_next
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cond_steps: int = 1,
        horizon_steps: int = 8,
        hidden_dims: List[int] = [256, 256, 256],
        activation_type: str = "Mish",
        use_layernorm: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        
        # Input: flattened state + action
        # state: (B, cond_steps, obs_dim) -> flattened to (B, cond_steps * obs_dim)
        # action: (B, horizon_steps, action_dim) -> flattened to (B, horizon_steps * action_dim)
        input_dim = cond_steps * obs_dim + horizon_steps * action_dim
        output_dim = cond_steps * obs_dim  # predict next state with same shape as current state
        
        mlp_dims = [input_dim] + hidden_dims + [output_dim]
        
        self.mlp = MLP(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (B, cond_steps, obs_dim) - chunked state
            action: (B, horizon_steps, action_dim) - chunked action
            
        Returns:
            next_state: (B, cond_steps, obs_dim) - predicted next state chunk
        """
        B = state.shape[0]  # batch size
        
        # Flatten inputs for processing
        state_flat = state.view(B, -1)  # (B, cond_steps * obs_dim)
        action_flat = action.view(B, -1)  # (B, horizon_steps * action_dim)
        
        # Concatenate flattened inputs
        x = torch.cat([state_flat, action_flat], dim=-1)  # (B, cond_steps * obs_dim + horizon_steps * action_dim)
        next_state_flat = self.mlp(x)  # (B, cond_steps * obs_dim)
        
        # Reshape back to chunked format
        next_state = next_state_flat.view(B, self.cond_steps, self.obs_dim)  # (B, cond_steps, obs_dim)
        
        return next_state


# Confidence estimation is now handled by the pretrained flow matching policy
# No separate ConfidenceEstimator class needed


class DistillRLModel(nn.Module):
    """
    Main model for distilled RL finetuning.
    
    This orchestrates all components:
    - DistilledActor: π_θ(a|s,z)
    - DistilledCritic: Q_φ(s,z,a)
    - DynamicsModel: f_ρ(s,a)
    - ConfidenceEstimator: w(s,z)
    - PretrainedFlowPolicy: π_pre(a|s,z) (frozen)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        noise_dim: int,
        pretrained_flow_policy_path: str,
        # Network configurations
        actor_hidden_dims: List[int] = [256, 256, 256],
        critic_hidden_dims: List[int] = [256, 256, 256],
        dynamics_hidden_dims: List[int] = [256, 256, 256],
        activation_type: str = "Mish",
        # Transformer actor settings
        use_transformer_actor: bool = False,
        transformer_hidden_dim: int = 256,
        transformer_num_heads: int = 4,
        transformer_num_layers: int = 3,
        transformer_dropout: float = 0.1,
        # Loss coefficients
        bc_loss_weight: float = 1.0,
        future_confidence_weight: float = 1.0,
        dynamics_weight: float = 1.0,
        critic_weight: float = 1.0,
        # Confidence estimation parameters
        use_confidence_estimation: bool = True,
        use_gfc_loss: bool = False,
        use_dtr_loss: bool = False,
        confidence_metric: str = "score_matching",  # "curvature", "finite_difference", "score_matching"
        # confidence_normalization: str = "running_stats",  # "running_stats", "min_max", "none"
        confidence_normalization: str = "running_stats",  # "running_stats", "min_max", "none"
        # Q-filtering parameters
        use_soft_q_filtering: bool = False,
        use_tube_loss: bool = False,
        q_filtering_warmup_steps: int = 25000,
        q_underestimation_threshold: float = -0.1,  # Threshold for detecting Q underestimation
        # Exploration strategy warmup (separate from Q-filtering warmup)
        replay_flow_warmup_steps: int = 1000,
        # Intrinsic reward parameters
        use_intrinsic_reward: bool = False,
        intrinsic_reward_lambda: float = 0.012,
        intrinsic_reward_tau: float = 2.0,
        intrinsic_reward_samples: int = 5,
        intrinsic_reward_baseline_thres: float = 0.5,
        clamp_intrinsic_reward: bool = False,  # If True, clamp intrinsic reward to be <= 0
        # q normalization
        use_q_normalization: bool = False,
        # Q-function noise dependency
        q_depends_on_noise: bool = True,
        multi_sample_next_noise: bool = False,  # If True, use multiple samples for next noise
        num_next_noise_samples: int = 4,  # Number of samples for next noise (K)
        # Optimistic target settings
        optimistic_target: bool = False,  # If True, use percentile instead of mean for multi-sample targets
        optimistic_percentile: float = 0.75,  # Which percentile to use (0.75 = 75th percentile)
        # Action clipping
        clip_action: bool = False,  # If True, clip actions to [-1, 1]
        # Q-value clipping
        clip_q: tuple = None,  # If not None, clip Q-values to (min, max) range
        # Critic ensemble settings
        critic_ensemble_size: int = 2,  # Number of critic pairs in ensemble
        # Dynamics model settings
        use_dynamics: bool = True,
        # Chunk parameters
        cond_steps: int = 1,
        horizon_steps: int = 4,
        # Discount within horizon chunks
        discount_within_horizon: bool = False,
        # Residual action clipping (for residual RL)
        clip_residual_action: bool = False,
        # N-step returns
        use_n_step: bool = False,
        n_step: int = 1,
        # Disable Q loss for expert data (ablation)
        disable_q_loss_for_expert_data: bool = False,
        # Disable TD loss for expert data (ablation)
        disable_td_loss_for_expert_data: bool = False,
        always_retain_bc_loss_for_expert_data: bool = False,
        # TD loss type for critic
        td_loss: str = "mse",  # "mse", "huber", "bce"
        # Asymmetric regression to suppress overestimation
        asymmetric_regression: bool = False,
        # Multi-z sampling for actor loss
        sample_multi_z_for_actor_loss: bool = False,
        num_multi_z_for_actor_loss: int = 8,
        topk_divisor_for_self_imitation: int = 4,
        use_softmax_weighted: bool = False,
        self_imitation_for_actor_loss: bool = False,
        self_imitation_loss_weight: float = 40,
        self_imitation_loss_distributional: bool = False,
        winner_loser: bool = False,
        multi_z_actor_loss_warmup_steps: int = 5000,
        # DDIM inference steps for pretrained policy during RL training
        rl_num_inference_steps: int = None,  # If set, override pretrained policy's num_inference_steps
        # Device
        device: str = "cuda",
        **kwargs
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.noise_dim = noise_dim
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        self.discount_within_horizon = discount_within_horizon
        self.clip_residual_action = clip_residual_action
        self.use_n_step = use_n_step
        self.n_step = n_step
        self.disable_q_loss_for_expert_data = disable_q_loss_for_expert_data
        self.disable_td_loss_for_expert_data = disable_td_loss_for_expert_data
        self.always_retain_bc_loss_for_expert_data = always_retain_bc_loss_for_expert_data
        self.td_loss = td_loss
        self.asymmetric_regression = asymmetric_regression
        self.sample_multi_z_for_actor_loss = sample_multi_z_for_actor_loss
        self.num_multi_z_for_actor_loss = num_multi_z_for_actor_loss
        self.topk_divisor_for_self_imitation = topk_divisor_for_self_imitation
        self.use_softmax_weighted = use_softmax_weighted
        self.self_imitation_for_actor_loss = self_imitation_for_actor_loss
        self.self_imitation_loss_weight = self_imitation_loss_weight
        self.self_imitation_loss_distributional = self_imitation_loss_distributional
        self.winner_loser = winner_loser
        self.multi_z_actor_loss_warmup_steps = multi_z_actor_loss_warmup_steps
        self.rl_num_inference_steps = rl_num_inference_steps
        self.device = device
        
        # Loss coefficients
        self.bc_loss_weight = bc_loss_weight
        self.future_confidence_weight = future_confidence_weight
        self.dynamics_weight = dynamics_weight
        self.critic_weight = critic_weight
        
        # Confidence estimation settings
        self.use_confidence_estimation = use_confidence_estimation
        self.use_gfc_loss = use_gfc_loss
        self.use_dtr_loss = use_dtr_loss
        self.confidence_metric = confidence_metric
        self.confidence_normalization = confidence_normalization
        
        # Q-filtering settings
        self.use_soft_q_filtering = use_soft_q_filtering
        self.use_tube_loss = use_tube_loss
        self.q_filtering_warmup_steps = q_filtering_warmup_steps
        self.replay_flow_warmup_steps = replay_flow_warmup_steps
        self.q_underestimation_threshold = q_underestimation_threshold
        self.use_q_normalization = use_q_normalization
        self.q_depends_on_noise = q_depends_on_noise
        self.multi_sample_next_noise = multi_sample_next_noise
        self.num_next_noise_samples = num_next_noise_samples
        self.optimistic_target = optimistic_target
        self.optimistic_percentile = optimistic_percentile
        self.clip_action = clip_action
        self.clip_q = clip_q
        self.critic_ensemble_size = critic_ensemble_size
        self.conservative_q_method = kwargs.get('conservative_q_method', 'min')
        self.lcb_kappa = kwargs.get('lcb_kappa', 0.1)
        print(f"Q-value clipping set to: {self.clip_q}")
        print(f"Critic ensemble size: {self.critic_ensemble_size}")
        print(f"Conservative Q method: {self.conservative_q_method}")
        if self.conservative_q_method == 'lcb':
            print(f"LCB kappa: {self.lcb_kappa}")
        
        # Intrinsic reward settings
        self.use_intrinsic_reward = use_intrinsic_reward
        self.intrinsic_reward_lambda = intrinsic_reward_lambda
        self.intrinsic_reward_tau = intrinsic_reward_tau
        self.intrinsic_reward_samples = intrinsic_reward_samples
        self.intrinsic_reward_baseline_thres = intrinsic_reward_baseline_thres
        self.clamp_intrinsic_reward = clamp_intrinsic_reward
    
        # Initialize networks with explicit dimensions for chunked data
        # Choose between MLP and Transformer actor
        use_layernorm_val = kwargs.pop('use_layernorm', True)

        if use_transformer_actor:
            log.info("Using Transformer actor")
            # Extract use_layernorm from kwargs to avoid duplicate
            self.actor = DistilledTransformerActor(
                obs_dim=obs_dim,
                action_dim=action_dim,
                cond_steps=cond_steps,
                horizon_steps=horizon_steps,
                hidden_dim=transformer_hidden_dim,
                num_heads=transformer_num_heads,
                num_layers=transformer_num_layers,
                dropout=transformer_dropout,
                activation_type=activation_type,
                use_layernorm=use_layernorm_val
            ).to(device)
        else:
            log.info("Using MLP actor")
            self.actor = DistilledActor(
                obs_dim=obs_dim,
                action_dim=action_dim,
                cond_steps=cond_steps,
                horizon_steps=horizon_steps,
                hidden_dims=actor_hidden_dims,
                activation_type=activation_type,
                use_layernorm=use_layernorm_val,
                **kwargs
            ).to(device)
        
        self.critic = DistilledCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            cond_steps=cond_steps,
            horizon_steps=horizon_steps,
            hidden_dims=critic_hidden_dims,
            activation_type=activation_type,
            q_depends_on_noise=q_depends_on_noise,
            critic_ensemble_size=critic_ensemble_size,
            conservative_q_method=kwargs.get('conservative_q_method', 'min'),
            lcb_kappa=kwargs.get('lcb_kappa', 0.1),
            td_loss=td_loss,
            use_layernorm=use_layernorm_val
        ).to(device)
        
        # Target critic for stable Q-learning (SAC-style)
        self.target_critic = DistilledCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            cond_steps=cond_steps,
            horizon_steps=horizon_steps,
            hidden_dims=critic_hidden_dims,
            activation_type=activation_type,
            q_depends_on_noise=q_depends_on_noise,
            critic_ensemble_size=critic_ensemble_size,
            conservative_q_method=kwargs.get('conservative_q_method', 'min'),
            lcb_kappa=kwargs.get('lcb_kappa', 0.1),
            td_loss=td_loss,
            use_layernorm=use_layernorm_val
        ).to(device)
        
        # Initialize target critic with same weights as critic
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Dynamics model (optional - saves memory when disabled)
        self.use_dynamics = use_dynamics
        if use_dynamics:
            self.dynamics = DynamicsModel(
                obs_dim=obs_dim,
                action_dim=action_dim,
                cond_steps=cond_steps,
                horizon_steps=horizon_steps,
                hidden_dims=dynamics_hidden_dims,
                **kwargs
            ).to(device)
        else:
            self.dynamics = None
            log.info("Dynamics model disabled to save memory")
        
        # Confidence estimation is handled by the pretrained flow matching policy
        # No separate confidence estimator needed
        
        # Load pretrained diffusion policy
        self.pretrained_policy = self._load_pretrained_policy(
            pretrained_flow_policy_path, device
        )
        
        # Confidence normalization statistics
        if confidence_normalization == "running_stats":
            self.register_buffer('confidence_mean', torch.zeros(1, device=device))
            self.register_buffer('confidence_std', torch.ones(1, device=device))
            self.register_buffer('confidence_count', torch.zeros(1, device=device))
            self.register_buffer('confidence_M2', torch.zeros(1, device=device))  # Sum of squared deviations
        
        log.info(f"DistillRLModel initialized with:")
        log.info(f"  obs_dim: {obs_dim}")
        log.info(f"  action_dim: {action_dim}")
        log.info(f"  noise_dim: {noise_dim}")
        log.info(f"  confidence_metric: {confidence_metric}")
        log.info(f"  confidence_normalization: {confidence_normalization}")
        log.info(f"  use_dynamics: {use_dynamics}")
        log.info(f"  td_loss: {td_loss}")
        log.info(f"  asymmetric_regression: {asymmetric_regression}")

    def _clip_q_values(self, q_values):
        """Helper function to clip Q-values with None handling"""
        if self.clip_q is None:
            return q_values
        min_val = self.clip_q[0] if self.clip_q[0] is not None else float('-inf')
        max_val = self.clip_q[1] if self.clip_q[1] is not None else float('inf')
        return torch.clamp(q_values, min=min_val, max=max_val)
    

    def _load_pretrained_policy(self, checkpoint_path: str, device: str):
        """
        Load pretrained diffusion policy using load_policy utility.

        Uses the same loading pattern as env_runners/residual_online_learning_env_runner.py.

        Args:
            checkpoint_path: Path to the pretrained policy checkpoint (.ckpt file or directory)
            device: Device to load the model on

        Returns:
            Loaded and frozen DiffusionUnetTimmMod1Policy
        """
        log.info(f"Loading pretrained diffusion policy from: {checkpoint_path}")

        # Use load_policy utility (handles workspace loading, EMA, etc.)
        pretrained_policy, shape_meta, cfg = load_policy(checkpoint_path, device)

        # Store shape_meta and cfg for later use
        self.pretrained_shape_meta = shape_meta
        self.pretrained_cfg = cfg

        # Freeze the pretrained model
        for param in pretrained_policy.parameters():
            param.requires_grad = False
        pretrained_policy.eval()

        # Override DDIM inference steps if configured
        if self.rl_num_inference_steps is not None:
            log.info(f"  Overriding num_inference_steps: {pretrained_policy.num_inference_steps} -> {self.rl_num_inference_steps}")
            pretrained_policy.num_inference_steps = self.rl_num_inference_steps

        log.info("Pretrained diffusion policy loaded and frozen successfully")
        log.info(f"  Action horizon: {pretrained_policy.sparse_action_horizon}")
        log.info(f"  Action dim: {pretrained_policy.action_dim}")
        log.info(f"  Obs feature dim: {pretrained_policy.obs_feature_dim}")
        log.info(f"  Num inference steps: {pretrained_policy.num_inference_steps}")

        return pretrained_policy
        
    # def compute_confidence(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    #     """
    #     Compute confidence score for given state and noise using the pretrained flow matching policy.
        
    #     Args:
    #         state: (B, cond_steps, obs_dim) - current state chunk
    #         noise: (B, horizon_steps, action_dim) - initial noise for action generation
            
    #     Returns:
    #         confidence: (B, 1) - normalized confidence score
    #     """
    #     # Use get_action_and_confidence and discard the action
    #     _, confidence = self.get_action_and_confidence(state, noise)
    #     return confidence
    
    def compute_confidence_simple(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Simple version: Compute confidence weight with raw confidence clipping and inverse mapping.
        
        Args:
            state: (B, cond_steps, obs_dim) - current state chunk
            noise: (B, horizon_steps, action_dim) - initial noise for action generation
            
        Returns:
            confidence_weight: (B, 1) - confidence weight in [1, 0] range
        """
        # Use get_action_and_confidence_simple and discard the action
        _, confidence_weight = self.get_action_and_confidence_simple(state, noise)
        return confidence_weight
    
    # # deprecated - kept for reference
    # def get_action_and_confidence(self, state: torch.Tensor, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Get both action and confidence score for given state and noise using the same initial noise.
        
    #     Args:
    #         state: (B, cond_steps, obs_dim) - current state chunk
    #         noise: (B, horizon_steps, action_dim) - initial noise for action generation
            
    #     Returns:
    #         action: (B, horizon_steps, action_dim) - predicted action
    #         confidence: (B, 1) - normalized confidence score
    #     """
    #     B = state.shape[0]  # batch size
        
    #     # Use the pretrained flow matching policy to compute both action and confidence
    #     with torch.no_grad():
    #         # Ensure both state and noise are on the same device as the pretrained policy
    #         state = state.to(self.device)
    #         noise = noise.to(self.device)
            
    #         # The pretrained flow matching policy expects state in proper format
    #         cond = {"state": state}  # (B, cond_steps, obs_dim)
            
    #         # Skip expensive confidence computation if not using confidence estimation
    #         compute_conf = self.use_confidence_estimation
    #         sample = self.pretrained_flow_policy(cond, deterministic=False, init_noise=noise, compute_confidence=compute_conf)
            
    #         action = sample.trajectories  # (B, horizon_steps, action_dim)
            
    #         # Get confidence score from the sample (simplified - use default for now)
    #         # if (self.use_confidence_estimation and 
    #         #     hasattr(sample, 'confidence') and sample.confidence is not None and 
    #         #     self.confidence_metric in sample.confidence):
    #         if self.use_confidence_estimation:
    #             assert hasattr(sample, 'confidence'), "Sample must have 'confidence' attribute when using confidence estimation"
    #             assert self.confidence_metric in sample.confidence, f"Confidence metric '{self.confidence_metric}' not found in sample"
    #             confidence_raw = sample.confidence[self.confidence_metric]  # (B,)
    #             confidence = confidence_raw.unsqueeze(-1)  # (B, 1)
    #         else:
    #             # Fallback: use a default confidence of 1.0 (when disabled or missing)
    #             confidence = torch.ones(B, 1, device=state.device)  # (B, 1)
                

    #     # Apply normalization to confidence if specified
    #     if self.confidence_normalization == "running_stats":
    #         # Update running statistics using Chan/Welford's parallel algorithm
    #         with torch.no_grad():
    #             Nb = torch.tensor(B, device=self.device, dtype=torch.float32)
    #             No = self.confidence_count
    #             N = No + Nb
                
    #             batch_mean = confidence.mean()  # scalar
    #             batch_var = confidence.var(unbiased=False)  # scalar, population variance
    #             M2_b = batch_var * Nb  # Sum of squared deviations for batch
                
    #             # Update mean
    #             delta = batch_mean - self.confidence_mean
    #             new_mean = self.confidence_mean + delta * (Nb / N)
                
    #             # Update M2 (sum of squared deviations) with between-means correction
    #             new_M2 = self.confidence_M2 + M2_b + (delta * delta) * (No * Nb / N)
                
    #             # Store updated statistics
    #             self.confidence_mean = new_mean
    #             self.confidence_M2 = new_M2
    #             self.confidence_count = N
                
    #             # Compute running variance and std
    #             running_var = self.confidence_M2 / torch.clamp(self.confidence_count, min=1.0)
    #             self.confidence_std = torch.sqrt(running_var + 1e-12)
            
    #         # Normalize
    #         confidence = (confidence - self.confidence_mean) / (self.confidence_std + 1e-8)  # (B, 1)
        
    #     return action, confidence
    
    def get_action_and_confidence_simple(self, state: torch.Tensor, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple version: Get both action and confidence with u(w) = 1/(1+(2w)^2) transformation.
        
        Larger w means more uncertain, so u(w) should be inverse:
        - w=0 (certain) → u=1 (high weight for BC loss)
        - w=0.5 (threshold) → u=0.5
        - w=2 (uncertain) → u≈0.059 (low weight for BC loss)
        
        Args:
            state: (B, cond_steps, obs_dim) - current state chunk
            noise: (B, horizon_steps, action_dim) - initial noise for action generation
            
        Returns:
            action: (B, horizon_steps, action_dim) - predicted action
            confidence_weight: (B, 1) - confidence weight u(w) in [0, 1] range
        """
        B = state.shape[0]  # batch size

        # Ensure both state and noise are on the same device
        state = state.to(self.device)
        noise = noise.to(self.device)

        # For diffusion policy: state is pre-encoded features (B, obs_feature_dim)
        # Squeeze if state has cond_steps dimension: (B, cond_steps, obs_dim) -> (B, obs_dim)
        if state.dim() == 3:
            state = state.view(B, -1)  # Flatten to (B, cond_steps * obs_dim)

        # Use pretrained diffusion policy to get action (normalized)
        result = self.pretrained_policy.predict_action_from_features(
            sparse_nobs_encode=state,
            init_noise=noise,
            unnormalize=False,
        )
        action = result["sparse"]  # (B, horizon_steps, action_dim) normalized

        if self.clip_action:
            action = torch.clamp(action, -1.0, 1.0)

        # Confidence estimation deprecated - return uniform weights
        confidence_weight = torch.ones(B, 1, device=self.device)
        
        return action, confidence_weight
    
    def compute_intrinsic_reward(self, next_state: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic reward based on confidence scores at next state.

        NOTE: Confidence estimation is deprecated - always returns zeros.

        Args:
            next_state: (B, cond_steps, obs_dim) - next state

        Returns:
            intrinsic_reward: (B, 1) - always zeros (confidence deprecated)
        """
        return torch.zeros(next_state.shape[0], 1, device=self.device)
    
    def update_target_networks(self, tau: float = 0.005):
        """
        Update target networks using Polyak averaging.
        
        Args:
            tau: Polyak averaging coefficient (target = tau * current + (1-tau) * target)
        """
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    

    def get_exploration_action(self, state: torch.Tensor, num_samples: int = 10, 
                              exploration_strategy: str = "max_q_std", training_step: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get exploration action using specified strategy.
        
        This method is used for online exploration to maximize diversity in collected data.
        Supports two strategies:
        - max_q_std: Select action with highest Q-std across ensemble (epistemic uncertainty)
        - max_q_min: Select action with highest minimum Q-value across ensemble (optimistic)
        
        If training_step < replay_flow_warmup_steps, uses single sample regardless of strategy.
        
        Args:
            state: (B, cond_steps, obs_dim) - current state
            num_samples: Number of noise samples to evaluate
            exploration_strategy: "max_q_std" or "max_q_min"
            training_step: Current training step (to check against warmup)
            
        Returns:
            selected_action: (B, horizon_steps, action_dim) - selected action
            selected_noise: (B, horizon_steps, action_dim) - corresponding noise
        """
        B = state.shape[0]
        device = state.device
        
        # During warmup, use single sample regardless of strategy
        if training_step <= self.replay_flow_warmup_steps:
            # Single sample during warmup
            noise = torch.randn(B, self.horizon_steps, self.action_dim, device=device)
            action = self.get_action(state, noise)
            return action, noise
        
        # After warmup, use specified exploration strategy
        # Sample multiple noise vectors
        noise_samples = torch.randn(num_samples, B, self.horizon_steps, self.action_dim, device=device)
        
        # Expand state for batch processing
        state_expanded = state.unsqueeze(0).expand(num_samples, -1, -1, -1)
        state_flat = state_expanded.reshape(num_samples * B, *state.shape[1:])
        noise_flat = noise_samples.reshape(num_samples * B, self.horizon_steps, self.action_dim)
        
        # Get actions for all noise samples
        with torch.no_grad():
            actions_flat = self.get_action(state_flat, noise_flat)  # (num_samples * B, horizon_steps, action_dim)
            
            # Get Q-values from all critics for all samples
            q_all = self.critic(state_flat, noise_flat, actions_flat, return_all=True)  # List of (num_samples * B, 1)
            
            # Stack Q-values from ensemble: (ensemble_size, num_samples * B, 1)
            q_stacked = torch.stack(q_all, dim=0)
            
            # Reshape to separate samples: (ensemble_size, num_samples, B, 1)
            q_reshaped = q_stacked.view(len(q_all), num_samples, B, 1)
            
            if exploration_strategy == "max_q_min":
                # Select action with max of min Q-value across ensemble
                # min over ensemble for each sample: (num_samples, B, 1)
                q_min = q_reshaped.min(dim=0)[0]
                
                # Select sample with max min Q for each batch element
                max_min_indices = q_min.squeeze(-1).argmax(dim=0)  # (B,)
                selection_indices = max_min_indices

            elif exploration_strategy == 'max_q_std':  # max_q_std (default)
                # Select action with max Q-std across ensemble
                # Compute std across ensemble for each sample: (num_samples, B, 1)
                q_std = q_reshaped.std(dim=0)
                
                # Select sample with max std for each batch element
                max_std_indices = q_std.squeeze(-1).argmax(dim=0)  # (B,)
                selection_indices = max_std_indices
            
            elif exploration_strategy == "max_q_std_filtered_by_min":
                # Select top 3 samples based on min q value, then from those select max std
                # Get min Q-value across ensemble for each sample: (num_samples, B, 1)
                q_min = q_reshaped.min(dim=0)[0]
                
                # Get top 3 samples for each batch element based on min Q
                top_k = min(3, num_samples)  # In case we have fewer than 3 samples
                top_q_values, top_indices = q_min.squeeze(-1).topk(top_k, dim=0)  # (top_k, B)
                
                # Compute std for the top-k samples only
                # Extract Q-values for top-k samples: (ensemble_size, top_k, B, 1)
                top_q_reshaped = torch.stack([
                    q_reshaped[:, top_indices[k], torch.arange(B)]  # (ensemble_size, B, 1)
                    for k in range(top_k)
                ], dim=1)  # (ensemble_size, top_k, B, 1)
                
                # Compute std across ensemble for top-k samples: (top_k, B, 1)
                top_q_std = top_q_reshaped.std(dim=0)
                
                # Select the sample with max std from the top-k
                max_std_in_topk_indices = top_q_std.squeeze(-1).argmax(dim=0)  # (B,)
                
                # Get the actual sample indices from the top-k indices
                selection_indices = torch.gather(top_indices, 0, max_std_in_topk_indices.unsqueeze(0)).squeeze(0)  # (B,)
            else:
                raise ValueError(f"Unknown exploration strategy: {exploration_strategy}")
            # Reshape actions back
            actions_reshaped = actions_flat.view(num_samples, B, self.horizon_steps, self.action_dim)
            
            # Select actions based on strategy
            selected_actions = torch.stack([
                actions_reshaped[selection_indices[b], b] for b in range(B)
            ])
            
            # Select corresponding noise
            selected_noise = torch.stack([
                noise_samples[selection_indices[b], b] for b in range(B)
            ])
        
        return selected_actions, selected_noise
    
    def actor_loss(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
        current_actions: torch.Tensor,  # (B, H, A) - total actions (pretrained + residual)
        q_values: torch.Tensor,         # (B, 1)
        confidence: torch.Tensor,       # (B, 1)
        pretrained_actions: torch.Tensor,   # (B, H, A)
        next_state: Optional[torch.Tensor] = None,
        next_noise: Optional[torch.Tensor] = None,
        training_step: int = 0,
        q_overestimation: Optional[torch.Tensor] = None,  # (B,1) if provided
        data_source: Optional[torch.Tensor] = None,  # (B,1) - 0 for online, 1 for expert
    ) -> Dict[str, torch.Tensor]:
        """
        Actor loss for residual RL with multi-z sampling (always assumes sample_multi_z_for_actor_loss=True).
        
        Different behavior based on training_step vs q_filtering_warmup_steps:
        1. During warmup (≤ q_filtering_warmup_steps): Simple Q + BC loss, no filtering
        2. After warmup: BC filtering and self-imitation when conditions are met
        """
        B = state.shape[0]
        K = self.num_multi_z_for_actor_loss
        
        # Sample K noise vectors for each state
        noise_samples = torch.randn(B, K, *noise.shape[1:], device=self.device)  # (B, K, H, A)
        
        # Compute actions for all K noise samples
        state_expanded = state.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, cond_steps, obs_dim)
        state_flat = state_expanded.reshape(B * K, *state.shape[1:])  # (B*K, cond_steps, obs_dim)
        noise_flat = noise_samples.reshape(B * K, *noise_samples.shape[2:])  # (B*K, H, A) - use noise_samples shape, not noise shape!
        
        # Get actions with pretrained actions returned
        actions_flat, pretrained_actions_flat = self.get_action(state_flat, noise_flat, return_pretrained_actions=True)  # (B*K, H, A)
        
        # Reshape back to (B, K, H, A)
        actions_samples = actions_flat.reshape(B, K, *actions_flat.shape[1:])  # (B, K, H, A)
        pretrained_actions_samples = pretrained_actions_flat.reshape(B, K, *pretrained_actions_flat.shape[1:])  # (B, K, H, A)
        
        # Compute Q-values for current actions (with gradients for actor loss)
        q_current_flat = self.critic(state_flat, noise_flat, actions_flat)  # (B*K, 1)
        q_current_samples = q_current_flat.reshape(B, K)  # (B, K)
        
        # Compute Q-values for pretrained actions (no gradients)
        with torch.no_grad():
            q_pretrained_flat = self.critic(state_flat, noise_flat, pretrained_actions_flat)  # (B*K, 1)
            q_pretrained_samples = q_pretrained_flat.reshape(B, K)  # (B, K)
        
        # Check if we're in warmup phase
        in_warmup = training_step <= self.q_filtering_warmup_steps
        
        if in_warmup:
            # WARMUP PHASE: Simple loss without filtering or self-imitation
            
            # Q-value loss: -mean(Q(s, a^current_k))
            q_loss_per_batch = q_current_samples.mean(dim=1)  # (B,) - mean over K samples
            
            # Apply disable_q_loss_for_expert_data if enabled
            online_mask = None
            if self.disable_q_loss_for_expert_data and data_source is not None:
                # Mask Q loss for expert data: only apply Q loss to online data (data_source == 0)
                online_mask = (data_source == 0).float().squeeze(-1)  # (B, 1) -> (B,) - 1.0 for online, 0.0 for expert
                q_loss_per_batch = q_loss_per_batch * online_mask  # (B,)
            
            q_loss = -q_loss_per_batch.mean()  # scalar
            
            # Normalize Q-loss by mean absolute Q-value for stability (like FQL)
            if self.use_q_normalization:
                # Compute normalization constant only from the Q-values we're actually using for loss
                if online_mask is not None:
                    # Only normalize by online Q-values when we're masking expert data
                    online_mask_expanded = online_mask.unsqueeze(-1).expand(-1, K)  # (B, K)
                    online_q_values = q_current_samples * online_mask_expanded  # Zero out expert Q-values
                    online_count = online_mask_expanded.sum()  # Count of online samples
                    if online_count > 0:
                        q_abs_mean = (online_q_values.abs().sum() / online_count).detach()  # Mean of online Q-values only
                    else:
                        q_abs_mean = q_current_samples.abs().mean().detach()  # Fallback to all if no online samples
                else:
                    # Use all Q-values for normalization
                    q_abs_mean = q_current_samples.abs().mean().detach()
                
                if q_abs_mean > 1e-8:  # Avoid division by zero
                    q_loss_scale = 1.0 / q_abs_mean
                    q_loss = q_loss_scale * q_loss
                
            # BC regularization loss: mean(||a^current_k - a^pre_k||²)
            action_diff = actions_samples - pretrained_actions_samples  # (B, K, horizon_steps, action_dim)
            # Compute MSE across action dimensions for each timestep, then average across time
            mse_per_timestep = (action_diff ** 2).mean(dim=-1)  # (B, K, horizon_steps)
            # Average across timesteps to get per-sample MSE: (B, K)
            mse_per_sample = mse_per_timestep.mean(dim=-1)  # (B, K)
            # Mean across K samples and average across batch
            filtered_bc_loss = mse_per_sample.mean(dim=1).mean()  # scalar
            
            # Total loss for warmup
            total_loss = q_loss + self.bc_loss_weight * filtered_bc_loss
            
            # Set placeholders for metrics
            bc_filter = torch.ones(B, 1, device=self.device)
            better_percentage = torch.tensor(0.0, device=self.device)
            q_advantage = torch.zeros(B, 1, device=self.device)
            avg_q_current = q_current_samples.mean(dim=1, keepdim=True)
            avg_q_pretrained = q_pretrained_samples.mean(dim=1, keepdim=True)
            filtered_self_imitation_loss = torch.tensor(0.0, device=self.device)
            
        else:
            # POST-WARMUP PHASE: With BC filtering and self-imitation
            
            # Compute per-sample Q-advantages for filtering (no gradients needed for filtering)
            with torch.no_grad():
                # Per-sample Q-advantage: q_current_bk - q_pretrained_bk for each k
                q_advantage_per_sample = q_current_samples - q_pretrained_samples  # (B, K)
                better_than_expert_per_sample = (q_advantage_per_sample > 0).float()  # (B, K) - 1.0 if better, 0.0 if worse
                
                # Compute average metrics for logging (keep these for backward compatibility)
                avg_q_current = q_current_samples.mean(dim=1, keepdim=True)  # (B, 1)
                avg_q_pretrained = q_pretrained_samples.mean(dim=1, keepdim=True)  # (B, 1)
                q_advantage = avg_q_current - avg_q_pretrained  # (B, 1) - average advantage for logging
                better_than_expert = (q_advantage > 0).float()  # (B, 1) - average for logging
                better_percentage = better_than_expert_per_sample.mean()  # Fraction of all (B,K) samples where policy is better
                
                # Apply soft Q-filtering on per-sample basis
                if self.use_soft_q_filtering:
                    # Filter when better than expert AND Q is underestimated
                    if q_overestimation is not None:
                        # q_overestimation < 0 means underestimation
                        q_underestimated = (q_overestimation < self.q_underestimation_threshold).float()  # (B, 1)
                        q_underestimated_expanded = q_underestimated.expand(-1, K)  # (B, K)
                        # Apply filter if BOTH conditions are true: better_than_expert AND q_underestimated
                        # Per-sample: should_filter_bk = better_than_expert_bk * q_underestimated_b
                        should_filter_per_sample = better_than_expert_per_sample * q_underestimated_expanded  # (B, K)
                        bc_filter_expanded = 1.0 - should_filter_per_sample  # (B, K)
                    else:
                        # Fallback to original behavior if q_overestimation not provided
                        bc_filter_expanded = 1.0 - better_than_expert_per_sample  # (B, K)
                else:
                    bc_filter_expanded = torch.ones_like(better_than_expert_per_sample, device=self.device)  # (B, K)
                
                # Override bc_filter for expert data if always_retain_bc_loss_for_expert_data is True
                if self.always_retain_bc_loss_for_expert_data and data_source is not None:
                    # For expert data (data_source == 1), always set bc_filter to 1.0
                    expert_mask = (data_source == 1).float()  # (B, 1) - 1.0 for expert, 0.0 for online
                    expert_mask_expanded = expert_mask.expand(-1, K)  # (B, K)
                    # Override bc_filter: 1.0 for expert data, keep original for online data
                    bc_filter_expanded = expert_mask_expanded + (1.0 - expert_mask_expanded) * bc_filter_expanded  # (B, K)
                
                # Keep a (B, 1) version for backward compatibility with logging
                bc_filter = bc_filter_expanded.mean(dim=1, keepdim=True)  # (B, 1) - average for logging
            
            # Q-value loss: -mean(Q(s, a_k)) (direct average, no softmax weights)
            q_loss_per_batch = q_current_samples.mean(dim=1)  # (B,) - average over K samples
            
            # Apply disable_q_loss_for_expert_data if enabled
            online_mask = None
            if self.disable_q_loss_for_expert_data and data_source is not None:
                # Mask Q loss for expert data: only apply Q loss to online data (data_source == 0)
                online_mask = (data_source == 0).float().squeeze(-1)  # (B, 1) -> (B,) - 1.0 for online, 0.0 for expert
                q_loss_per_batch = q_loss_per_batch * online_mask  # (B,)
            
            q_loss = -q_loss_per_batch.mean()  # scalar
            
            # Normalize Q-loss by mean absolute Q-value for stability (like FQL)
            if self.use_q_normalization:
                # Compute normalization constant only from the Q-values we're actually using for loss
                if online_mask is not None:
                    # Only normalize by online Q-values when we're masking expert data
                    online_mask_expanded = online_mask.unsqueeze(-1).expand(-1, K)  # (B, K)
                    online_q_values = q_current_samples * online_mask_expanded  # Zero out expert Q-values
                    online_count = online_mask_expanded.sum()  # Count of online samples
                    if online_count > 0:
                        q_abs_mean = (online_q_values.abs().sum() / online_count).detach()  # Mean of online Q-values only
                    else:
                        q_abs_mean = q_current_samples.abs().mean().detach()  # Fallback to all if no online samples
                else:
                    # Use all Q-values for normalization
                    q_abs_mean = q_current_samples.abs().mean().detach()
                
                if q_abs_mean > 1e-8:  # Avoid division by zero
                    q_loss_scale = 1.0 / q_abs_mean
                    q_loss = q_loss_scale * q_loss
            
            # Compute BC-style loss with filtering: ||current_action - pretrained_action||²
            action_diff = actions_samples - pretrained_actions_samples  # (B, K, horizon_steps, action_dim)
            # Compute MSE across action dimensions for each timestep, then average across time
            mse_per_timestep = (action_diff ** 2).mean(dim=-1)  # (B, K, horizon_steps)
            # Average across timesteps to get per-sample MSE: (B, K)
            mse_per_sample = mse_per_timestep.mean(dim=-1)  # (B, K)
            # Apply filtering and equal weighting (no softmax weights, just equal average)
            uniform_weights = torch.ones(B, K, device=self.device) / K  # (B, K) - equal weights
            weighted_filtered_mse = uniform_weights * bc_filter_expanded * mse_per_sample  # (B, K)
            # Sum across K samples and average across batch
            filtered_bc_loss = weighted_filtered_mse.sum(dim=1).mean()  # scalar
            
            # Add self-imitation loss if enabled and conditions are met
            if self.self_imitation_for_actor_loss and self.use_soft_q_filtering:
                # Check conditions for self-imitation (same as BC filter conditions)
                # Only apply when Q is underestimated AND current Q > pretrained Q
                if q_overestimation is not None:
                    q_underestimated = (q_overestimation < self.q_underestimation_threshold).float()  # (B, 1)
                    should_apply_si = better_than_expert * q_underestimated  # (B, 1)
                else:
                    should_apply_si = better_than_expert  # (B, 1)
                
                # Distributional self-imitation (always use this variant as specified)
                with torch.no_grad():
                    # Select top actions based on Q-values (configurable percentage)
                    k_top = max(1, K // self.topk_divisor_for_self_imitation)  # Top percentage of K samples
                    
                    # Find top-k indices based on Q-values for each batch element
                    _, teacher_indices = q_current_samples.topk(k_top, dim=1)  # (B, k_top)
                    
                    # Gather the top-k actions as teacher actions
                    gather_idx = teacher_indices.view(B, k_top, 1, 1).expand(-1, -1, self.horizon_steps, self.action_dim)
                    teacher_actions = actions_samples.gather(1, gather_idx).detach()  # (B, k_top, H, A)
                    
                    # Create mask for top-k indices
                    is_topk = torch.zeros(B, K, dtype=torch.bool, device=self.device)
                    is_topk.scatter_(1, teacher_indices, True)
                
                # Fully vectorized computation of self-imitation loss
                # Compute pairwise L2 distances between all actions and teacher actions
                actions_expanded = actions_samples.unsqueeze(2)  # (B, K, 1, H, A)
                teachers_expanded = teacher_actions.unsqueeze(1)  # (B, 1, k_top, H, A)
                
                # Compute squared L2 distances
                pairwise_distances = ((actions_expanded - teachers_expanded) ** 2).sum(dim=(3, 4))  # (B, K, k_top)
                
                # Find nearest teacher for each action
                min_distances, nearest_teacher_indices = pairwise_distances.min(dim=2)  # (B, K)
                
                # Gather nearest teacher actions using advanced indexing - fully vectorized
                # nearest_teacher_indices: (B, K) with values from 0 to k_top-1
                # teacher_actions: (B, k_top, H, A)
                # We want: for each b,k -> teacher_actions[b, nearest_teacher_indices[b,k], :, :]
                
                # Use advanced indexing without unnecessary expansion
                batch_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, K)  # (B, K)
                # Flatten indices for gathering
                batch_idx_flat = batch_idx.reshape(-1)  # (B*K,)
                teacher_idx_flat = nearest_teacher_indices.reshape(-1)  # (B*K,)
                # Gather and reshape
                nearest_teachers_all = teacher_actions[batch_idx_flat, teacher_idx_flat].reshape(B, K, self.horizon_steps, self.action_dim)  # (B, K, H, A)
                
                # Compute imitation loss (MSE between current and nearest teacher)
                imitation_diff = actions_samples - nearest_teachers_all  # (B, K, H, A)
                imitation_loss_per_sample = (imitation_diff ** 2).mean(dim=(2, 3))  # (B, K)
                
                # Zero out loss for top-k samples (they don't need to imitate)
                imitation_loss_per_sample = imitation_loss_per_sample * (~is_topk).float()  # (B, K)
                
                # Apply self-imitation filter (same conditions as BC filter)
                should_apply_si_expanded = should_apply_si.expand(-1, K)  # (B, K)
                filtered_imitation_loss_per_sample = should_apply_si_expanded * imitation_loss_per_sample  # (B, K)
                
                # Average with uniform weights
                weighted_imitation_loss = uniform_weights * filtered_imitation_loss_per_sample  # (B, K)
                filtered_self_imitation_loss = weighted_imitation_loss.sum(dim=1).mean()  # scalar
                
                # Check if winner_loser mode is enabled
                if self.winner_loser:
                    # Only top-k samples contribute to Q loss
                    # Recompute Q loss using only top-k samples
                    topk_q_values = q_current_samples[is_topk]  # Select only top-k Q-values
                    
                    # Apply online mask if needed (for top-k samples only)
                    if online_mask is not None:
                        # Expand online_mask to match K dimension and select top-k
                        online_mask_expanded = online_mask.unsqueeze(-1).expand(-1, K)  # (B, K)
                        online_mask_topk = online_mask_expanded[is_topk]  # Select only top-k mask values
                        topk_q_values_masked = topk_q_values * online_mask_topk  # Apply mask to top-k Q-values
                        topk_online_count = online_mask_topk.sum()
                        winner_q_loss = -topk_q_values_masked.sum() / topk_online_count if topk_online_count > 0 else torch.tensor(0.0, device=self.device)
                        
                        # Compute normalization from ALL online Q-values (not just top-k) for consistency with warmup
                        if self.use_q_normalization:
                            # Use the same normalization as computed earlier (lines 927-938)
                            # which is based on all online samples, not just top-k
                            if q_abs_mean > 1e-8:  # q_abs_mean was computed earlier from all samples
                                winner_q_loss = winner_q_loss / q_abs_mean
                    else:
                        # No expert data masking, just use mean of top-k
                        winner_q_loss = -topk_q_values.mean()
                        
                        # Compute normalization from ALL Q-values (not just top-k) for consistency
                        if self.use_q_normalization:
                            # Use all Q-values for normalization, same as warmup stage
                            all_q_abs_mean = q_current_samples.abs().mean().detach()
                            if all_q_abs_mean > 1e-8:
                                winner_q_loss = winner_q_loss / all_q_abs_mean
                    
                    # Total loss uses winner_q_loss instead of regular q_loss
                    # Replace q_loss with winner_q_loss for logging consistency
                    q_loss = winner_q_loss
                    total_loss = q_loss + self.bc_loss_weight * filtered_bc_loss + self.self_imitation_loss_weight * filtered_self_imitation_loss
                else:
                    # Standard mode: all samples contribute to Q loss
                    total_loss = q_loss + self.bc_loss_weight * filtered_bc_loss + self.self_imitation_loss_weight * filtered_self_imitation_loss
            else:
                # No self-imitation loss
                filtered_self_imitation_loss = torch.tensor(0.0, device=self.device)
                total_loss = q_loss + self.bc_loss_weight * filtered_bc_loss
        
        # Compute additional metrics for logging
        with torch.no_grad():
            residual_norm = ((actions_samples - pretrained_actions_samples) ** 2).mean().sqrt()  # RMS of residual actions
            pretrained_q_values = avg_q_pretrained if not in_warmup else q_pretrained_samples.mean(dim=1, keepdim=True)
            q_values = avg_q_current if not in_warmup else q_current_samples.mean(dim=1, keepdim=True)
        
        return {
            'actor_total': total_loss,
            'actor_q_loss': q_loss,
            'actor_residual_loss': filtered_bc_loss,  # This is the filtered BC loss (equivalent to residual loss)
            'actor_bc_loss': filtered_bc_loss,  # BC-style loss for compatibility with logging
            'actor_future_confidence_loss': torch.tensor(0.0, device=self.device),  # Not used
            'actor_gfc_loss': torch.tensor(0.0, device=self.device),  # Not used
            'actor_dtr_loss': torch.tensor(0.0, device=self.device),  # Not used
            'actor_self_imitation_loss': filtered_self_imitation_loss,
            # Metrics
            'q_advantage_mean': q_advantage.mean() if not in_warmup else torch.tensor(0.0, device=self.device),
            'better_than_expert_percentage': better_percentage,
            'pretrained_q_mean': pretrained_q_values.mean(),
            'current_q_mean': q_values.mean(),
            'residual_norm': residual_norm,
            'q_filtering_active': bc_filter.mean(),  # Mean of filtering mask
        }
    
    def critic_loss(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
        action: torch.Tensor,
        target_q: torch.Tensor,
        data_source: Optional[torch.Tensor] = None,  # (B,1) - 0 for online, 1 for expert
    ) -> Dict[str, torch.Tensor]:
        """
        Compute critic loss for both Q-networks (double Q-learning).
        
        Args:
            state: (B, cond_steps, obs_dim) - chunked current state
            noise: (B, horizon_steps, action_dim) - chunked noise
            action: (B, horizon_steps, action_dim) - chunked action
            target_q: (B, 1) - target Q-values
            data_source: (B, 1) - 0 for online, 1 for expert data (optional)
            
        Returns:
            loss_dict: Dictionary containing loss components
        """
        # Get predictions from all Q-networks in ensemble
        q_all = self.critic(state, noise, action, return_all=True)
        
        # Compute loss for all networks in ensemble
        total_loss = 0
        loss_dict = {}
        
        for i, q_pred in enumerate(q_all):
            # Compute TD loss based on selected loss type
            if self.td_loss == "mse":
                # Mean Squared Error loss
                if self.disable_td_loss_for_expert_data and data_source is not None:
                    # Mask TD loss for expert data: only apply TD loss to online data (data_source == 0)
                    online_mask = (data_source == 0).float()  # (B, 1) - 1.0 for online, 0.0 for expert
                    # Compute per-sample MSE loss
                    per_sample_loss = F.mse_loss(q_pred, target_q, reduction='none')  # (B, 1)
                    # Apply mask and reduce
                    q_loss = (per_sample_loss * online_mask).mean()
                else:
                    # Standard TD loss for all samples
                    q_loss = F.mse_loss(q_pred, target_q)
                    
            elif self.td_loss == "huber":
                # Huber loss (smooth L1 loss) - more robust to outliers
                if self.disable_td_loss_for_expert_data and data_source is not None:
                    online_mask = (data_source == 0).float()
                    per_sample_loss = F.smooth_l1_loss(q_pred, target_q, reduction='none', beta=1.0)  # (B, 1)
                    q_loss = (per_sample_loss * online_mask).mean()
                else:
                    q_loss = F.smooth_l1_loss(q_pred, target_q, beta=1.0)
                    
            elif self.td_loss == "bce":
                # Binary Cross Entropy with Logits loss
                # q_pred are raw logits, target_q should be in [0,1]
                # Assert that target values are in valid range
                assert torch.all(target_q >= 0.0) and torch.all(target_q <= 1.0), \
                    f"BCE loss requires target Q-values in [0,1], got range [{target_q.min():.4f}, {target_q.max():.4f}]"
                
                if self.disable_td_loss_for_expert_data and data_source is not None:
                    online_mask = (data_source == 0).float()
                    # Use BCEWithLogitsLoss for numerical stability
                    per_sample_loss = F.binary_cross_entropy_with_logits(q_pred, target_q, reduction='none')  # (B, 1)
                    q_loss = (per_sample_loss * online_mask).mean()
                else:
                    q_loss = F.binary_cross_entropy_with_logits(q_pred, target_q)
            else:
                raise ValueError(f"Unknown td_loss type: {self.td_loss}")
            
            # Add asymmetric regression loss to suppress overestimation
            if self.asymmetric_regression:
                # For BCE, q_pred is logits, so we need to apply sigmoid first
                if self.td_loss == "bce":
                    q_pred_for_asym = torch.sigmoid(q_pred)
                else:
                    q_pred_for_asym = q_pred
                
                # Compute overestimation penalty: max(0, q_pred - target_q)^2
                overestimation = torch.relu(q_pred_for_asym - target_q)  # (B, 1)
                
                # Apply expert data masking if enabled
                if self.disable_td_loss_for_expert_data and data_source is not None:
                    online_mask = (data_source == 0).float()
                    asym_loss = (overestimation.pow(2) * online_mask).mean()
                else:
                    asym_loss = overestimation.pow(2).mean()
                
                # Add asymmetric loss to the main TD loss
                q_loss = q_loss + 0.5 * asym_loss
            
            total_loss += q_loss
            
            # Store individual losses for debugging (first 3 critics)
            if i < 3:
                loss_dict[f'q{i+1}_loss'] = q_loss
        
        loss_dict['critic_loss'] = total_loss
        return loss_dict
    
    def dynamics_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute dynamics model loss.
        
        Args:
            state: (B, cond_steps, obs_dim) - chunked current state
            action: (B, horizon_steps, action_dim) - chunked action
            next_state: (B, cond_steps, obs_dim) - chunked actual next state
            
        Returns:
            loss_dict: Dictionary containing loss components
        """
        predicted_next_state = self.dynamics(state, action)
        dynamics_loss = F.mse_loss(predicted_next_state, next_state)
        
        return {
            'dynamics_loss': dynamics_loss,
        }
    
    def get_action(self, state: torch.Tensor, noise: torch.Tensor, return_pretrained_actions: bool = False) -> torch.Tensor:
        """
        Get action from actor given state and noise. Can be overridden for residual RL.
        
        Args:
            state: (B, cond_steps, obs_dim) - current state
            noise: (B, horizon_steps, action_dim) - noise for action generation
            return_pretrained_actions: If True, return tuple (current_actions, pretrained_actions)
            
        Returns:
            action: (B, horizon_steps, action_dim) - action from actor
            OR if return_pretrained_actions:
            (action, pretrained_actions): tuple of actions
        """
        current_actions = self.actor(state, noise)
        if self.clip_action:
            current_actions = torch.clamp(current_actions, -1.0, 1.0)
        
        if return_pretrained_actions:
            # Get pretrained actions for comparison (normalized)
            with torch.no_grad():
                B = state.shape[0]
                # Flatten state if needed for diffusion policy
                state_flat = state.view(B, -1) if state.dim() == 3 else state
                result = self.pretrained_policy.predict_action_from_features(
                    sparse_nobs_encode=state_flat,
                    init_noise=noise,
                    unnormalize=False,
                )
                pretrained_actions = result["sparse"]  # (B, horizon_steps, action_dim) normalized
            return current_actions, pretrained_actions
        
        return current_actions
    
    def loss(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        gamma: float = 0.99,
        training_step: int = 0,
        q_overestimation: Optional[torch.Tensor] = None,
        n_steps: Optional[torch.Tensor] = None,
        data_source: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for the distilled RL model.
        
        Args:
            state: (B, cond_steps, obs_dim) - chunked current state
            noise: (B, horizon_steps, action_dim) - initial noise for action generation
            action: (B, horizon_steps, action_dim) - action chunk
            next_state: (B, cond_steps, obs_dim) - chunked next state
            reward: (B, 1) - reward
            done: (B, 1) - done flag
            gamma: float - discount factor
            
        Returns:
            loss_dict: Dictionary containing all loss components
        """
        batch_size = state.shape[0]
        
        # All models work with the entire chunks - no need to extract single states/actions
        
        # Get pretrained policy actions and confidence using the same initial noise (simple version)
        with torch.no_grad():
            pretrained_actions, confidence = self.get_action_and_confidence_simple(state, noise)  # (B, horizon_steps, action_dim), (B, 1)
        
        # Note: intrinsic reward is now pre-computed and stored in replay buffer
        # The 'reward' parameter already includes intrinsic reward if enabled
        
        # Get current actions from actor for Q-value computation
        current_actions = self.get_action(state, noise)

        q_values = self.critic(state, noise, current_actions)  # (B, 1)
        
        # Clip Q-values if specified
        # q_values = self._clip_q_values(q_values)

        # Compute target Q-values using target networks (SAC-style)
        with torch.no_grad():
            # Sample noise for next actions (always single sample for actor loss consistency)
            next_noise = torch.randn(batch_size, action.shape[1], self.action_dim, device=self.device)
            
            if not self.multi_sample_next_noise:
                next_actions = self.get_action(next_state, next_noise)  # (B, horizon_steps, action_dim) - use get_action for residual RL compatibility
                if self.clip_action:
                    next_actions = torch.clamp(next_actions, -1.0, 1.0)
                # Use target critic ensemble for stable targets
                target_next_q = self.target_critic(next_state, next_noise, next_actions)  # (B, 1) - already min across ensemble
                # target_next_q = self._clip_q_values(target_next_q)
                
                # Apply appropriate discount factor based on training configuration
                if self.use_n_step:
                    # For n-step returns, use gamma^n_step
                    gamma_effective = gamma ** n_steps.float()  # (B, 1)
                elif self.discount_within_horizon:
                    # For discount_within_horizon, use gamma^horizon_steps
                    gamma_effective = gamma ** (self.horizon_steps * n_steps.float())
                else:
                    # Default single-step return
                    gamma_effective = gamma
                    
                target_q = reward + gamma_effective * (1 - done.float()) * target_next_q  # (B, 1) - reward already includes intrinsic or n-step rewards
            else:
                # Multi-sample for more stable Q-targets
                K = self.num_next_noise_samples
                next_noise_samples = torch.randn(K, batch_size, action.shape[1], self.action_dim, device=self.device)
                next_state_rep = next_state.unsqueeze(0).expand(K, -1, -1, -1).reshape(K*batch_size, *next_state.shape[1:])
                next_noise_flat = next_noise_samples.reshape(K*batch_size, *next_noise_samples.shape[2:])
                next_actions = self.get_action(next_state_rep, next_noise_flat)  # (K*B, horizon_steps, action_dim)
                if self.clip_action:
                    next_actions = torch.clamp(next_actions, -1.0, 1.0)
                target_q_samples = self.target_critic(next_state_rep, next_noise_flat, next_actions)  # (K*B, 1) - already min across ensemble
                # target_q_samples = self._clip_q_values(target_q_samples)  # Clip before averaging to prevent outliers
                
                # Compute target next Q based on optimistic_target setting
                if self.optimistic_target:
                    # Use percentile instead of mean for more optimistic targets
                    target_q_reshaped = target_q_samples.reshape(K, batch_size, 1)  # (K, B, 1)
                    # Use torch.quantile for accurate percentile computation
                    # Note: quantile operates along dim=0 (K samples per batch element)
                    target_next_q = torch.quantile(
                        target_q_reshaped.squeeze(-1),  # (K, B)
                        q=self.optimistic_percentile,
                        dim=0,
                        keepdim=False
                    ).unsqueeze(-1)  # (B, 1)
                else:
                    # Standard mean aggregation
                    target_next_q = target_q_samples.reshape(K, batch_size, 1).mean(dim=0)  # (B, 1)
                
                # Apply appropriate discount factor based on training configuration
                if self.use_n_step:
                    # For n-step returns, use gamma^n_step
                    gamma_effective = gamma ** n_steps.float()  # (B, 1)
                elif self.discount_within_horizon:
                    # For discount_within_horizon, use gamma^horizon_steps
                    gamma_effective = gamma ** (self.horizon_steps * n_steps.float())
                else:
                    # Default single-step return
                    gamma_effective = gamma
                    
                target_q = reward + gamma_effective * (1 - done.float()) * target_next_q  # (B, 1) - reward already includes intrinsic or n-step rewards
       
        # Compute all losses (pass pre-computed current_actions to avoid redundant call)
        actor_losses = self.actor_loss(
            state, noise, current_actions, q_values, confidence, pretrained_actions, next_state, next_noise, training_step, q_overestimation, data_source=data_source
        )
        critic_losses = self.critic_loss(state, noise, action, target_q)  # Use dataset actions for critic
        
        # Compute dynamics loss only if dynamics model is enabled
        if self.use_dynamics:
            dynamics_losses = self.dynamics_loss(state, action, next_state)
            dynamics_loss_value = self.dynamics_weight * dynamics_losses['dynamics_loss']
        else:
            dynamics_losses = {'dynamics_loss': torch.tensor(0.0, device=self.device)}
            dynamics_loss_value = torch.tensor(0.0, device=self.device)
        
        # Combine all losses
        total_loss = (
            actor_losses['actor_total'] +
            self.critic_weight * critic_losses['critic_loss'] +
            dynamics_loss_value
        )
        
        # Return all losses
        return {
            'total_loss': total_loss,
            **actor_losses,
            **critic_losses,
            **dynamics_losses,
            'confidence_weight_mean': confidence.mean(),  # This is actually confidence weights [1,0] not raw confidence
            'confidence_weight_std': confidence.std(),
        }
    

    # def pretrain_on_expert_data(
    #     self,
    #     dataloader,
    #     num_epochs: int = 100,
    #     actor_lr: float = 1e-4,
    #     critic_lr: float = 1e-4,
    #     dynamics_lr: float = 1e-4,
    #     gamma: float = 0.99,
    # ) -> Dict[str, List[float]]:
    #     """
    #     Pretrain all components (actor, critic, dynamics) on expert data.
        
    #     Training procedure:
    #     - Actor: Learn to match pretrained policy output π_pre(a|s,z)
    #     - Critic: Learn Q-values using pretrained policy for next actions
    #     - Dynamics: Learn to predict next states
        
    #     Args:
    #         dataloader: DataLoader providing (state, action, next_state, reward) tuples
    #         num_epochs: int - number of training epochs
    #         actor_lr: float - learning rate for actor
    #         critic_lr: float - learning rate for critic
    #         dynamics_lr: float - learning rate for dynamics
    #         gamma: float - discount factor
            
    #     Returns:
    #         history: Dictionary containing training history
    #     """
    #     # Initialize optimizers
    #     actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
    #     critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
    #     if self.use_dynamics:
    #         dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=dynamics_lr)
    #     else:
    #         dynamics_optimizer = None
        
    #     history = {
    #         'actor_pretrain_loss': [],
    #         'critic_pretrain_loss': [],
    #         'dynamics_pretrain_loss': [],
    #     }
        
    #     # Set models to training mode
    #     self.actor.train()
    #     self.critic.train()
    #     if self.use_dynamics:
    #         self.dynamics.train()
        
    #     for epoch in range(num_epochs):
    #         epoch_actor_loss = 0.0
    #         epoch_critic_loss = 0.0
    #         epoch_dynamics_loss = 0.0
    #         num_batches = 0
            
    #         for batch in dataloader:
    #             # Extract data from Transition format: (actions, conditions, rewards, dones)
    #             # After DataLoader batching, the shapes are:
    #             actions = batch.actions.to(self.device)  # (B, horizon_steps, action_dim)
    #             conditions = batch.conditions  # dict with 'state' and 'next_state'
    #             rewards = batch.rewards.to(self.device)  # (B, 1)
    #             dones = batch.dones.to(self.device)  # (B, 1)
                
    #             # Extract current state and next state from conditions
    #             state = conditions['state'].to(self.device)  # (B, cond_steps, obs_dim)
    #             next_state = conditions['next_state'].to(self.device)  # (B, cond_steps, obs_dim)
                
    #             # Use full observation history and action sequences for learning
                
    #             batch_size = state.shape[0]
    #             horizon_steps = actions.shape[1]  # Get horizon_steps from actions shape
                
    #             # Sample noise for current and next states - should match action shape
    #             noise = torch.randn(batch_size, horizon_steps, self.action_dim, device=self.device)
    #             next_noise = torch.randn(batch_size, horizon_steps, self.action_dim, device=self.device)
                
    #             # ===== Actor Pretraining =====
    #             # During pretraining, actor should match the EXPERT actions from dataset, not pretrained policy
    #             predicted_actions = self.actor(state, noise)  # (B, horizon_steps, action_dim)

    #             # compute target actions from pretrained policy
    #             with torch.no_grad():
    #                 target_actions, _ = self.get_action_and_confidence_simple(state, noise)  # (B, horizon_steps, action_dim), (B, 1)
    #                 target_actions = target_actions.detach()
    #             actor_loss = F.mse_loss(predicted_actions, target_actions)  # Match expert actions from dataset

    #             # ===== Critic Pretraining =====
    #             # During pretraining, we don't have next actions from expert data, so compute simpler targets
    #             # Option 1: Bootstrap using current critic predictions (but this may be unstable early on)
    #             # Option 2: Use a simpler target like rewards only for terminal states
                
    #             # Use the current actor's predictions for next state (since we're training them together)
    #             with torch.no_grad():
    #                 next_actor_actions = self.actor(next_state, next_noise)  # (B, horizon_steps, action_dim)
    #                 if self.clip_action:
    #                     next_actor_actions = torch.clamp(next_actor_actions, -1.0, 1.0)
    #                 # Use ensemble min for target (like in main training)
    #                 next_q_values = self.critic(next_state, next_noise, next_actor_actions)  # (B, 1) - already min across ensemble
    #                 # next_q_values = self._clip_q_values(next_q_values)
    #                 target_q = rewards + gamma * (1 - dones.float()) * next_q_values  # (B, 1)
                
    #             # Critic learns to predict Q-values for expert state-action pairs (all ensemble networks)
    #             q_preds = self.critic(state, noise, actions, return_all=True)  # List of (B, 1)
    #             critic_loss = 0
    #             for q_pred in q_preds:
    #                 # q_pred = self._clip_q_values(q_pred)
    #                 critic_loss += F.mse_loss(q_pred, target_q)  # Sum losses from all networks
                
    #             # ===== Dynamics Pretraining =====
    #             # Dynamics learns to predict next states (only if enabled)
    #             if self.use_dynamics:
    #                 predicted_next_state = self.dynamics(state, actions)  # (B, cond_steps, obs_dim)
    #                 dynamics_loss = F.mse_loss(predicted_next_state, next_state)  # scalar
    #             else:
    #                 dynamics_loss = torch.tensor(0.0, device=self.device)
                
    #             # ===== Backward Pass =====
    #             # Update actor
    #             actor_optimizer.zero_grad()
    #             actor_loss.backward()
    #             actor_optimizer.step()
                
    #             # Update critic
    #             critic_optimizer.zero_grad()
    #             critic_loss.backward()
    #             critic_optimizer.step()
                
    #             # Update dynamics (only if enabled)
    #             if self.use_dynamics:
    #                 dynamics_optimizer.zero_grad()
    #                 dynamics_loss.backward()
    #                 dynamics_optimizer.step()
                
    #             # Accumulate losses
    #             epoch_actor_loss += actor_loss.item()
    #             epoch_critic_loss += critic_loss.item()
    #             if self.use_dynamics:
    #                 epoch_dynamics_loss += dynamics_loss.item()
    #             else:
    #                 epoch_dynamics_loss += 0.0
    #             num_batches += 1
            
    #         # Record average losses
    #         avg_actor_loss = epoch_actor_loss / num_batches
    #         avg_critic_loss = epoch_critic_loss / num_batches
    #         avg_dynamics_loss = epoch_dynamics_loss / num_batches
            
    #         history['actor_pretrain_loss'].append(avg_actor_loss)
    #         history['critic_pretrain_loss'].append(avg_critic_loss)
    #         history['dynamics_pretrain_loss'].append(avg_dynamics_loss)
            
    #         if epoch % 10 == 0:
    #             log.info(f"Pretrain epoch {epoch}: "
    #                     f"actor_loss={avg_actor_loss:.6f}, "
    #                     f"critic_loss={avg_critic_loss:.6f}, "
    #                     f"dynamics_loss={avg_dynamics_loss:.6f}")
        
    #     return history
