"""
Distilled RL Model for image-based online finetuning.

This module extends DistillRLModel to handle image observations with merged visual features.
The key difference is that the state already contains visual features merged by the environment wrapper.

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

log = logging.getLogger(__name__)

from dice_rl.model.distill_rl import DistillRLModel


class DistillRLImgModel(DistillRLModel):
    """
    Image-based distilled RL model for online finetuning.
    
    This model works with augmented state dimensions where visual features
    are already merged with low-dim state by the environment wrapper.
    
    The main differences from DistillRLModel:
    1. get_action_and_confidence_simple uses forward_from_features
    2. pretrain_on_expert_data handles image data and extracts features
    3. _load_pretrained_policy loads VisionFlowMatchingMLP
    """
    
    def __init__(
        self,
        pretrained_flow_policy_path: str,
        obs_dim: int,  # This should be augmented (visual_feature_dim + original_obs_dim)
        action_dim: int,
        cond_steps: int = 1,
        horizon_steps: int = 8,
        # Actor settings
        actor_hidden_dims: List[int] = [256, 256, 256],
        actor_activation: str = "Gelu",
        # Critic settings
        critic_hidden_dims: List[int] = [256, 256, 256],
        critic_activation: str = "Gelu",
        num_critics: int = 2,
        q_depends_on_noise: bool = True,
        # Dynamics model settings
        use_dynamics: bool = False,
        dynamics_hidden_dims: List[int] = [256, 256],
        dynamics_activation: str = "Gelu",
        # Training settings
        bc_loss_weight: float = 1.0,
        use_soft_q_filtering: bool = False,
        use_intrinsic_reward: bool = False,
        intrinsic_reward_scale: float = 1.0,
        multi_z_actor_loss_warmup_steps: int = None,  # New parameter for multi-z actor loss warmup
        **kwargs
    ):
        """
        Initialize image-based distilled RL model.
        
        Args:
            obs_dim: Augmented observation dimension (visual_feature_dim + original_obs_dim)
            Other args same as DistillRLModel
        """
        # Store augmented obs_dim
        self.augmented_obs_dim = obs_dim
        
        # Store multi_z_actor_loss_warmup_steps before calling parent init
        # If not provided, will use q_filtering_warmup_steps from parent (backward compatibility)
        self._multi_z_actor_loss_warmup_steps_override = multi_z_actor_loss_warmup_steps
        
        # Call parent init with augmented obs_dim
        super().__init__(
            pretrained_flow_policy_path=pretrained_flow_policy_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            cond_steps=cond_steps,
            horizon_steps=horizon_steps,
            actor_hidden_dims=actor_hidden_dims,
            actor_activation=actor_activation,
            critic_hidden_dims=critic_hidden_dims,
            critic_activation=critic_activation,
            num_critics=num_critics,
            q_depends_on_noise=q_depends_on_noise,
            use_dynamics=use_dynamics,
            dynamics_hidden_dims=dynamics_hidden_dims,
            dynamics_activation=dynamics_activation,
            bc_loss_weight=bc_loss_weight,
            use_soft_q_filtering=use_soft_q_filtering,
            use_intrinsic_reward=use_intrinsic_reward,
            intrinsic_reward_scale=intrinsic_reward_scale,
            **kwargs
        )
        
        # Override multi_z_actor_loss_warmup_steps if provided
        if self._multi_z_actor_loss_warmup_steps_override is not None:
            self.multi_z_actor_loss_warmup_steps = self._multi_z_actor_loss_warmup_steps_override
            log.info(f"Using custom multi_z_actor_loss_warmup_steps={self.multi_z_actor_loss_warmup_steps}")
        else:
            # Default to q_filtering_warmup_steps for backward compatibility
            self.multi_z_actor_loss_warmup_steps = self.q_filtering_warmup_steps
            log.info(f"Using q_filtering_warmup_steps={self.q_filtering_warmup_steps} for multi_z_actor_loss_warmup")
        
        log.info(f"DistillRLImgModel initialized with augmented obs_dim={obs_dim}")
    
    def _load_pretrained_policy(self, checkpoint_path: str, device: str):
        """
        Load pretrained diffusion policy.

        Override parent's method to verify it's an image-based model.

        Args:
            checkpoint_path: Path to the pretrained policy checkpoint
            device: Device to load the model on

        Returns:
            Loaded and frozen DiffusionUnetTimmMod1Policy
        """
        # Use parent's loading logic
        pretrained_model = super()._load_pretrained_policy(checkpoint_path, device)

        # Verify it has required methods for image-based model
        if not hasattr(pretrained_model, 'extract_visual_features'):
            log.warning("Pretrained model doesn't have extract_visual_features method")
        if not hasattr(pretrained_model, 'predict_action_from_features'):
            log.warning("Pretrained model doesn't have predict_action_from_features method")

        log.info("Pretrained image-based diffusion policy loaded and frozen successfully")
        return pretrained_model
    
    def get_action_and_confidence_simple(self, state: torch.Tensor, noise: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action for image-based model.

        Uses predict_action_from_features with pre-encoded visual features.

        Args:
            state: (B, obs_feature_dim) - pre-encoded visual features from obs_encoder
            noise: (B, horizon_steps, action_dim) - initial noise for action generation

        Returns:
            action: (B, horizon_steps, action_dim) - predicted action
            confidence_weight: (B, 1) - always 1.0 (confidence deprecated)
        """
        B = state.shape[0]

        # Ensure both state and noise are on the same device
        state = state.to(self.device)
        noise = noise.to(self.device)

        # Flatten state if needed: (B, cond_steps, obs_dim) -> (B, cond_steps * obs_dim)
        if state.dim() == 3:
            state = state.view(B, -1)

        # Use predict_action_from_features for diffusion policy (normalized)
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
    
    # TODO: parts could be abstracted away 
    # This func is deprecated
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
    #     Pretrain actor, critic, and dynamics on expert dataset with image data.
        
    #     This method handles raw image data and extracts visual features using
    #     the pretrained flow policy's visual encoder. The image preprocessing
    #     follows the same pattern as vision_mlp_flow_matching.py and 
    #     train_flow_matching_img_agent.py.
        
    #     Args:
    #         dataloader: DataLoader providing (state, action, next_state, reward) tuples with image data
    #         num_epochs: Number of training epochs
    #         actor_lr: Learning rate for actor
    #         critic_lr: Learning rate for critic  
    #         dynamics_lr: Learning rate for dynamics
    #         gamma: Discount factor
            
    #     Returns:
    #         Dictionary of training history
    #     """
    #     log.info(f"Pretraining on expert dataset with image data")
        
    #     # Create optimizers
    #     actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
    #     critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
    #     if self.use_dynamics:
    #         dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=dynamics_lr)
        
    #     # Training history
    #     history = {
    #         'actor_pretrain_loss': [],
    #         'critic_pretrain_loss': [],
    #         'dynamics_pretrain_loss': [],
    #     }
        
    #     for epoch in range(num_epochs):
    #         epoch_actor_loss = 0.0
    #         epoch_critic_loss = 0.0
    #         epoch_dynamics_loss = 0.0
    #         num_batches = 0
            
    #         for batch in dataloader:
    #             # Extract data from batch - this contains raw images
    #             conditions = batch.conditions
                
    #             # Get raw state and RGB data
    #             raw_state = conditions['state'].to(self.device)  # (B, cond_steps, original_obs_dim)
    #             rgb = conditions['rgb'].to(self.device) if 'rgb' in conditions else None  # (B, cond_steps, H, W, C)
    #             next_raw_state = conditions['next_state'].to(self.device)
    #             next_rgb = conditions['next_rgb'].to(self.device) if 'next_rgb' in conditions else None
                
    #             actions = batch.actions.to(self.device)  # (B, horizon_steps, action_dim)
    #             rewards = batch.rewards.to(self.device).unsqueeze(-1)  # (B, 1)
    #             dones = batch.dones.to(self.device).unsqueeze(-1)  # (B, 1)
                
    #             B = raw_state.shape[0]
                
    #             # Extract visual features and merge with state
    #             with torch.no_grad():
    #                 if rgb is not None and hasattr(self.pretrained_flow_policy.network, 'extract_visual_features'):
    #                     # The dataset already provides images in the correct format: (B, T, C, H, W)
    #                     # where C = num_cameras * 3 (e.g., 12 for 4 cameras)
    #                     # No permutation needed!
                        
    #                     # Extract visual features for current state using exact same pattern as vision_mlp_flow_matching.py
    #                     cond = {'state': raw_state, 'rgb': rgb}
    #                     visual_features = self.pretrained_flow_policy.network.extract_visual_features(cond)  # (B, feature_dim)
                        
    #                     # Extract visual features for next state
    #                     next_cond = {'state': next_raw_state, 'rgb': next_rgb}
    #                     next_visual_features = self.pretrained_flow_policy.network.extract_visual_features(next_cond)
                        
    #                     # Merge visual features with raw state to create augmented state
    #                     # Repeat visual features for each cond_step
    #                     visual_features_expanded = visual_features.unsqueeze(1).repeat(1, self.cond_steps, 1)  # (B, cond_steps, feature_dim)
    #                     next_visual_features_expanded = next_visual_features.unsqueeze(1).repeat(1, self.cond_steps, 1)
                        
    #                     # Concatenate to create augmented state
    #                     state = torch.cat([visual_features_expanded, raw_state], dim=-1)  # (B, cond_steps, augmented_obs_dim)
    #                     next_state = torch.cat([next_visual_features_expanded, next_raw_state], dim=-1)
    #                 else:
    #                     # Fallback: no visual features (shouldn't happen for image models)
    #                     log.warning("No visual feature extraction available, using raw state only")
    #                     state = raw_state
    #                     next_state = next_raw_state
                
    #             # Sample noise for actor
    #             noise = torch.randn(B, self.horizon_steps, self.action_dim, device=self.device)
                
    #             predicted_actions = self.actor(state, noise)  # (B, horizon_steps, action_dim)
    #             actor_loss = F.mse_loss(predicted_actions, actions)  # Match expert actions from dataset

    #             # ===== Critic Pretraining =====
    #             # Target Q-value for TD learning
    #             with torch.no_grad():
    #                 # Sample next noise and get next actions
    #                 next_noise = torch.randn(B, self.horizon_steps, self.action_dim, device=self.device)
    #                 next_actor_actions = self.actor(next_state, next_noise)
                    
    #                 # Get next Q-values
    #                 if self.q_depends_on_noise:
    #                     next_q_values = self.critic(next_state, next_noise, next_actor_actions)
    #                 else:
    #                     next_q_values = self.critic(next_state, noise=None, action=next_actor_actions)
                    
    #                 next_q_values = self._clip_q_values(next_q_values)
    #                 target_q = rewards + gamma * (1 - dones.float()) * next_q_values
                
    #             # Critic learns to predict Q-values
    #             if self.q_depends_on_noise:
    #                 q_preds = self.critic(state, noise, actions, return_all=True)
    #             else:
    #                 q_preds = self.critic(state, noise=None, action=actions, return_all=True)
                
    #             critic_loss = 0
    #             for q_pred in q_preds:
    #                 q_pred = self._clip_q_values(q_pred)
    #                 critic_loss += F.mse_loss(q_pred, target_q)
                
    #             # ===== Dynamics Pretraining =====
    #             if self.use_dynamics:
    #                 predicted_next_state = self.dynamics(state, actions)
    #                 dynamics_loss = F.mse_loss(predicted_next_state, next_state)
    #             else:
    #                 dynamics_loss = torch.tensor(0.0, device=self.device)
                
    #             # ===== Backward Pass =====
    #             # Update actor
    #             actor_optimizer.zero_grad()
    #             actor_loss.backward()
    #             actor_optimizer.step()
                
    #             # # Update critic
    #             # critic_optimizer.zero_grad()
    #             # critic_loss.backward()
    #             # critic_optimizer.step()
                
    #             # Update dynamics
    #             if self.use_dynamics:
    #                 dynamics_optimizer.zero_grad()
    #                 dynamics_loss.backward()
    #                 dynamics_optimizer.step()
                
    #             # Accumulate losses
    #             epoch_actor_loss += actor_loss.item()
    #             epoch_critic_loss += critic_loss.item()
    #             epoch_dynamics_loss += dynamics_loss.item() if self.use_dynamics else 0.0
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


class DistillResidualRLImgModel(DistillRLImgModel):
    """
    Residual RL model for image-based observations.
    
    Inherits from DistillRLImgModel but overrides the get_action method to compute
    action as: a = π_pre(s,z) + r_θ(s,z) where r_θ is the residual actor network.
    
    The visual feature extraction is handled by the training agent, so this model
    receives already augmented states.
    """
    
    def __init__(self, 
                 condition_residual_on_base_action: bool = False,
                 sample_multi_z_for_actor_loss: bool = False,
                 num_multi_z_for_actor_loss: int = 8,
                 topk_divisor_for_self_imitation: int = 4,
                 use_softmax_weighted: bool = False,
                 self_imitation_for_actor_loss: bool = False,
                 self_imitation_loss_weight: float = 40,
                 self_imitation_loss_distributional: bool = False,
                 winner_loser: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.condition_residual_on_base_action = condition_residual_on_base_action
        self.sample_multi_z_for_actor_loss = sample_multi_z_for_actor_loss
        self.num_multi_z_for_actor_loss = num_multi_z_for_actor_loss
        self.topk_divisor_for_self_imitation = topk_divisor_for_self_imitation
        self.use_softmax_weighted = use_softmax_weighted
        self.self_imitation_for_actor_loss = self_imitation_for_actor_loss
        self.self_imitation_loss_weight = self_imitation_loss_weight
        self.self_imitation_loss_distributional = self_imitation_loss_distributional
        self.winner_loser = winner_loser
        
        print(f'DistillResidualRLImgModel initialized with:')
        print(f'  condition_residual_on_base_action: {self.condition_residual_on_base_action}')
        print(f'  sample_multi_z_for_actor_loss: {self.sample_multi_z_for_actor_loss}')
        if self.sample_multi_z_for_actor_loss:
            print(f'    num_multi_z_for_actor_loss: {self.num_multi_z_for_actor_loss}')
            print(f'    topk_divisor_for_self_imitation: {self.topk_divisor_for_self_imitation} (top {100//self.topk_divisor_for_self_imitation}%)')
            print(f'    use_softmax_weighted: {self.use_softmax_weighted}')
            print(f'    self_imitation_for_actor_loss: {self.self_imitation_for_actor_loss}')
            print(f'    self_imitation_loss_distributional: {self.self_imitation_loss_distributional}')
            print(f'    self_imitation_loss_weight: {self.self_imitation_loss_weight}')
            print(f'    winner_loser: {self.winner_loser}')

        print(f'  bc_loss_weight: {self.bc_loss_weight} (used for residual regularization)')
        print(f'  clip_residual_action: {self.clip_residual_action}')
        if self.condition_residual_on_base_action:
            print(f'  Residual actor input: r_θ(state, base_action)')
        else:
            print(f'  Residual actor input: r_θ(state, noise)')
        print(f'  All other parameters inherited from DistillRLImgModel')
        
    def get_action(self, state: torch.Tensor, noise: torch.Tensor, return_pretrained_actions: bool = False):
        """
        Get action as sum of pretrained policy and residual actor.
        
        Action = π_pre(s,z) + r_θ(s,z) or π_pre(s,z) + r_θ(s,a_base)
        
        Args:
            state: (B, cond_steps, augmented_obs_dim) - state with visual features already merged
            noise: (B, horizon_steps, action_dim) - noise for action generation
            return_pretrained_actions: if True, return tuple (total_actions, pretrained_actions)
            
        Returns:
            if return_pretrained_actions:
                tuple: (total_actions, pretrained_actions) both (B, horizon_steps, action_dim)
            else:
                action: (B, horizon_steps, action_dim) - total action (pretrained + residual)
        """
        # Get pretrained action in normalized space (no gradient)
        state = state.to(self.device)
        noise = noise.to(self.device)
        with torch.no_grad():
            B = state.shape[0]
            flat_state = state.view(B, -1) if state.dim() == 3 else state
            result = self.pretrained_policy.predict_action_from_features(
                sparse_nobs_encode=flat_state,
                init_noise=noise,
                unnormalize=False,
            )
            pretrained_actions = result["sparse"].detach()  # normalized
        
        # Get residual action from actor - condition on base action or noise
        if self.condition_residual_on_base_action:
            # Use base action as input: r_θ(s, a_base)
            residual_actions = self.actor(state, pretrained_actions)  # (B, horizon_steps, action_dim)
        else:
            # Use noise as input: r_θ(s, z) 
            residual_actions = self.actor(state, noise)  # (B, horizon_steps, action_dim)
        
        # Clip residual actions if enabled
        if self.clip_residual_action:
            residual_actions = torch.clamp(residual_actions, -0.3, 0.3)
        
        # Total action = pretrained + residual
        total_actions = pretrained_actions + residual_actions
        
        # Apply clipping to total action if needed
        if self.clip_action:
            total_actions = torch.clamp(total_actions, -1.0, 1.0)
        
        if return_pretrained_actions:
            return total_actions, pretrained_actions
        else:
            return total_actions
    
    def get_exploration_action(self, state: torch.Tensor, num_samples: int = 10,
                              exploration_strategy: str = "max_q_std", training_step: int = 0,
                              replay_flow_model=None, replay_flow_config=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get exploration action using specified strategy.
        
        Image-based version that works with augmented states.
        Supports two strategies:
        - max_q_std: Select action with highest Q-std across ensemble (epistemic uncertainty)
        - max_q_min: Select action with highest minimum Q-value across ensemble (optimistic)
        
        If training_step < replay_flow_warmup_steps, uses single sample regardless of strategy.
        
        Args:
            state: (B, cond_steps, augmented_obs_dim) - state with visual features already merged
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

            elif exploration_strategy == "max_q_std":  # max_q_std (default)
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
            
            elif exploration_strategy == "max_replay_diversity":
                # Max replay diversity exploration - select action with minimum replay flow likelihood
                # Assert that replay flow model is provided
                assert replay_flow_model is not None, "replay_flow_model required for max_replay_diversity"
                assert not replay_flow_model.training, "replay_flow_model must be in eval mode"
                
                # Get config parameters
                q_filter_k = replay_flow_config['q_filter_k']
                exploration_flow_steps = replay_flow_config['exploration_flow_steps']
                exploration_n_mc = replay_flow_config['exploration_n_mc']
                use_surrogate_loglik = replay_flow_config['use_surrogate_loglik']

                # 1. Filter by top-k Q-values
                q_min = q_reshaped.min(dim=0)[0]  # (num_samples, B, 1)
                top_k = q_filter_k
                top_q_values, top_indices = q_min.squeeze(-1).topk(min(top_k, num_samples), dim=0)  # (k, B)
                
                # 2. Get corresponding actions for top-k samples
                # Reshape actions back first
                actions_reshaped = actions_flat.view(num_samples, B, self.horizon_steps, self.action_dim)
                top_actions = torch.stack([
                    actions_reshaped[top_indices[k], torch.arange(B)]  # (B, horizon_steps, action_dim)
                    for k in range(min(top_k, num_samples))
                ], dim=0)  # (k, B, horizon_steps, action_dim)
                
                # 3. Compute replay flow likelihoods or surrogate scores (batched)
                # Reshape for batch processing: (k*B, horizon_steps, action_dim)
                k_actual = min(top_k, num_samples)
                top_actions_flat = top_actions.view(k_actual * B, self.horizon_steps, self.action_dim)
                
                # Expand state for all top-k samples: (k*B, cond_steps, obs_dim)
                state_for_topk = state.unsqueeze(0).expand(k_actual, -1, -1, -1).reshape(k_actual * B, *state.shape[1:])
                cond = {"state": state_for_topk}
                
                if use_surrogate_loglik:
                    # Use surrogate score matching instead of exact log_prob
                    # Note: surrogate_loglik_rf returns negative scores (higher = better)
                    scores = replay_flow_model.surrogate_loglik_rf(
                        a=top_actions_flat,  # Actions to score
                        cond=cond,  # Dict with 'state' key
                        k_t=4,  # Fixed for now
                        k_x0=2  # Fixed for now
                    )  # (k*B,)
                    # Reshape scores: (k, B) 
                    likelihoods_tensor = scores.view(k_actual, B)
                else:
                    # Use exact log probability
                    log_probs = replay_flow_model.log_prob(
                        a=top_actions_flat,  # Actions to score
                        cond=cond,  # Dict with 'state' key
                        n_mc=exploration_n_mc,
                        steps=exploration_flow_steps
                    )  # (k*B,)
                    # Reshape likelihoods: (k, B)
                    likelihoods_tensor = log_probs.view(k_actual, B)
                
                # 4. Select action with minimum likelihood/score (maximum diversity)
                # For both log_prob and surrogate_loglik_rf, lower values mean less likely/diverse
                min_likelihood_indices = likelihoods_tensor.argmin(dim=0)  # (B,)
                selection_indices = torch.gather(top_indices, 0, min_likelihood_indices.unsqueeze(0)).squeeze(0)
            
            elif exploration_strategy == "graph_laplacian":
                # Graph Laplacian exploration - select action with high Q-value relative to geometric neighbors
                # This strategy emphasizes local improvement in action space
                
                # 1. Get min Q-values across ensemble for all samples (conservative): (num_samples, B)
                q_min = q_reshaped.min(dim=0)[0].squeeze(-1)  # (num_samples, B, 1) -> (num_samples, B)
                
                # 2. Reshape actions for distance computation: (num_samples, B, horizon_steps, action_dim)
                actions_reshaped = actions_flat.view(num_samples, B, self.horizon_steps, self.action_dim)
                
                # 3. Normalize actions per dimension to handle different scales
                # Compute mean and std across samples and batch for each action dimension
                # Shape: (num_samples*B, horizon_steps, action_dim)
                actions_for_stats = actions_reshaped.view(-1, self.horizon_steps, self.action_dim)
                
                # Compute per-dimension statistics across all samples
                # Using dim=0 to get stats across all samples*batches
                actions_mean = actions_for_stats.mean(dim=0, keepdim=True)  # (1, horizon_steps, action_dim)
                actions_std = actions_for_stats.std(dim=0, keepdim=True)  # (1, horizon_steps, action_dim)
                
                # Add small epsilon to avoid division by zero for constant dimensions
                actions_std = torch.clamp(actions_std, min=1e-6)
                
                # Normalize: (num_samples, B, horizon_steps, action_dim)
                actions_normalized = (actions_reshaped - actions_mean.unsqueeze(0)) / actions_std.unsqueeze(0)
                
                # Flatten normalized actions for distance computation: (num_samples, B, D) where D = horizon*action_dim
                actions_flattened = actions_normalized.view(num_samples, B, -1)
                
                # 4. Compute pairwise distances in normalized action space
                # Transpose to (B, num_samples, D) for batched cdist
                actions_batched = actions_flattened.transpose(0, 1)  # (B, num_samples, D)
                
                # Compute pairwise distances using normalized actions: (B, num_samples, num_samples)
                distances = torch.cdist(actions_batched, actions_batched, p=2)
                
                # 5. Find k=3 nearest neighbors (excluding itself)
                k_neighbors = min(4, num_samples - 1)  # In case we have very few samples
                
                # Set diagonal to infinity to exclude self
                inf_mask = torch.eye(num_samples, device=device, dtype=torch.bool)
                distances.masked_fill_(inf_mask.unsqueeze(0), float('inf'))
                
                # Find k nearest neighbors for each sample: (B, num_samples, k)
                _, neighbor_indices = distances.topk(k_neighbors, dim=2, largest=False)
                
                # 6. Compute Graph Laplacian: q_i - mean(q_neighbors)
                # Gather neighbor Q-values: (B, num_samples, k)
                q_min_transposed = q_min.transpose(0, 1)  # (B, num_samples)
                q_neighbors = torch.gather(
                    q_min_transposed.unsqueeze(2).expand(-1, -1, k_neighbors),  # (B, num_samples, k)
                    1,
                    neighbor_indices  # (B, num_samples, k)
                )
                
                # Compute mean of neighbors: (B, num_samples)
                q_neighbors_mean = q_neighbors.mean(dim=2)
                
                # Compute Laplacian score: (B, num_samples)
                laplacian_scores = q_min_transposed - q_neighbors_mean
                
                # 7. Select action with maximum Laplacian score (highest Q relative to neighbors)
                selection_indices = laplacian_scores.argmax(dim=1)  # (B,)
            
            elif exploration_strategy == "mppi":
                mppi_samples = 64
                cem_iterations = 5
                temperature = 0.1  # λ in MPPI
                elite_fraction = 0.2
                elite_size = max(1, int(mppi_samples * elite_fraction))

                # Initialize distribution
                noise_mean = torch.zeros(B, self.horizon_steps, self.action_dim, device=device)
                noise_std  = torch.ones(B, self.horizon_steps, self.action_dim, device=device)
                min_std    = 0.05
                damping    = 0.7

                best_actions = None
                best_noise   = None

                for it in range(cem_iterations):
                    prev_mean = noise_mean
                    prev_std  = noise_std
                    # Sample K noise sequences from current Gaussian
                    eps = torch.randn(
                        mppi_samples, B, self.horizon_steps, self.action_dim, device=device
                    )
                    noise_samples = noise_mean.unsqueeze(0) + noise_std.unsqueeze(0) * eps  # (K,B,H,A)

                    # Flatten for network calls
                    state_expanded = state.unsqueeze(0).expand(mppi_samples, -1, -1, -1)
                    state_flat = state_expanded.reshape(mppi_samples * B, *state.shape[1:])
                    noise_flat = noise_samples.reshape(mppi_samples * B, self.horizon_steps, self.action_dim)
                    with torch.no_grad():
                        actions_flat = self.get_action(state_flat, noise_flat)
                        q_all = self.critic(state_flat, noise_flat, actions_flat, return_all=True)

                        q_stacked = torch.stack(q_all, dim=0)                      # (E, K*B, 1)
                        q_min = q_stacked.min(dim=0)[0].view(mppi_samples, B)      # (K,B)
                        q_values = q_min.transpose(0, 1)                           # (B,K)

                        # Center Q for numerical stability
                        q_centered = q_values - q_values.max(dim=1, keepdim=True)[0]
                        # MPPI weights
                        weights = torch.softmax(q_centered / temperature, dim=1)   # (B,K)

                        # Track best sample by Q (optional)
                        best_idx = q_values.argmax(dim=1)                          # (B,)
                        actions_reshaped = actions_flat.view(
                            mppi_samples, B, self.horizon_steps, self.action_dim
                        )
                        best_actions = torch.stack([actions_reshaped[best_idx[b], b] for b in range(B)])
                        best_noise   = torch.stack([noise_samples[best_idx[b], b]   for b in range(B)])
                        if it < cem_iterations - 1:
                            # Either use elites (CEM style) or full weighted MPPI update.
                            # Here: CEM over top-K, then smooth with previous mean/std.

                            _, top_idx = q_values.topk(elite_size, dim=1)  # (B, elite_size)

                            elite_noise = torch.stack([
                                noise_samples[top_idx[b], b] for b in range(B)
                            ])  # (B, elite_size, H, A)

                            new_mean = elite_noise.mean(dim=1)                # (B,H,A)
                            new_std  = elite_noise.std(dim=1, unbiased=False) # (B,H,A)
                            new_std = torch.clamp(new_std, min=min_std)

                            # Smooth updates
                            noise_mean = damping * new_mean + (1 - damping) * prev_mean
                            noise_std  = damping * new_std  + (1 - damping) * prev_std
            
            # Reshape actions back (if not MPPI)
            if exploration_strategy != "mppi":
                actions_reshaped = actions_flat.view(num_samples, B, self.horizon_steps, self.action_dim)
            
            # Select actions based on strategy
            if exploration_strategy == "mppi":
                # MPPI already computed selected actions and noise
                selected_actions = best_actions  # Already selected in the MPPI loop
                selected_noise = best_noise      # Already selected in the MPPI loop
            else:
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
    
    def compute_self_imitation_loss(
        self, 
        state: torch.Tensor,
        training_step: int = 0,
        q_overestimation: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute standalone self-imitation loss for periodic updates.
        
        This function computes self-imitation loss separately from the main actor loss,
        allowing for periodic updates with gradient clipping to prevent interference
        with RL gradients.
        
        Args:
            state: (B, cond_steps, obs_dim) - augmented state with visual features
            training_step: Current training step
            q_overestimation: (B, 1) - Q-value overestimation for filtering
            
        Returns:
            Dictionary containing:
            - self_imitation_loss: The loss to backpropagate
            - self_imitation_active_ratio: Ratio of samples where SI is applied
            - num_elite_samples: Number of elite samples selected
        """
        B = state.shape[0]
        K = self.num_multi_z_for_actor_loss
        
        # Check if we're past warmup phase
        if training_step <= self.q_filtering_warmup_steps:
            # During warmup, no self-imitation
            return {
                'self_imitation_loss': torch.tensor(0.0, device=self.device),
                'self_imitation_active_ratio': torch.tensor(0.0, device=self.device),
                'num_elite_samples': torch.tensor(0.0, device=self.device),
            }
        
        # Sample K noise vectors for each state
        noise_samples = torch.randn(B, K, self.horizon_steps, self.action_dim, device=self.device)  # (B, K, H, A)
        
        # Compute actions for all K noise samples
        state_expanded = state.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, cond_steps, obs_dim)
        state_flat = state_expanded.reshape(B * K, *state.shape[1:])  # (B*K, cond_steps, obs_dim)
        noise_flat = noise_samples.reshape(B * K, self.horizon_steps, self.action_dim)  # (B*K, H, A)
        
        # Get actions with gradients (needed for actor update)
        actions_flat, pretrained_actions_flat = self.get_action(state_flat, noise_flat, return_pretrained_actions=True)
        actions_samples = actions_flat.reshape(B, K, *actions_flat.shape[1:])  # (B, K, H, A)
        pretrained_actions_samples = pretrained_actions_flat.reshape(B, K, *pretrained_actions_flat.shape[1:])
        
        # Compute Q-values and filtering conditions (no gradients needed)
        with torch.no_grad():
            # Compute Q-values for all samples
            q_current_flat = self.critic(state_flat, noise_flat, actions_flat)  # (B*K, 1)
            q_current_samples = q_current_flat.reshape(B, K)  # (B, K)
            
            q_pretrained_flat = self.critic(state_flat, noise_flat, pretrained_actions_flat)  # (B*K, 1)
            q_pretrained_samples = q_pretrained_flat.reshape(B, K)  # (B, K)
            
            # Compute average Q-values for filtering
            avg_q_current = q_current_samples.mean(dim=1, keepdim=True)  # (B, 1)
            avg_q_pretrained = q_pretrained_samples.mean(dim=1, keepdim=True)  # (B, 1)
            q_advantage = avg_q_current - avg_q_pretrained  # (B, 1)
            better_than_expert = (q_advantage > 0).float()  # (B, 1) - 1.0 if better, 0.0 if worse
            
            # Determine which samples should have self-imitation applied
            if q_overestimation is not None:
                q_underestimated = (q_overestimation < self.q_underestimation_threshold).float()  # (B, 1)
                should_apply_si = better_than_expert * q_underestimated  # (B, 1)
            else:
                should_apply_si = better_than_expert  # (B, 1)
            
            # If no samples should apply SI, return zero loss
            if should_apply_si.sum() == 0:
                return {
                    'self_imitation_loss': torch.tensor(0.0, device=self.device),
                    'self_imitation_active_ratio': torch.tensor(0.0, device=self.device),
                    'num_elite_samples': torch.tensor(0.0, device=self.device),
                }
            
            # Select top actions based on Q-values (elite selection)
            k_top = max(1, K // self.topk_divisor_for_self_imitation)  # Top percentage of K samples
            
            # Find top-k indices based on Q-values for each batch element
            _, teacher_indices = q_current_samples.topk(k_top, dim=1)  # (B, k_top)
            
            # Gather the top-k actions as teacher actions (detached, no gradients)
            gather_idx = teacher_indices.view(B, k_top, 1, 1).expand(-1, -1, self.horizon_steps, self.action_dim)
            teacher_actions = actions_samples.gather(1, gather_idx).detach()  # (B, k_top, H, A)
            
            # Create mask for top-k indices
            is_topk = torch.zeros(B, K, dtype=torch.bool, device=self.device)
            is_topk.scatter_(1, teacher_indices, True)
        
        # Compute self-imitation loss using nearest neighbor matching
        # Expand for pairwise distance computation
        actions_expanded = actions_samples.unsqueeze(2)  # (B, K, 1, H, A)
        teachers_expanded = teacher_actions.unsqueeze(1)  # (B, 1, k_top, H, A)
        
        # Compute squared L2 distances
        pairwise_distances = ((actions_expanded - teachers_expanded) ** 2).sum(dim=(3, 4))  # (B, K, k_top)
        
        # Find nearest teacher for each action
        min_distances, nearest_teacher_indices = pairwise_distances.min(dim=2)  # (B, K)
        
        # Gather nearest teacher actions
        batch_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, K)  # (B, K)
        batch_idx_flat = batch_idx.reshape(-1)  # (B*K,)
        teacher_idx_flat = nearest_teacher_indices.reshape(-1)  # (B*K,)
        nearest_teachers_all = teacher_actions[batch_idx_flat, teacher_idx_flat].reshape(
            B, K, self.horizon_steps, self.action_dim
        )  # (B, K, H, A)
        
        # Compute imitation loss (MSE between current and nearest teacher)
        imitation_diff = actions_samples - nearest_teachers_all  # (B, K, H, A)
        imitation_loss_per_sample = (imitation_diff ** 2).mean(dim=(2, 3))  # (B, K)
        
        # Zero out loss for top-k samples (they don't need to imitate themselves)
        imitation_loss_per_sample = imitation_loss_per_sample * (~is_topk).float()  # (B, K)
        
        # Apply self-imitation filter
        should_apply_si_expanded = should_apply_si.expand(-1, K)  # (B, K)
        filtered_imitation_loss_per_sample = should_apply_si_expanded * imitation_loss_per_sample  # (B, K)
        
        # Compute final loss (mean across all samples)
        self_imitation_loss = filtered_imitation_loss_per_sample.mean()  # scalar
        
        # Compute metrics
        active_ratio = should_apply_si.mean()
        # num_elite is k_top per batch element (elite samples are selected for all batches, not just SI-active ones)
        num_elite_total = k_top * B  # Total elite samples across batch
        
        return {
            'self_imitation_loss': self_imitation_loss,
            'self_imitation_active_ratio': active_ratio,
            'num_elite_samples': torch.tensor(k_top, device=self.device),  # Elite samples per batch element
            'num_elite_total': torch.tensor(num_elite_total, device=self.device),  # Total elite samples
        }
    
    # NOTE: temporarily commented out
    # def actor_loss(
    #     self,
    #     state: torch.Tensor,
    #     noise: torch.Tensor,
    #     current_actions: torch.Tensor,  # (B, H, A) - total actions (pretrained + residual)
    #     q_values: torch.Tensor,         # (B, 1)
    #     confidence: torch.Tensor,       # (B, 1)
    #     pretrained_actions: torch.Tensor,   # (B, H, A)
    #     next_state: Optional[torch.Tensor] = None,
    #     next_noise: Optional[torch.Tensor] = None,
    #     training_step: int = 0,
    #     q_overestimation: Optional[torch.Tensor] = None,  # (B,1) if provided
    #     data_source: Optional[torch.Tensor] = None,  # (B,1) - 0 for online, 1 for expert
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     Actor loss for residual RL: L_θ = -Q_φ(s,z,a) + β||r_θ(s,z)||²
        
    #     Note: current_actions = π_pre(s,z) + r_θ(s,z) from get_action()
    #           We need to extract r_θ(s,z) for regularization.
    #     """
    #     if not self.sample_multi_z_for_actor_loss or training_step <= self.multi_z_actor_loss_warmup_steps:
    #         # Q-value loss (negative because we want to maximize Q)
    #         if self.disable_q_loss_for_expert_data and data_source is not None:
    #             # Mask Q loss for expert data: only apply Q loss to online data (data_source == 0)
    #             online_mask = (data_source == 0).float()  # (B, 1) - 1.0 for online, 0.0 for expert
    #             q_loss = -(q_values * online_mask).mean()
    #         else:
    #             # Standard Q loss for all samples
    #             q_loss = -q_values.mean()
            
    #         # Normalize Q-loss by mean absolute Q-value for stability (like FQL)
    #         if self.use_q_normalization and q_values.abs().mean() > 1e-8:  # Avoid division by zero
    #             q_loss_scale = 1.0 / q_values.abs().mean().detach()
    #             q_loss = q_loss_scale * q_loss
            
    #         # Compute metrics for logging and Q-filtering
    #         with torch.no_grad():
    #             pretrained_q_values = self.critic(state, noise, pretrained_actions)  # (B, 1)
    #             pretrained_q_values = self._clip_q_values(pretrained_q_values)
    #             q_advantage = q_values - pretrained_q_values  # (B, 1)
    #             better_than_expert = (q_advantage > 0).float()  # (B, 1) - 1.0 if better, 0.0 if worse
    #             better_percentage = better_than_expert.mean()  # Fraction of samples where policy is better
            
    #         # Apply soft Q-filtering: zero out regularization where current policy is better than expert AND Q is underestimated
    #         # Only apply after warm-up period to let critic learn meaningful Q-values first
    #         use_q_filtering = self.use_soft_q_filtering and (training_step > self.q_filtering_warmup_steps)
            
    #         if use_q_filtering:
    #             # Filter when better than expert AND Q is underestimated
    #             if q_overestimation is not None:
    #                 # q_overestimation < 0 means underestimation
    #                 q_underestimated = (q_overestimation < self.q_underestimation_threshold).float()  # (B, 1) - 1.0 if underestimated, 0.0 if overestimated
    #                 # Apply filter if BOTH conditions are true: better_than_expert AND q_underestimated
    #                 should_filter = better_than_expert * q_underestimated  # (B, 1) - 1.0 only when both are true
    #                 bc_filter = 1.0 - should_filter  # (B, 1)
    #             else:
    #                 # Fallback to original behavior if q_overestimation not provided
    #                 bc_filter = 1.0 - better_than_expert  # (B, 1)
    #         else:
    #             bc_filter = torch.ones_like(better_than_expert, device=self.device)  # (B, 1)
            
    #         # Override bc_filter for expert data if always_retain_bc_loss_for_expert_data is True
    #         if self.always_retain_bc_loss_for_expert_data and data_source is not None:
    #             # For expert data (data_source == 1), always set bc_filter to 1.0
    #             expert_mask = (data_source == 1).float()  # (B, 1) - 1.0 for expert, 0.0 for online
    #             # Override bc_filter: 1.0 for expert data, keep original for online data
    #             bc_filter = expert_mask + (1.0 - expert_mask) * bc_filter  # (B, 1)
            
    #         # Compute BC-style loss with filtering: ||current_action - pretrained_action||²
    #         # This is equivalent to ||r_θ(s,z)||² since current_action = pretrained_action + r_θ(s,z)
    #         action_diff = current_actions - pretrained_actions  # (B, horizon_steps, action_dim)
    #         # Compute MSE across action dimensions for each timestep, then average across time
    #         mse_per_timestep = (action_diff ** 2).mean(dim=-1, keepdim=True)  # (B, horizon_steps, 1)
    #         # Average across timesteps to get per-batch MSE: (B, 1)
    #         mse_per_batch = mse_per_timestep.mean(dim=1)  # (B, 1)
    #         # Apply filtering (no confidence weighting needed for residual RL)
    #         filtered_bc_loss = (bc_filter * mse_per_batch).mean()  # scalar
            
    #         # Total loss
    #         total_loss = q_loss + self.bc_loss_weight * filtered_bc_loss
            
    #         # Compute additional metrics for logging
    #         with torch.no_grad():
    #             residual_norm = (action_diff ** 2).mean().sqrt()  # RMS of residual actions (same as action_diff)

    #     elif self.sample_multi_z_for_actor_loss and self.use_softmax_weighted:
    #         # Multi-z sampling for actor loss
    #         B = state.shape[0]
    #         K = self.num_multi_z_for_actor_loss
            
    #         # Sample K noise vectors for each state
    #         noise_samples = torch.randn(B, K, *noise.shape[1:], device=self.device)  # (B, K, H, A)
            
    #         # Compute actions for all K noise samples
    #         # We need to reshape for batch processing
    #         state_expanded = state.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, cond_steps, obs_dim)
    #         state_flat = state_expanded.reshape(B * K, *state.shape[1:])  # (B*K, cond_steps, obs_dim)
    #         noise_flat = noise_samples.reshape(B * K, *noise.shape[1:])  # (B*K, H, A)
            
    #         # Get actions with pretrained actions returned
    #         actions_flat, pretrained_actions_flat = self.get_action(state_flat, noise_flat, return_pretrained_actions=True)  # (B*K, H, A)
            
    #         # Reshape back to (B, K, H, A)
    #         actions_samples = actions_flat.reshape(B, K, *actions_flat.shape[1:])  # (B, K, H, A)
    #         pretrained_actions_samples = pretrained_actions_flat.reshape(B, K, *pretrained_actions_flat.shape[1:])  # (B, K, H, A)
            
    #         # Compute Q-values for current actions (with gradients for actor loss)
    #         q_current_flat = self.critic(state_flat, noise_flat, actions_flat)  # (B*K, 1)
    #         q_current_samples = q_current_flat.reshape(B, K)  # (B, K)
            
    #         # Compute Q-values for pretrained actions and importance weights (no gradients)
    #         with torch.no_grad():
    #             q_pretrained_flat = self.critic(state_flat, noise_flat, pretrained_actions_flat)  # (B*K, 1)
    #             q_pretrained_samples = q_pretrained_flat.reshape(B, K)  # (B, K)
                
    #             # Compute TWO sets of importance weights
    #             temperature = 1.0  # Can be made a hyperparameter if needed
                
    #             # Weights from current Q-values: w_current_k = exp(Q(s, a_k) / τ) / Σ_j exp(Q(s, a_j) / τ)
    #             q_current_logits = q_current_samples.detach() / temperature  # (B, K) - detach for no grad
    #             q_current_weights = torch.softmax(q_current_logits, dim=1)  # (B, K)
                
    #             # Weights from pretrained Q-values: w_pretrained_k = exp(Q(s, π_pre(s,z_k)) / τ) / Σ_j exp(Q(s, π_pre(s,z_j)) / τ)
    #             q_pretrained_logits = q_pretrained_samples / temperature  # (B, K)
    #             q_pretrained_weights = torch.softmax(q_pretrained_logits, dim=1)  # (B, K)
            
    #         # Q-value loss: -Σ_k w_current_k * Q(s, a_k)
    #         # Use current weights for the actor loss
    #         weighted_q_loss_per_batch = (q_current_weights * q_current_samples).sum(dim=1)  # (B,)
            
    #         # Apply disable_q_loss_for_expert_data if enabled (for consistency with if branch)
    #         if self.disable_q_loss_for_expert_data and data_source is not None:
    #             # Mask Q loss for expert data: only apply Q loss to online data (data_source == 0)
    #             online_mask = (data_source == 0).float().squeeze(-1)  # (B, 1) -> (B,) - 1.0 for online, 0.0 for expert
    #             weighted_q_loss_per_batch = weighted_q_loss_per_batch * online_mask  # (B,)
            
    #         q_loss = -weighted_q_loss_per_batch.mean()  # scalar
            
    #         # Normalize Q-loss by mean absolute Q-value for stability (like FQL)
    #         if self.use_q_normalization and q_current_samples.abs().mean() > 1e-8:  # Avoid division by zero
    #             q_loss_scale = 1.0 / q_current_samples.abs().mean().detach()
    #             q_loss = q_loss_scale * q_loss
            
    #         # Compute metrics for logging and Q-filtering
    #         with torch.no_grad():
    #             # Weighted Q-values using respective weights
    #             weighted_q_current = (q_current_weights * q_current_samples).sum(dim=1, keepdim=True)  # (B, 1)
    #             weighted_q_pretrained = (q_pretrained_weights * q_pretrained_samples).sum(dim=1, keepdim=True)  # (B, 1)
    #             q_advantage = weighted_q_current - weighted_q_pretrained  # (B, 1)
    #             better_than_expert = (q_advantage > 0).float()  # (B, 1) - 1.0 if better, 0.0 if worse
    #             better_percentage = better_than_expert.mean()  # Fraction of samples where policy is better
            
    #         # Apply soft Q-filtering: zero out regularization where current policy is better than expert AND Q is underestimated
    #         # Only apply after warm-up period to let critic learn meaningful Q-values first
    #         use_q_filtering = self.use_soft_q_filtering and (training_step > self.q_filtering_warmup_steps)
            
    #         if use_q_filtering:
    #             # Filter when better than expert AND Q is underestimated
    #             if q_overestimation is not None:
    #                 # q_overestimation < 0 means underestimation
    #                 q_underestimated = (q_overestimation < self.q_underestimation_threshold).float()  # (B, 1)
    #                 # Apply filter if BOTH conditions are true: better_than_expert AND q_underestimated
    #                 should_filter = better_than_expert * q_underestimated  # (B, 1) - 1.0 only when both are true
    #                 bc_filter = 1.0 - should_filter  # (B, 1)
    #             else:
    #                 # Fallback to original behavior if q_overestimation not provided
    #                 bc_filter = 1.0 - better_than_expert  # (B, 1)
    #         else:
    #             bc_filter = torch.ones_like(better_than_expert, device=self.device)  # (B, 1)
            
    #         # Override bc_filter for expert data if always_retain_bc_loss_for_expert_data is True
    #         if self.always_retain_bc_loss_for_expert_data and data_source is not None:
    #             # For expert data (data_source == 1), always set bc_filter to 1.0
    #             expert_mask = (data_source == 1).float()  # (B, 1) - 1.0 for expert, 0.0 for online
    #             # Override bc_filter: 1.0 for expert data, keep original for online data
    #             bc_filter = expert_mask + (1.0 - expert_mask) * bc_filter  # (B, 1)
            
    #         # Expand bc_filter to match K samples
    #         bc_filter_expanded = bc_filter.expand(-1, K)  # (B, K)
            
    #         # Compute BC-style loss with filtering: ||current_action - pretrained_action||²
    #         # This is equivalent to ||r_θ(s,z)||² since current_action = pretrained_action + r_θ(s,z)
    #         action_diff = actions_samples - pretrained_actions_samples  # (B, K, horizon_steps, action_dim)
    #         # Compute MSE across action dimensions for each timestep, then average across time
    #         mse_per_timestep = (action_diff ** 2).mean(dim=-1)  # (B, K, horizon_steps)
    #         # Average across timesteps to get per-sample MSE: (B, K)
    #         mse_per_sample = mse_per_timestep.mean(dim=-1)  # (B, K)
    #         # Apply filtering and weighting (use pretrained weights for BC loss)
    #         weighted_filtered_mse = q_pretrained_weights * bc_filter_expanded * mse_per_sample  # (B, K)
    #         # Sum across K samples and average across batch
    #         filtered_bc_loss = weighted_filtered_mse.sum(dim=1).mean()  # scalar
            
    #         # Add self-imitation loss if enabled
    #         if self.self_imitation_for_actor_loss:
    #             # Choose imitation targets based on the variant
    #             with torch.no_grad():
    #                 if self.self_imitation_loss_distributional:
    #                     # Distributional self-imitation: select top 25% elite and match each action to closest elite
    #                     # Select top 25% actions based on Q-values
    #                     k_top = max(1, K // 8)  # Top 25% of K samples
                        
    #                     # Find top-k indices based on Q-values for each batch element
    #                     _, teacher_indices = q_current_samples.topk(k_top, dim=1)  # (B, k_top)
                        
    #                     # Gather the top-k actions as teacher actions
    #                     gather_idx = teacher_indices.view(B, k_top, 1, 1).expand(-1, -1, self.horizon_steps, self.action_dim)
    #                     teacher_actions = actions_samples.gather(1, gather_idx).detach()  # (B, k_top, H, A)
                        
    #                     # Gather Q-values for the teacher actions
    #                     teacher_q_values = q_current_samples.gather(1, teacher_indices)  # (B, k_top)
                        
    #                     # Prepare student actions (pretrained) and targets for transport cost computation
    #                     student_priors = pretrained_actions_samples.unsqueeze(2)  # (B, K, 1, H, A)
    #                     targets = teacher_actions.unsqueeze(1)  # (B, 1, k_top, H, A)
                        
    #                     # Compute L2 transport cost between each student and teacher action
    #                     transport_cost = ((student_priors - targets) ** 2).sum(dim=(-2, -1))  # (B, K, k_top)
                        
    #                     # Find nearest teacher for each student action
    #                     min_cost, nearest_teacher_idx = transport_cost.min(dim=2)  # (B, K)
                        
    #                     # Get Q-values of the nearest teacher for each student
    #                     nearest_teacher_q = teacher_q_values.gather(1, nearest_teacher_idx)  # (B, K)
                        
    #                     # Create filter: use teacher action if its Q-value > pretrained Q-value, else use pretrained
    #                     use_teacher_action = nearest_teacher_q > q_pretrained_samples  # (B, K)
                        
    #                     # Get the teacher actions that each student would imitate
    #                     nearest_teacher_idx_expanded = nearest_teacher_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.horizon_steps, self.action_dim)  # (B, K, H, A)
    #                     nearest_teacher_actions = teacher_actions.gather(1, nearest_teacher_idx_expanded)  # (B, K, H, A)
                        
    #                     # Imitation targets: use nearest teacher if Q(teacher) > Q(pretrained), else use pretrained
    #                     use_teacher_action_expanded = use_teacher_action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.horizon_steps, self.action_dim)  # (B, K, H, A)
    #                     imitation_targets = use_teacher_action_expanded * nearest_teacher_actions + (~use_teacher_action_expanded) * pretrained_actions_samples  # (B, K, H, A)
    #                 else:
    #                     # Original self-imitation: all actions mimic single best action
    #                     # Find best current action: a_best = argmax_k Q(s, a^current_bk)
    #                     best_current_idx = q_current_samples.argmax(dim=1)  # (B,)
    #                     best_current_q = q_current_samples.gather(1, best_current_idx.unsqueeze(1))  # (B, 1)
                        
    #                     # Create (B,K) filter: use_best_action[b,k] = True if Q(s_b, a_best) > Q(s_b, a^pre_bk)
    #                     use_best_action = best_current_q > q_pretrained_samples  # (B, K) - broadcast (B,1) > (B,K)
                        
    #                     # Get best current actions for each sample (B,K,H,A) where each k gets the same best action
    #                     best_current_actions = actions_samples.gather(1, best_current_idx.view(B, 1, 1, 1).expand(-1, K, self.horizon_steps, self.action_dim))  # (B, K, H, A)
                        
    #                     # Imitation targets: best current action if Q(s, a_best) > Q(s, a^pre_bk), else a^pre_bk
    #                     use_best_action_expanded = use_best_action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.horizon_steps, self.action_dim)  # (B, K, H, A)
    #                     imitation_targets = use_best_action_expanded * best_current_actions + (~use_best_action_expanded) * pretrained_actions_samples  # (B, K, H, A)
                
    #             # Common self-imitation loss computation for both variants
    #             imitation_diff = actions_samples - imitation_targets  # (B, K, horizon_steps, action_dim)
    #             # Compute MSE across action dimensions for each timestep, then average across time
    #             imitation_mse_per_timestep = (imitation_diff ** 2).mean(dim=-1)  # (B, K, horizon_steps)
    #             # Average across timesteps to get per-sample MSE: (B, K)
    #             imitation_mse_per_sample = imitation_mse_per_timestep.mean(dim=-1)  # (B, K)
    #             # Apply uniform weighting (no bc_filter since targets are already selected based on Q-values)
    #             uniform_weights = torch.ones(B, K, device=self.device) / K  # (B, K) - equal weights
    #             weighted_imitation_mse = uniform_weights * imitation_mse_per_sample  # (B, K)
    #             # Sum across K samples and average across batch
    #             filtered_self_imitation_loss = weighted_imitation_mse.sum(dim=1).mean()  # scalar
                
    #             # Total loss with self-imitation
    #             # total_loss = q_loss + self.bc_loss_weight * (filtered_bc_loss + 0.5 * filtered_self_imitation_loss)
    #             total_loss = q_loss + self.bc_loss_weight * filtered_bc_loss + self.self_imitation_loss_weight * filtered_self_imitation_loss
    #         else:
    #             # Total loss without self-imitation
    #             total_loss = q_loss + self.bc_loss_weight * filtered_bc_loss
            
    #         # Compute additional metrics for logging
    #         with torch.no_grad():
    #             residual_norm = (action_diff ** 2).mean().sqrt()  # RMS of residual actions
    #             # Create compatible variables for return statement
    #             pretrained_q_values = weighted_q_pretrained  # (B, 1)
    #             q_values = weighted_q_current  # (B, 1)
    #             # bc_filter is already defined as (B, 1)

    #     elif self.sample_multi_z_for_actor_loss and self.self_imitation_for_actor_loss:
    #         # Self-imitation variant: copy structure from softmax weighted branch, don't use softmax weights but use direct average
    #         B = state.shape[0]
    #         K = self.num_multi_z_for_actor_loss
            
    #         # Sample K noise vectors for each state
    #         noise_samples = torch.randn(B, K, *noise.shape[1:], device=self.device)  # (B, K, H, A)
            
    #         # Compute actions for all K noise samples
    #         # We need to reshape for batch processing
    #         state_expanded = state.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, cond_steps, obs_dim)
    #         state_flat = state_expanded.reshape(B * K, *state.shape[1:])  # (B*K, cond_steps, obs_dim)
    #         noise_flat = noise_samples.reshape(B * K, *noise.shape[1:])  # (B*K, H, A)
            
    #         # Get actions with pretrained actions returned
    #         actions_flat, pretrained_actions_flat = self.get_action(state_flat, noise_flat, return_pretrained_actions=True)  # (B*K, H, A)
            
    #         # Reshape back to (B, K, H, A)
    #         actions_samples = actions_flat.reshape(B, K, *actions_flat.shape[1:])  # (B, K, H, A)
    #         pretrained_actions_samples = pretrained_actions_flat.reshape(B, K, *pretrained_actions_flat.shape[1:])  # (B, K, H, A)
            
    #         # Compute Q-values for current actions (with gradients for actor loss)
    #         q_current_flat = self.critic(state_flat, noise_flat, actions_flat)  # (B*K, 1)
    #         q_current_samples = q_current_flat.reshape(B, K)  # (B, K)
            
    #         # Compute Q-values for pretrained actions and imitation targets (no gradients)
    #         with torch.no_grad():
    #             q_pretrained_flat = self.critic(state_flat, noise_flat, pretrained_actions_flat)  # (B*K, 1)
    #             q_pretrained_samples = q_pretrained_flat.reshape(B, K)  # (B, K)
                
    #             # Find best current action: a_best = argmax_k Q(s, a^current_bk)
    #             best_current_idx = q_current_samples.argmax(dim=1)  # (B,)
    #             best_current_q = q_current_samples.gather(1, best_current_idx.unsqueeze(1))  # (B, 1)
                
    #             # Create (B,K) filter: use_best_action[b,k] = True if Q(s_b, a_best) > Q(s_b, a^pre_bk)
    #             use_best_action = best_current_q > q_pretrained_samples  # (B, K) - broadcast (B,1) > (B,K)
                
    #             # Get best current actions for each sample (B,K,H,A) where each k gets the same best action
    #             best_current_actions = actions_samples.gather(1, best_current_idx.view(B, 1, 1, 1).expand(-1, K, self.horizon_steps, self.action_dim))  # (B, K, H, A)
                
    #             # Imitation targets: best current action if Q(s, a_best) > Q(s, a^pre_bk), else a^pre_bk
    #             use_best_action_expanded = use_best_action.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.horizon_steps, self.action_dim)  # (B, K, H, A)
    #             imitation_targets = use_best_action_expanded * best_current_actions + (~use_best_action_expanded) * pretrained_actions_samples  # (B, K, H, A)
            
    #         # Q-value loss: -mean(Q(s, a_k)) (different from sample_multi_z_for_actor_loss, i wouldn't be using softmax weights now. i will directly take the average)
    #         q_loss_per_batch = q_current_samples.mean(dim=1)  # (B,) - average over K samples
            
    #         # Apply disable_q_loss_for_expert_data if enabled (for consistency with other branches)
    #         if self.disable_q_loss_for_expert_data and data_source is not None:
    #             # Mask Q loss for expert data: only apply Q loss to online data (data_source == 0)
    #             online_mask = (data_source == 0).float().squeeze(-1)  # (B, 1) -> (B,) - 1.0 for online, 0.0 for expert
    #             q_loss_per_batch = q_loss_per_batch * online_mask  # (B,)
            
    #         q_loss = -q_loss_per_batch.mean()  # scalar
            
    #         # Normalize Q-loss by mean absolute Q-value for stability (like FQL)
    #         if self.use_q_normalization and q_current_samples.abs().mean() > 1e-8:  # Avoid division by zero
    #             q_loss_scale = 1.0 / q_current_samples.abs().mean().detach()
    #             q_loss = q_loss_scale * q_loss
            
    #         # Compute metrics for logging and Q-filtering
    #         with torch.no_grad():
    #             # Average Q-values for metrics (no softmax weighting, just direct average)
    #             avg_q_current = q_current_samples.mean(dim=1, keepdim=True)  # (B, 1)
    #             avg_q_pretrained = q_pretrained_samples.mean(dim=1, keepdim=True)  # (B, 1)
    #             q_advantage = avg_q_current - avg_q_pretrained  # (B, 1)
    #             better_than_expert = (q_advantage > 0).float()  # (B, 1) - 1.0 if better, 0.0 if worse
    #             better_percentage = better_than_expert.mean()  # Fraction of samples where policy is better
            
    #         # Apply soft Q-filtering: zero out regularization where current policy is better than expert AND Q is underestimated
    #         # Only apply after warm-up period to let critic learn meaningful Q-values first
    #         use_q_filtering = self.use_soft_q_filtering and (training_step > self.q_filtering_warmup_steps)
            
    #         if use_q_filtering:
    #             # Filter when better than expert AND Q is underestimated
    #             if q_overestimation is not None:
    #                 # q_overestimation < 0 means underestimation
    #                 q_underestimated = (q_overestimation < self.q_underestimation_threshold).float()  # (B, 1)
    #                 # Apply filter if BOTH conditions are true: better_than_expert AND q_underestimated
    #                 should_filter = better_than_expert * q_underestimated  # (B, 1) - 1.0 only when both are true
    #                 bc_filter = 1.0 - should_filter  # (B, 1)
    #             else:
    #                 # Fallback to original behavior if q_overestimation not provided
    #                 bc_filter = 1.0 - better_than_expert  # (B, 1)
    #         else:
    #             bc_filter = torch.ones_like(better_than_expert, device=self.device)  # (B, 1)
            
    #         # Override bc_filter for expert data if always_retain_bc_loss_for_expert_data is True
    #         if self.always_retain_bc_loss_for_expert_data and data_source is not None:
    #             # For expert data (data_source == 1), always set bc_filter to 1.0
    #             expert_mask = (data_source == 1).float()  # (B, 1) - 1.0 for expert, 0.0 for online
    #             # Override bc_filter: 1.0 for expert data, keep original for online data
    #             bc_filter = expert_mask + (1.0 - expert_mask) * bc_filter  # (B, 1)
            
    #         # Expand bc_filter to match K samples
    #         bc_filter_expanded = bc_filter.expand(-1, K)  # (B, K)
            
    #         # Compute BC-style loss with filtering: ||current_action - pretrained_action||²
    #         # This is equivalent to ||r_θ(s,z)||² since current_action = pretrained_action + r_θ(s,z)
    #         action_diff = actions_samples - pretrained_actions_samples  # (B, K, horizon_steps, action_dim)
    #         # Compute MSE across action dimensions for each timestep, then average across time
    #         mse_per_timestep = (action_diff ** 2).mean(dim=-1)  # (B, K, horizon_steps)
    #         # Average across timesteps to get per-sample MSE: (B, K)
    #         mse_per_sample = mse_per_timestep.mean(dim=-1)  # (B, K)
    #         # Apply filtering and equal weighting (no softmax weights, just equal average)
    #         uniform_weights = torch.ones(B, K, device=self.device) / K  # (B, K) - equal weights
    #         weighted_filtered_mse = uniform_weights * bc_filter_expanded * mse_per_sample  # (B, K)
    #         # Sum across K samples and average across batch
    #         filtered_bc_loss = weighted_filtered_mse.sum(dim=1).mean()  # scalar
            
    #         # Self-imitation loss: β||a^current_bk - imitation_target_bk||²
    #         imitation_diff = actions_samples - imitation_targets  # (B, K, horizon_steps, action_dim)
    #         # Compute MSE across action dimensions for each timestep, then average across time
    #         imitation_mse_per_timestep = (imitation_diff ** 2).mean(dim=-1)  # (B, K, horizon_steps)
    #         # Average across timesteps to get per-sample MSE: (B, K)
    #         imitation_mse_per_sample = imitation_mse_per_timestep.mean(dim=-1)  # (B, K)
    #         # Apply equal weighting (no bc_filter since imitation targets are already selected based on Q-values)
    #         weighted_imitation_mse = uniform_weights * imitation_mse_per_sample  # (B, K)
    #         # Sum across K samples and average across batch
    #         filtered_self_imitation_loss = weighted_imitation_mse.sum(dim=1).mean()  # scalar
            
    #         # Total loss
    #         # total_loss = q_loss + self.bc_loss_weight * (filtered_bc_loss + filtered_self_imitation_loss)
    #         total_loss = q_loss + self.bc_loss_weight * filtered_bc_loss + self.self_imitation_loss_weight * filtered_self_imitation_loss
    #         # total_loss = q_loss + self.bc_loss_weight // 2 * filtered_bc_loss + 10 * filtered_self_imitation_loss
    #         # total_loss = q_loss + self.bc_loss_weight * filtered_bc_loss + 20 * filtered_self_imitation_loss
            
    #         # Compute additional metrics for logging
    #         with torch.no_grad():
    #             residual_norm = (action_diff ** 2).mean().sqrt()  # RMS of residual actions
    #             # Create compatible variables for return statement
    #             pretrained_q_values = avg_q_pretrained  # (B, 1)
    #             q_values = avg_q_current  # (B, 1)
    #             # bc_filter is already defined as (B, 1)
    #     else:
    #         raise RuntimeError("shouldn't get to here")

    #     return {
    #         'actor_total': total_loss,
    #         'actor_q_loss': q_loss,
    #         'actor_residual_loss': filtered_bc_loss,  # This is the filtered BC loss (equivalent to residual loss)
    #         'actor_bc_loss': filtered_bc_loss,  # BC-style loss for compatibility with logging
    #         'actor_future_confidence_loss': torch.tensor(0.0, device=self.device),  # Not used
    #         'actor_gfc_loss': torch.tensor(0.0, device=self.device),  # Not used
    #         'actor_dtr_loss': torch.tensor(0.0, device=self.device),  # Not used
    #         # Metrics
    #         'q_advantage_mean': q_advantage.mean(),
    #         'better_than_expert_percentage': better_percentage,
    #         'pretrained_q_mean': pretrained_q_values.mean(),
    #         'current_q_mean': q_values.mean(),
    #         'residual_norm': residual_norm,
    #         'q_filtering_active': bc_filter.mean(),  # Mean of filtering mask
    #     }
    