"""
Residual RL model that learns residual actions on top of pretrained flow matching policy.

Action = π_pre(s,z) + r_θ(s,z)
Loss = -Q(s,a) + β||r_θ(s,z)||²

This approach avoids unlearning the pretrained policy while allowing selective deviation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .distill_rl import DistillRLModel
import logging
log = logging.getLogger(__name__)


class DistillResidualRLModel(DistillRLModel):
    """
    Residual RL model that learns residual actions on top of pretrained flow matching policy.
    
    Inherits from DistillRLModel but overrides the get_action method to compute
    action as: a = π_pre(s,z) + r_θ(s,z) where r_θ is the residual actor network.
    """
    
    def __init__(
        self,
        condition_residual_on_base_action: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.condition_residual_on_base_action = condition_residual_on_base_action
        
        print(f'DistillResidualRLModel initialized with:')
        print(f'  condition_residual_on_base_action: {self.condition_residual_on_base_action}')
        print(f'  bc_loss_weight: {self.bc_loss_weight} (used for residual regularization)')
        print(f'  clip_residual_action: {self.clip_residual_action}')
        if self.condition_residual_on_base_action:
            print(f'  Residual actor input: r_θ(state, base_action)')
        else:
            print(f'  Residual actor input: r_θ(state, noise)')
        print(f'  All other parameters inherited from DistillRLModel')
        
    def get_action(self, state: torch.Tensor, noise: torch.Tensor, return_pretrained_actions: bool = False) -> torch.Tensor:
        """
        Get action as sum of pretrained policy and residual actor.
        
        Action = π_pre(s,z) + r_θ(s,z) or π_pre(s,z) + r_θ(s,a_base)
        
        Args:
            state: (B, cond_steps, obs_dim) - current state
            noise: (B, horizon_steps, action_dim) - noise for action generation
            return_pretrained_actions: If True, return tuple (total_actions, pretrained_actions)
            
        Returns:
            action: (B, horizon_steps, action_dim) - total action (pretrained + residual)
            OR if return_pretrained_actions:
            (action, pretrained_actions): tuple of actions
        """
        # Get pretrained action (no gradient)
        with torch.no_grad():
            cond = {"state": state}
            pretrained_sample = self.pretrained_flow_policy(cond, deterministic=False, init_noise=noise)
            pretrained_actions = pretrained_sample.trajectories  # (B, horizon_steps, action_dim)
        
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
        return total_actions
    
