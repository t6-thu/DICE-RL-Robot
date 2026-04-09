"""
Configuration for RL finetuning on real robot.

This config needs to be the same across learner and env_runner nodes.
"""
import os

if "DICE_HARDWARE_CONFIG_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable DICE_HARDWARE_CONFIG_FOLDERS")
if "DICE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable DICE_DATASET_FOLDERS")
if "DICE_CHECKPOINT_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable DICE_CHECKPOINT_FOLDERS")

hardware_config_folder_path = os.environ.get("DICE_HARDWARE_CONFIG_FOLDERS")
data_folder_path = os.environ.get("DICE_DATASET_FOLDERS")
checkpoint_folder_path = os.environ.get("DICE_CHECKPOINT_FOLDERS")

run_learner_on_server = False

control_para = {
    "raw_time_step_s": 0.002,  # dt of raw data collection. Used to compute time step from time_s such that the downsampling according to shape_meta works.
    "slow_down_factor": 1,  
    "sparse_execution_horizon": 20, 
    "delay_tolerance_s": 0.3, # delay larger than this will trigger termination
    "max_duration_s": 1620,
    "test_nominal_target": False,
    "translational_stiffness": [1500, 1500, 1500],  # gear
    "rotational_stiffness": 100,  # gear  
    "send_transitions_to_server": True,
    "fix_orientation": False,
    "pausing_mode": False,
    "no_visual_mode": False,
    "device": "cuda",
    "resume_rl": False,  # If True, resume from latest RL checkpoint; if False, start fresh
    "scale_and_cap_residual_action": False,
    "residual_action_scale_ratio": 0.3,  # ratio of residual to nominal action
    # Reset pose (pose7: x, y, z, qx, qy, qz, qw)
    "reset_pose": [0.0636259, 0.4380778, 0.5641403, 0.0056671, 0.9999500, -0.0011580, 0.0081614],  # gear insertion
}

# Hardware parameters
hardware_para = {
    "hardware_config_path": hardware_config_folder_path + "/belt_assembly.yaml",
}

# Model parameters (passed to DistillResidualRLImgModel)
model_para = {
    "pretrained_flow_policy_path": None,  # TODO: SET path to pretrained diffusion policy checkpoint
    "obs_dim": None,  # Required: augmented obs dim (visual_feature_dim + state_dim)
    "action_dim": None,  # Required: action dimension
    "noise_dim": None,  # Required: noise dimension (typically same as action_dim)
    "cond_steps": 2,  # Should match pretrained policy
    "horizon_steps": 20,  # Should match pretrained policy
    # Network activation (used by DistillRLModel for actor/critic construction)
    "activation_type": "GELU",
    # Actor network
    "actor_hidden_dims": [1024, 1024, 1024],
    # Critic network
    "critic_hidden_dims": [1024, 1024, 1024],
    "use_layernorm": True,
    # BC regularization
    "bc_loss_weight": 140,
    # Residual RL settings
    "condition_residual_on_base_action": False,
    "clip_residual_action": False,
    "clip_action": False,
    # Critic settings
    "use_q_normalization": True,
    "disable_q_loss_for_expert_data": True,
    "multi_sample_next_noise": True,
    "num_next_noise_samples": 4,
    "critic_ensemble_size": 5,
    "conservative_q_method": "min",
    # Multi-z actor loss
    "sample_multi_z_for_actor_loss": True,
    "num_multi_z_for_actor_loss": 8, # can be smaller to speed up training
    "use_dynamics": False,
    # DDIM inference steps for pretrained policy during RL training (None = use checkpoint default)
    "rl_num_inference_steps": 8, # can be smaller to speed up finetuning, default is 16
    "device": "cuda",
}

# Learner parameters (training settings)
learner_para = {
    # RL finetuning checkpoint (to resume training)
    "rl_ckpt_path": None, # TODO: SET path to RL checkpoint to resume from, or None to start fresh

    # Training settings
    "num_episodes_before_first_training": 20,  # Wait for this many episodes before training
    "gradient_steps": 2000,  # Gradient updates per training round
    "update_every_x_episode": 10,  # Number of online episodes to collect before each training round
    "batch_size": 256,

    # RLPD settings
    "use_rlpd": True,
    "expert_ratio": 0.5,  # Fraction of batch from expert data (initial if adaptive)
    "expert_dataset_path": None,  # TODO: SET path to expert dataset zarr, for RLPD
    # Adaptive expert ratio
    "use_adaptive_expert_ratio": True,
    "adaptive_expert_ratio_start": 0.7,
    "adaptive_expert_ratio_end": 0.2,
    "adaptive_expert_ratio_steps": 30000,  # Training steps over which to anneal

    # Discount and TD settings
    "gamma": 0.99,
    "tau": 0.01,  # Target network update rate

    # Learning rates
    "actor_lr": 1e-4,
    "critic_lr": 1e-4,
    "max_grad_norm": 1.0,

    "device": "cuda",
}

# Network communication (same across learner and actor)
online_learning_para = {
    "data_folder_path": data_folder_path + "/rl_finetuning_rollouts/",
    "network_weight_topic": "rl_network_weights_topic",
    "transitions_topic": "rl_transitions_topic",
    "network_weight_expire_time_s": 3500,
    "transitions_topic_expire_time_s": 3500,
}

if run_learner_on_server:
    online_learning_para["network_server_endpoint"] = "tcp://localhost:18889"
    online_learning_para["transitions_server_endpoint"] = "tcp://localhost:18888"
else:
    online_learning_para["network_server_endpoint"] = "ipc:///tmp/feeds/rl_weights"
    online_learning_para["transitions_server_endpoint"] = "ipc:///tmp/feeds/rl_transitions"

# HybridReplayBuffer settings
replay_buffer_para = {
    "max_size": 1000000,
    "robot_dt_ms": 2.0,  # Robot control frequency in ms
}
