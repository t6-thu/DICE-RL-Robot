"""DICE-RL finetuning config for YAM joint-space diffusion policy.

=============================================================
DICE-RL ALGORITHM — HOW IT WORKS ON YAM
=============================================================

Phase 1: Warmup (20 pure-BC rollouts)
--------------------------------------
  Collect `num_episodes_before_first_training` episodes using only the frozen
  BC diffusion policy (no residual actor yet).  All rollouts are stored in the
  online replay buffer.

Phase 2: RLPD training rounds (every 10 rollouts)
--------------------------------------------------
  After each block of `update_every_x_episode` new rollouts, the learner runs
  `gradient_steps` actor + critic updates:

    for each gradient step:
        batch = 128 expert samples  +  128 online samples   (batch_size=256)
            ← expert = BC training demonstrations (50% initially, annealed to 20%)
            ← online = robot rollouts collected so far

        1. Critic update (RLPD ensemble Q-learning):
               Q_target = r + γ · min_k Q_k(s', a'_actor)
               critic_loss = mean MSE over 5 critics

        2. Actor update (DDPO + BC regularisation):
               q_loss = -min_k Q_k(s, actor(s))
               bc_loss = MSE(actor(s), frozen_BC_action(s))
               actor_loss = q_loss + bc_loss_weight * bc_loss

        3. Target critic update (Polyak):
               θ_target = (1 - τ) θ_target + τ θ

  Then the learner pushes the updated residual actor weights to the env runner.

Phase 3: Inference with residual correction
--------------------------------------------
  For each inference step:
      features  = frozen_BC_encoder(obs_history)
      bc_action = BC_diffusion_forward(features)       # in normalized [-1, 1]
      delta     = residual_actor(features, noise)       # learned correction
      final_act = clamp(bc_action + delta, -1, 1)       # normalized
      joint_cmd = denormalize(final_act)                 # → i2rt raw joint targets

Reward
------
  Sparse binary: success = 1.0, failure = 0.0, collected at episode end via
  keyboard input (s / f / d).

=============================================================
KEY HYPERPARAMETERS & THEIR EFFECT
=============================================================

  num_episodes_before_first_training = 20
      How many rollouts to collect before any RL gradient steps.
      More = stabler Q estimates at first training; less = faster RL.

  update_every_x_episode = 10
      One training round per N new rollouts.
      Fewer = more frequent updates (aggressive); more = more data per round.

  gradient_steps = 2000
      Gradient updates per training round.
      2000 steps × batch_size 256 = 512k sample transitions per round.

  batch_size = 256
      Half expert (from BC training npz), half online.
      Expert fraction decays from 0.7 → 0.2 over 30000 steps.

  bc_loss_weight = 100
      How strongly the actor is regularised towards the BC policy.
      High value = conservative exploration; low value = more deviation.

  critic_ensemble_size = 5
      Number of Q-networks. min-Q across ensemble = pessimistic estimate
      (reduces overestimation in offline data).

  gamma = 0.99
      Discount factor. With episode length ~200 steps, effective horizon ≈ 50
      steps for 99% discount.

  tau = 0.01
      Polyak averaging rate for target critics. Slow = stable Q targets.
=============================================================
"""

import os

if "DICE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Run: . ./prepare.sh  first (to set DICE_DATASET_FOLDERS etc.)")

_ckpt_dir  = os.environ.get("DICE_CHECKPOINT_FOLDERS",
                             os.path.expanduser("~/training_outputs"))
_data_dir  = os.environ.get("DICE_DATASET_FOLDERS",
                             os.path.expanduser("~/data/real_processed"))

# ============================================================
# TODO: set before running
# ============================================================
BC_POLICY_CKPT = os.path.expanduser(
    "~/training_outputs/2026.05.19/00.34.02_yam_vit_clip_v1_yam_picknplace_arizonabottle"
    "/checkpoints/epoch=0500-train_loss=0.013.ckpt"
)
EXPERT_NPZ = os.path.join(_data_dir,
                          "yam_picknplace_arizonabottle_224", "train.npz")
NORM_NPZ   = os.path.join(_data_dir,
                          "yam_picknplace_arizonabottle_224", "normalization.npz")
# ---- Run name ----
# Change this string to spin up a fresh experiment without touching previous
# data / checkpoints / logs. Each value of RUN_NAME owns its own:
#   ~/data/real_processed/yam_rl_rollouts_<RUN_NAME>/      ← online episodes
#   ~/training_outputs/yam_rl_finetuning_<RUN_NAME>/       ← ckpts + learner.log + plots
RUN_NAME        = "hire_lambda09_fixedsuccess"
ONLINE_DATA_DIR = os.path.join(_data_dir, f"yam_rl_rollouts_{RUN_NAME}")
RL_CKPT_DIR     = os.path.join(_ckpt_dir, f"yam_rl_finetuning_{RUN_NAME}")

# Past-run directory used ONLY to seed the HiRE positive/negative buffers.
# Set to None to start HiRE training from scratch (no online data seed) —
# the positive buffer is still seeded by the offline expert (train.npz) and
# the negative buffer starts empty (fills up as new online failures arrive).
# (We previously pointed this at `yam_rl_rollouts_v2` but that was a
# different *baseline* run, so we don't want its episodes to bias our HiRE
# experiment. Fair comparison: HiRE starts with 0 online episodes, just
# like the baseline did.)
HIRE_INIT_DIR   = None

# Optional JSON file with `include`/`exclude` episode-index lists for the
# offline expert npz. Produced by `python scripts/curate_expert.py`. When
# present, ONLY the `include`'d expert trajectories are encoded into the
# HiRE positive buffer. Set to None to use ALL expert trajectories.
HIRE_EXPERT_CURATION_PATH = os.path.join(_data_dir,
    "yam_picknplace_arizonabottle_224", "expert_curation.json")

# ============================================================
# Training algorithm settings
# ============================================================
TRAINING = dict(
    # --- Rollout / training cadence ---
    # MATCHES ORIGINAL rl_finetuning_config.py (20 warmup, 10-episode block,
    # 2000 gradient steps per round). DO NOT change these without reason —
    # the original repo trained successfully with these exact values.
    num_episodes_before_first_training = 20,
    # Collect 20 pure-BC episodes first, then start RL.

    update_every_x_episode             = 10,
    # One training round per 10 new rollouts.

    gradient_steps                     = 2000,
    # 2000 actor+critic updates per training round.
    # Effective data: 2000 × 256 = 512k transitions per round.

    # --- Batch composition ---
    batch_size  = 256,       # 128 expert + 128 online per step
    obs_horizon = 2,         # must match BC training (2 frames of history)
    action_horizon = 16,     # must match BC training
    action_dim     = 7,      # YAM: 6 arm joints + 1 gripper

    # --- RLPD expert ratio (anneals from 70% to 20%) ---
    use_adaptive_expert_ratio   = True,
    adaptive_expert_ratio_start = 0.7,
    adaptive_expert_ratio_end   = 0.2,
    adaptive_expert_ratio_steps = 30000,

    # --- RL algorithm ---
    gamma                = 0.99,
    tau                  = 0.01,     # target critic Polyak rate
    actor_lr             = 1e-4,
    critic_lr            = 1e-4,
    bc_loss_weight       = 100.0,   # BC regularisation weight — matches the original codebase
                                     # (rl_finetuning_config.py model_para["bc_loss_weight"]=140).
                                     # The paper's appendix table reports 100, but real-robot tuning
                                     # in the codebase converged to 140.
    # --- Multi-sample stabilisers (from original) ---
    num_next_noise_samples       = 4,    # multi_sample_next_noise=True, K=4 (target Q averaging)
    num_multi_z_for_actor_loss   = 8,    # sample_multi_z_for_actor_loss=True, K=8 (actor loss averaging)
    use_q_normalization          = True, # divide q_loss by mean(|Q|) for scale stability
    disable_q_loss_for_expert_data = True,  # only push actor toward Q on online states
    # --- BC loss filter (DICE-RL's core innovation; codebase default = active) ---
    use_soft_q_filtering         = True, # turn on the BC filter after warmup
    q_filtering_warmup_steps     = 25000, # use simple Q+BC for first N steps (codebase default)
    # --- HiRE (Hindsight Reward Editing): contrastive_prompt + PBRS dense reward ---
    # When enabled, each transition's reward becomes:
    #     r_t = r_sparse_t + γ·Φ(s_{t+1}) − Φ(s_t)
    # with Φ(s) = reward_weight · ( sim_pos(s) − contrastive_lambda · sim_neg(s) )
    # computed via DINOv2 ViT-S/14 patch features over base + wrist cameras.
    #
    # Positive buffer  : ALL frames of offline expert demos (strided) AND
    #                    ALL frames of online success episodes.
    # Negative buffer  : LAST FRAME ONLY of online failure episodes.
    # The two buffers use *different* logsumexp temperatures so that positives
    # behave like a sharp max (β_pos=10, "is this close to any successful state?")
    # and negatives behave like a smooth mean (β_neg=1, "is this generally
    # similar to failure modes?"). Combined with a small contrastive λ=0.1,
    # this lets positives drive most of the signal while negatives gently push
    # the policy away from common failure patterns.
    use_hire_reward                = True,
    hire_reward_weight             = 1.0,    # scales Φ
    hire_contrastive_lambda        = 0.9,    # weight on neg sim subtraction (was 1.0)
    hire_logsumexp_beta_pos        = 10.0,   # SHARP max over positives
    hire_logsumexp_beta_neg        = 9.0,    # SMOOTH ≈ mean over negatives
    hire_gamma_pbrs                = 0.99,   # discount inside PBRS shaping
    hire_sample_K                  = 64,     # K samples drawn from each buffer
    hire_online_success_frames     = "all",  # all frames of online success → positive
    hire_online_failure_frames     = 15,     # last 15 frames of online failure → negative
                                              #   (was 1; 15 captures the "failure mode" build-up,
                                              #    not just the very last frame)
    hire_expert_frame_stride       = 5,      # subsample offline expert frames (stride)
    # FIFO caps per camera:
    #   pos_buffer (cap=4096): seeded by offline expert; online success frames
    #     get appended over time and gradually push out older expert frames.
    #   neg_buffer (cap=10): only the most-recent online failure end-frames —
    #     small on purpose so the policy adapts to its current failure modes.
    hire_max_pos_buffer_size       = 4096,
    hire_max_neg_buffer_size       = 300,    # 15 frames × 20 recent failures = 300; gives a meaningful
                                              # window of "current failure modes" without being too stale

    # Reward-recipe switch:
    #   False (default) → online success uses HiRE-shaped reward (full method)
    #   True            → online success reverts to sparse reward, like offline
    use_sparse_for_online_success  = True,
    critic_ensemble_size = 5,
    max_grad_norm        = 1.0,
)

# ============================================================
# Network architecture
# ============================================================
NETWORK = dict(
    actor_hidden_dims  = [1024, 1024, 1024],
    critic_hidden_dims = [1024, 1024, 1024],
)

# ============================================================
# Hardware
# ============================================================
HARDWARE = dict(
    can_channel       = "can_follower_l",
    gripper_type      = "linear_4310",
    base_cam_serial   = "218622278369",
    wrist_cam_serial  = "218622271309",
    home_joint_pos    = [-0.010, 0.833, 0.903, -0.598, -0.028, -0.029],
    home_gripper_pos  = 1.0,
    control_hz        = 30.0,
    max_episode_steps = 200,
)

# ============================================================
# ZMQ endpoints (IPC = same machine; TCP = different machines)
# ============================================================
COMM = dict(
    network_server_endpoint     = "ipc:///tmp/feeds/rl_weights",
    network_weight_topic        = "rl_network_weights_topic",
    transitions_server_endpoint = "ipc:///tmp/feeds/rl_transitions",
    transitions_topic           = "rl_transitions_topic",
)
