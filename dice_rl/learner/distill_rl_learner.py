"""
RL Finetuning Learner Node for Real Robot.

This learner follows the structure of train_distill_flow_agent.py but:
1. Does NOT create environments (that's on env_runner side)
2. Receives episode folder paths from env_runner via ZMQ
3. Loads raw zarr data and processes via HybridReplayBuffer
4. Trains DistillResidualRLImgModel with RLPD
5. Sends updated weights back to env_runner
"""

import os
from typing import Dict, Optional, List
import logging
import time

import numpy as np
import torch
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)
log = logging.getLogger(__name__)

from dice_rl.communication.learner_node import Learner
from dice_rl.replay_buffer.hybrid_replay_buffer import HybridReplayBuffer
from dice_rl.model.distill_rl_img import DistillResidualRLImgModel


class DistillRLLearner:
    """
    Learner node for online RL finetuning of diffusion policy on real robot.

    Follows train_distill_flow_agent.py structure but without environment creation.
    Uses ZMQ to receive episode data and send weights.
    """

    def __init__(self, learner_para, model_para, online_learning_para, replay_buffer_para, sparse_execution_horizon):
        """
        Initialize learner from config dicts.

        Args:
            learner_para: Training parameters (batch_size, lr, gamma, etc.)
            model_para: Model architecture parameters (obs_dim, action_dim, hidden_dims, etc.)
            online_learning_para: ZMQ communication parameters
            replay_buffer_para: Replay buffer parameters
            sparse_execution_horizon: From control_para, must match pretrained policy's sparse_action_horizon
        """
        self.sparse_execution_horizon = sparse_execution_horizon
        self.device = learner_para.get("device", "cuda")

        # Dimensions (will be set after model creation from pretrained policy)
        self.horizon_steps = None
        self.cond_steps = None

        # Training parameters
        self.batch_size = learner_para.get("batch_size", 256)
        self.gamma = learner_para.get("gamma", 0.99)
        self.tau = learner_para.get("tau", 0.005)
        self.gradient_steps = learner_para.get("gradient_steps", 100)
        self.update_every_x_episode = learner_para.get("update_every_x_episode", 1)
        self.max_grad_norm = learner_para.get("max_grad_norm", None)

        # RLPD settings
        self.use_rlpd = learner_para.get("use_rlpd", False)
        self.expert_ratio = learner_para.get("expert_ratio", 0.5)

        # Adaptive expert ratio
        self.use_adaptive_expert_ratio = learner_para.get("use_adaptive_expert_ratio", False)
        self.adaptive_expert_ratio_start = learner_para.get("adaptive_expert_ratio_start", 0.7)
        self.adaptive_expert_ratio_end = learner_para.get("adaptive_expert_ratio_end", 0.1)
        self.adaptive_expert_ratio_steps = learner_para.get("adaptive_expert_ratio_steps", 40000)

        # Training control
        self.num_episodes_before_first_training = learner_para.get("num_episodes_before_first_training", 5)

        # Data loading settings (episodes are saved under data_folder_path/raw/)
        self.raw_data_dir = os.path.join(online_learning_para.get("data_folder_path"), "raw")
        self.ft_sensor_configuration = learner_para.get("ft_sensor_configuration", "handle_on_robot")

        # Initialize ZMQ communication
        log.info("Initializing ZMQ communication...")
        self.learner_node = Learner(
            network_server_endpoint=online_learning_para["network_server_endpoint"],
            network_weight_topic=online_learning_para["network_weight_topic"],
            transitions_server_endpoint=online_learning_para["transitions_server_endpoint"],
            transitions_topic=online_learning_para["transitions_topic"],
            network_weight_expire_time_s=online_learning_para.get("network_weight_expire_time_s", 1200),
        )

        # Derive ALL dimensions from pretrained policy BEFORE creating model
        # This ensures the actor is created with correct input dimensions
        from utils.model_io import load_policy
        log.info("Loading pretrained policy to derive dimensions...")
        from omegaconf import OmegaConf
        pretrained_policy, shape_meta_omegaconf, pretrained_cfg = load_policy(
            model_para["pretrained_flow_policy_path"], self.device
        )
        # Convert to plain dict so we can add keys; copy data processing params
        # from BC config so online episode processing is identical to BC pretraining
        shape_meta = OmegaConf.to_container(shape_meta_omegaconf, resolve=True)
        sqfds = pretrained_cfg["task"]["dataset"]["sparse_query_frequency_down_sample_steps"]
        shape_meta["sparse_query_frequency_down_sample_steps"] = sqfds
        self.shape_meta = shape_meta
        print(f"[Learner] sparse_query_frequency_down_sample_steps = {sqfds}")

        # Derive dimensions from pretrained policy (these must match for RL to work)
        # obs_encoder.output_shape() may be (1, 786) etc., so use np.prod like the policy does
        obs_dim = pretrained_policy.obs_feature_dim  # visual feature dim = np.prod(obs_encoder.output_shape())
        assert obs_dim == np.prod(pretrained_policy.obs_encoder.output_shape()), \
            f"obs_feature_dim ({obs_dim}) != np.prod(obs_encoder.output_shape()) ({np.prod(pretrained_policy.obs_encoder.output_shape())})"
        action_dim = shape_meta["action"]["shape"][0]  # per-step action dim (e.g., 9)
        horizon_steps = shape_meta["sample"]["action"]["sparse"]["horizon"]  # sparse_action_horizon from pretraining
        cond_steps = shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["horizon"]

        print(f"[Learner] Derived from pretrained policy: obs_dim={obs_dim}, action_dim={action_dim}, "
              f"horizon_steps={horizon_steps}, cond_steps={cond_steps}")

        # Verify horizon consistency: sparse_execution_horizon (config) must match sparse_action_horizon (pretrained)
        assert horizon_steps == self.sparse_execution_horizon, \
            f"horizon_steps from pretrained policy ({horizon_steps}) != sparse_execution_horizon from config ({self.sparse_execution_horizon}). " \
            f"For RL finetuning, these must match."

        # Update model_para with derived values (overrides config)
        model_para = {
            **model_para,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "horizon_steps": horizon_steps,
            "cond_steps": cond_steps,
        }
        if model_para.get("noise_dim") is None:
            model_para["noise_dim"] = action_dim

        del pretrained_policy  # will be reloaded by model

        # Create model with correct dimensions
        log.info("Creating DistillResidualRLImgModel...")
        self.model = DistillResidualRLImgModel(**model_para)
        log.info(f"Model created: {type(self.model).__name__}")

        # Store dimensions (already derived above, just copy from model_para)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps

        # Store actor config for weight transfer to env_runner
        self.actor_config = {
            "obs_dim": model_para["obs_dim"],
            "action_dim": model_para["action_dim"],
            "cond_steps": self.cond_steps,
            "horizon_steps": self.horizon_steps,
            "hidden_dims": model_para.get("actor_hidden_dims", [256, 256, 256]),
            "activation_type": model_para.get("activation_type", "GELU"),
            "use_layernorm": model_para.get("use_layernorm", False),
        }

        # Access normalizer and obs_encoder from pretrained policy
        self.sparse_normalizer = self.model.pretrained_policy.sparse_normalizer
        self.obs_encoder = self.model.pretrained_policy.obs_encoder
        self.obs_encoder.eval()

        # Set up replay buffer with expert dataset
        self._setup_replay_buffer(learner_para, replay_buffer_para)

        # Create optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(),
            lr=learner_para.get("actor_lr", 1e-4),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=learner_para.get("critic_lr", 3e-4),
        )

        # Dynamics optimizer (if enabled)
        if self.model.use_dynamics:
            self.dynamics_optimizer = torch.optim.Adam(
                self.model.dynamics.parameters(),
                lr=learner_para.get("dynamics_lr", 3e-4),
            )
        else:
            self.dynamics_optimizer = None

        # Training state
        self.training_step = 0
        self.num_episodes_received = 0
        self.training_started = False
        self.processed_episodes = set()  # Track processed episode names for disk-based fallback

        # Wandb setup
        self.use_wandb = learner_para.get("use_wandb", False) and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=learner_para.get("wandb_project", "rl-finetuning"),
                name=learner_para.get("wandb_name", f"learner_{time.strftime('%Y%m%d_%H%M%S')}"),
                config={**learner_para, **model_para},
            )
            log.info("Wandb initialized")

        log.info("DistillRLLearner initialized successfully")

    def _setup_replay_buffer(self, learner_para, replay_buffer_para):
        """
        Set up replay buffer with expert dataset for RLPD.
        Pre-processes expert data through obs_encoder before storing.
        """
        # Pre-process expert dataset if RLPD is enabled
        processed_expert_dataset = None
        expert_dataset_path = learner_para.get("expert_dataset_path")
        if self.use_rlpd and expert_dataset_path is not None:
            from diffusion_policy.dataset.virtual_target_qlearning_dataset import VirtualTargetQLearningDataset
            log.info(f"Loading expert dataset from: {expert_dataset_path}")
            raw_expert_dataset = VirtualTargetQLearningDataset(
                shape_meta=self.model.pretrained_shape_meta,
                dataset_path=expert_dataset_path,
                gamma=self.gamma,
                robot_dt_ms=replay_buffer_para.get("robot_dt_ms", 2.0),
                sparse_query_frequency_down_sample_steps=8,
            )
            log.info(f"RLPD enabled: loaded raw expert dataset with {len(raw_expert_dataset)} transitions")
            log.info("Pre-processing expert dataset through obs_encoder...")
            processed_expert_dataset = self._preprocess_expert_dataset(raw_expert_dataset)
            log.info(f"Expert dataset pre-processed: {len(processed_expert_dataset)} transitions")

        # Create HybridReplayBuffer
        self.replay_buffer = HybridReplayBuffer(
            shape_meta=self.shape_meta,
            max_size=replay_buffer_para.get("max_size", 100000),
            device=self.device,
            gamma=self.gamma,
            robot_dt_ms=replay_buffer_para.get("robot_dt_ms", 50),
            # RLPD
            use_rlpd=self.use_rlpd,
            expert_ratio=self.expert_ratio,
            expert_dataset=processed_expert_dataset,
            # Visual encoder and normalizer from pretrained policy
            obs_encoder=self.obs_encoder,
            sparse_normalizer=self.sparse_normalizer,
        )

    def _preprocess_expert_dataset(self, expert_dataset) -> List:
        """
        Pre-process expert dataset by encoding observations through obs_encoder.

        Follows train_distill_flow_img_agent._preprocess_expert_dataset pattern:
        - Extracts visual features from RGB using pretrained obs_encoder
        - Converts actions from dict to flat tensor
        - Returns list of Transition namedtuples with encoded features

        This is done ONCE at init time so sampling is fast during training.

        Args:
            expert_dataset: VirtualTargetQLearningDataset with raw obs dicts

        Returns:
            List of Transition namedtuples with:
            - actions: tensor (horizon_steps, action_dim)
            - conditions: {"state": tensor (1, feature_dim), "next_state": tensor (1, feature_dim)}
            - rewards, dones, mc_return: tensor (1,)
        """
        from collections import namedtuple
        Transition = namedtuple("Transition", "actions conditions rewards dones mc_return")

        obs_encoder = self.model.pretrained_policy.obs_encoder
        obs_encoder.eval()

        processed_transitions = []
        batch_size = 64

        for i in tqdm(range(0, len(expert_dataset), batch_size),
                      desc="Pre-processing expert data",
                      total=(len(expert_dataset) + batch_size - 1) // batch_size):
            batch_end = min(i + batch_size, len(expert_dataset))
            batch_transitions = [expert_dataset[j] for j in range(i, batch_end)]

            # Batch encode obs and next_obs
            # Expert dataset conditions["obs"] = {"sparse": {"rgb_0": tensor, ...}, "dense": {}}
            # We need the inner "sparse" dict which has the individual obs keys
            obs_batch = [t.conditions["obs"]["sparse"] for t in batch_transitions]
            next_obs_batch = [t.conditions["next_obs"]["sparse"] for t in batch_transitions]

            encoded_obs = self._batch_encode_obs(obs_encoder, obs_batch)
            encoded_next_obs = self._batch_encode_obs(obs_encoder, next_obs_batch)

            for j, t in enumerate(batch_transitions):
                # Convert action to normalized tensor (RL operates in normalized action space)
                action_array = t.actions["sparse"]
                if isinstance(action_array, torch.Tensor):
                    action_tensor = action_array.float()
                else:
                    action_tensor = torch.from_numpy(action_array).float()
                action_tensor = self.sparse_normalizer["action"].normalize(action_tensor)

                # Add cond_steps=1 dim: (feature_dim,) -> (1, feature_dim)
                # model.loss() expects state shape (B, cond_steps, obs_dim)
                processed_transitions.append(Transition(
                    actions=action_tensor,
                    conditions={
                        "state": encoded_obs[j].unsqueeze(0).cpu(),  # (1, feature_dim)
                        "next_state": encoded_next_obs[j].unsqueeze(0).cpu(),  # (1, feature_dim)
                    },
                    rewards=t.rewards,
                    dones=t.dones,
                    mc_return=t.mc_return,
                ))

        return processed_transitions

    def _batch_encode_obs(self, obs_encoder, obs_list: List) -> torch.Tensor:
        """
        Encode a batch of observations through the obs_encoder.

        Args:
            obs_encoder: The pretrained obs_encoder (e.g., TimmObsEncoderWithForce)
            obs_list: List of obs dicts, each with keys like "rgb_0", "robot0_eef_pos", etc.
                      Values are tensors with shape (T, ...)

        Returns:
            Encoded features tensor (batch_size, feature_dim)
        """
        # Build batched obs_dict: {key: (B, T, ...)}
        obs_dict = {}
        first_obs = obs_list[0]
        for key in first_obs.keys():
            tensors = []
            for obs in obs_list:
                val = obs[key]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val).float()
                tensors.append(val)
            obs_dict[key] = torch.stack(tensors).to(self.device)

        # Normalize before encoding (obs_encoder was trained on normalized inputs)
        obs_dict = self.sparse_normalizer.normalize(obs_dict)

        with torch.no_grad():
            features = obs_encoder.forward(obs_dict)  # (B, feature_dim)

        return features

    def get_current_expert_ratio(self, training_step: int) -> float:
        """Calculate current expert ratio for RLPD (matches train_distill_flow_agent)."""
        if not self.use_adaptive_expert_ratio:
            return self.expert_ratio

        if training_step >= self.adaptive_expert_ratio_steps:
            return self.adaptive_expert_ratio_end

        progress = training_step / self.adaptive_expert_ratio_steps
        current_ratio = (
            self.adaptive_expert_ratio_start
            - (self.adaptive_expert_ratio_start - self.adaptive_expert_ratio_end) * progress
        )
        return current_ratio

    def receive_episodes(self) -> List[Dict]:
        """
        Receive all available episode names from env_runner.

        First tries ZMQ, then falls back to scanning disk for episodes
        with rl_metadata.json that haven't been processed yet.

        Returns:
            List of episode data dicts with keys:
            - episode_name: name of the episode folder (e.g., "episode_1742230408")
            - success: whether episode was successful
        """
        episodes = []

        # Primary path: ZMQ (only if server is reachable — pop_data blocks forever if server is down)
        status = self.learner_node.transitions_client.get_topic_status(
            self.learner_node.transitions_topic, timeout_s=2.0
        )
        if status > 0:  # Server reachable and has data
            while True:
                try:
                    episode_data = self.learner_node.receive_transitions()
                except Exception as e:
                    log.error(f"Error receiving transitions: {e}")
                    episode_data = None

                if episode_data is not None:
                    episodes.append(episode_data)
                    log.info(f"Received episode (ZMQ): {episode_data.get('episode_name', 'unknown')}")
                else:
                    break

        # Fallback: scan disk for unprocessed episodes with rl_metadata.json in case ZMQ dies
        if len(episodes) == 0 and os.path.exists(self.raw_data_dir):
            import json
            try:
                for folder_name in os.listdir(self.raw_data_dir):
                    if folder_name in self.processed_episodes:
                        continue
                    metadata_path = os.path.join(self.raw_data_dir, folder_name, "rl_metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        episodes.append(metadata)
                        log.info(f"Received episode (disk): {metadata.get('episode_name', folder_name)}")
            except Exception as e:
                log.error(f"Error scanning disk for episodes: {e}")

        return episodes

    def process_episode(self, episode_data: Dict):
        """
        Process a received episode and add to replay buffer.

        The env_runner saves raw observations to disk via ManipServer.
        HybridReplayBuffer loads raw files, processes them to zarr,
        and converts to RL transitions.

        Args:
            episode_data: Dict with episode_name and success flag
        """
        episode_name = episode_data["episode_name"]
        success = episode_data.get("success", False)

        print(f"[Learner] Processing episode: {episode_name}, success: {success}")

        # Track as processed (even if folder is missing, don't retry)
        self.processed_episodes.add(episode_name)

        # Verify episode folder exists before processing
        episode_folder = os.path.join(self.raw_data_dir, episode_name)
        if not os.path.exists(episode_folder):
            print(f"[Learner] WARNING: Episode folder not found, skipping: {episode_folder}")
            return

        # HybridReplayBuffer handles loading raw files and processing
        num_samples = self.replay_buffer.load_episode_from_disk(
            episode_name=episode_name,
            raw_data_dir=self.raw_data_dir,
            success=success,
            ft_sensor_configuration=self.ft_sensor_configuration,
            has_correction=True,
        )

        self.num_episodes_received += 1
        print(f"[Learner] Episode processed. Total episodes: {self.num_episodes_received}, "
              f"Buffer size: {self.replay_buffer.size}, Samples added: {num_samples}")

    def update_networks(self) -> Dict[str, float]:
        """
        Perform one training step. Follows train_distill_flow_agent.y.

        Returns:
            Dict of loss values for logging
        """
        if self.replay_buffer.size < self.batch_size:
            return {}

        # Get current expert ratio
        current_expert_ratio = self.get_current_expert_ratio(self.training_step)

        # Sample from replay buffer
        # Returns: state, action, reward, next_state, done, mc_return, n_steps, data_source
        state, action, reward, next_state, done, mc_return, n_steps, data_source = \
            self.replay_buffer.sample(self.batch_size, expert_ratio=current_expert_ratio)

        # Generate fresh noise (matching DPPO - noise is not stored, always re-sampled)
        noise = torch.randn(state.shape[0], self.horizon_steps, self.action_dim, device=self.device)

        # Compute losses using model's loss function
        loss_dict = self.model.loss(
            state=state,
            noise=noise,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done,
            gamma=self.gamma,
            training_step=self.training_step,
            n_steps=n_steps,
            data_source=data_source,
        )

        # Update actor
        self.actor_optimizer.zero_grad()
        loss_dict['actor_total'].backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.actor.parameters(), self.max_grad_norm
            )
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        loss_dict['critic_loss'].backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.critic.parameters(), self.max_grad_norm
            )
        self.critic_optimizer.step()

        # Update target networks
        self.model.update_target_networks(tau=self.tau)

        # Update dynamics if enabled
        if self.model.use_dynamics and 'dynamics_loss' in loss_dict:
            self.dynamics_optimizer.zero_grad()
            loss_dict['dynamics_loss'].backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.dynamics.parameters(), self.max_grad_norm
                )
            self.dynamics_optimizer.step()

        self.training_step += 1

        # Convert losses to floats for logging
        metrics = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v.item()
            else:
                metrics[k] = v
        metrics['expert_ratio'] = current_expert_ratio

        return metrics

    def send_weights(self):
        """Send updated model weights to env_runner."""
        actor = self.model.actor
        payloads = {
            "actor_state_dict": {k: v.cpu() for k, v in actor.state_dict().items()},
            "actor_config": self.actor_config,
            "training_step": self.training_step,
        }
        self.learner_node.send_network_weights(payloads)
        print(f"[Learner] Sent updated weights to env_runner (training_step={self.training_step})")

    def save_checkpoint(self, ckpt_path: str):
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "training_step": self.training_step,
            "num_episodes_received": self.num_episodes_received,
        }
        torch.save(checkpoint, ckpt_path)
        log.info(f"Checkpoint saved to: {ckpt_path}")

    def run(self, checkpoint_dir: Optional[str] = None, checkpoint_freq: int = 1000):
        """
        Main learner loop.

        Args:
            checkpoint_dir: Directory to save checkpoints
            checkpoint_freq: Save checkpoint every N training steps
        """
        log.info("Starting learner loop...")

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # First training at num_episodes_before_first_training,
        # then every update_every_x_episode episodes (by divisibility)
        next_training_episode = self.num_episodes_before_first_training

        while True:
            # Wait for and receive episodes
            log.info("Waiting for new episodes...")

            while True:
                episodes = self.receive_episodes()
                if len(episodes) > 0:
                    break
                time.sleep(1.0)

            # Process received episodes
            for episode_data in episodes:
                self.process_episode(episode_data)

            # Check if we've reached the next training trigger
            if self.num_episodes_received < next_training_episode:
                print(f"[Learner] {self.num_episodes_received} episodes, next training at {next_training_episode}")
                continue

            # Set next trigger: next multiple of update_every_x_episode beyond current count
            next_training_episode = (self.num_episodes_received // self.update_every_x_episode + 1) * self.update_every_x_episode
            print(f"[Learner] Training triggered at {self.num_episodes_received} episodes, next training at {next_training_episode}")

            # Training loop
            if not self.training_started:
                print(f"[Learner] === STARTING TRAINING ===")
                print(f"[Learner] Buffer size: {self.replay_buffer.size}")
                self.training_started = True

            num_gradient_steps = self.gradient_steps
            current_expert_ratio = self.get_current_expert_ratio(self.training_step)
                        
            print(f"[Learner] Training for {num_gradient_steps} gradient steps... (expert_ratio={current_expert_ratio:.3f}, training_step={self.training_step}, episodes={self.num_episodes_received})")
            pbar = tqdm(range(num_gradient_steps), desc="Training", leave=False)
            for _ in pbar:
                metrics = self.update_networks()

                if metrics:
                    pbar.set_postfix({
                        'step': self.training_step,
                        'actor': f"{metrics.get('actor_total', 0):.4f}",
                        'critic': f"{metrics.get('critic_loss', 0):.4f}",
                    })

                # Save checkpoint periodically
                if checkpoint_dir is not None and self.training_step % checkpoint_freq == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{self.training_step}.pt")
                    self.save_checkpoint(ckpt_path)

            # Send updated weights to env_runner
            self.send_weights()


def main():
    from dice_rl.config.rl_finetuning_config import (
        learner_para,
        model_para,
        online_learning_para,
        replay_buffer_para,
        control_para,
        checkpoint_folder_path,
    )

    learner = DistillRLLearner(
        learner_para=learner_para,
        model_para=model_para,
        online_learning_para=online_learning_para,
        replay_buffer_para=replay_buffer_para,
        sparse_execution_horizon=control_para["sparse_execution_horizon"],
    )

    checkpoint_dir = os.path.join(checkpoint_folder_path, "rl_finetuning")
    learner.run(checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    main()
