"""
Hybrid replay buffer for real robot RL finetuning.

This buffer combines:
1. Offline expert data (from VirtualTargetQLearningDataset)
2. Online data collected during RL finetuning

Key design decisions:
- Single environment (no parallel envs) for real robot
- Stores raw observations during collection, processes during episode completion
- Actions computed from raw obs using raw_to_action9() (same as offline data)
- MC return uses ceiling-based chunk counting with timestamps (same as offline)
- RGB encoded to features during post-processing before storing
"""

import os
import math
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import namedtuple

import numpy as np
import torch
import zarr

from utils.common_type_conversions import (
    raw_to_obs,
    raw_to_action9,
    obs_to_obs_sample,
    action9_to_action_sample,
)
from utils.data_processing.processing_functions import (
    process_one_episode_into_zarr,
)

log = logging.getLogger(__name__)

# Named tuple for Q-learning transitions (matches dppo's Transition)
Transition = namedtuple("Transition", "actions conditions rewards dones mc_return")


class HybridReplayBuffer:
    """
    Hybrid replay buffer for real robot RL finetuning.

    Storage strategy:
    1. Collect raw observations during episode rollout
    2. On episode completion, process raw data to training chunks (same as offline)
    3. Store encoded features + processed obs/actions for fast sampling
    4. Mix with offline expert data using RLPD sampling

    Key features:
    - Single env design (real robot, not parallel simulation)
    - Raw obs stored during collection, processed at episode end
    - Actions computed from raw data (ts_pose_command, etc.) like offline
    - MC return uses ceiling-based chunk counting with timestamps
    - RGB encoded to features during post-processing
    """

    def __init__(
        self,
        shape_meta: dict,
        max_size: int = 100000,
        device: str = "cuda",
        gamma: float = 0.99,
        robot_dt_ms: float = 2.0,
        # RLPD settings
        use_rlpd: bool = True,
        expert_ratio: float = 0.5,
        expert_dataset=None,  # VirtualTargetQLearningDataset
        # Visual encoder (policy's obs_encoder) and normalizer
        obs_encoder=None,
        sparse_normalizer=None,  # LinearNormalizer from pretrained policy
        # N-step returns
        use_n_step: bool = False,
        n_step: int = 1,
        expert_n_step: int = 1,
    ):
        """
        Initialize hybrid replay buffer.

        Args:
            shape_meta: Shape metadata from task config (defines obs/action structure)
            max_size: Maximum number of training samples to store
            device: Device for tensor storage
            gamma: Discount factor for MC return (per action chunk, not per step)
            robot_dt_ms: Robot control period in ms (default 2.0 for 500Hz)
            use_rlpd: Whether to mix expert + online data
            expert_ratio: Fraction of batch from expert data
            expert_dataset: VirtualTargetQLearningDataset for expert data
            obs_encoder: Policy's obs_encoder for encoding RGB to features
            use_n_step: Whether to use n-step returns for online data
            n_step: Number of steps for n-step returns
            expert_n_step: N-step value for expert data
        """
        self.shape_meta = shape_meta
        self.max_size = max_size
        self.device = device
        self.gamma = gamma
        self.robot_dt_ms = robot_dt_ms

        # Compute action chunk duration (same formula as VirtualTargetQLearningDataset)
        action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
        action_down_sample_steps = shape_meta["sample"]["action"]["sparse"]["down_sample_steps"]
        self.action_chunk_duration_ms = (
            (action_horizon - 1) * action_down_sample_steps + 1
        ) * robot_dt_ms

        # RLPD settings
        self.use_rlpd = use_rlpd
        self.expert_ratio = expert_ratio
        self.expert_dataset = expert_dataset

        # Visual encoder and normalizer
        self.obs_encoder = obs_encoder
        self.sparse_normalizer = sparse_normalizer

        # N-step settings
        self.use_n_step = use_n_step
        self.n_step = n_step
        self.expert_n_step = expert_n_step

        # Storage for processed training samples
        # Each sample is a Transition namedtuple
        self.samples: List[Transition] = []

        # Ongoing episode storage (raw observations)
        # Structure matches what raw_to_obs expects
        self.ongoing_episode: Optional[Dict[str, List]] = None

        # Buffer state
        self.ptr = 0
        self.size = 0
        self.num_episodes = 0

        # ID list from shape_meta
        self.id_list = shape_meta.get("id_list", [0])

        # In-memory zarr for processing episodes from disk
        # This is reused for each episode (no persistence needed)
        self._temp_zarr_store = zarr.MemoryStore()
        self._temp_zarr_root = zarr.open(store=self._temp_zarr_store, mode="w")
        self._temp_zarr_root.create_group("data")

        log.info(f"HybridReplayBuffer initialized")
        log.info(f"  Max samples: {max_size}")
        log.info(f"  Action chunk duration: {self.action_chunk_duration_ms} ms")
        log.info(f"  Gamma: {gamma}")
        if use_rlpd and expert_dataset is not None:
            log.info(f"  RLPD enabled with expert_ratio={expert_ratio}")
            log.info(f"  Expert dataset size: {len(expert_dataset)}")

    def start_episode(self):
        """
        Initialize storage for a new episode.

        Call this at the beginning of each rollout episode.
        """
        self.ongoing_episode = {
            # Raw observation keys (will be populated during add())
            # These match the keys expected by raw_to_obs()
        }
        for key in self.shape_meta["raw"].keys():
            self.ongoing_episode[key] = []

    def add(self, raw_obs: Dict[str, np.ndarray]):
        """
        Add a single timestep of raw observations to the ongoing episode.

        This stores raw data (RGB, poses, timestamps, etc.) which will be
        processed into training samples when the episode completes.

        Args:
            raw_obs: Dict with keys from shape_meta["raw"], each value is
                     a single timestep observation (e.g., shape (3, 224, 224) for RGB)
        """
        if self.ongoing_episode is None:
            raise RuntimeError("Must call start_episode() before add()")

        for key, value in raw_obs.items():
            if key in self.ongoing_episode:
                self.ongoing_episode[key].append(value)

    def end_episode(self, success: bool = True):
        """
        Process completed episode into training samples.

        This converts raw observations to processed obs/actions using the same
        pipeline as offline data, computes MC returns, and stores samples.

        Args:
            success: Whether the episode was successful (reward=1 at end)
        """
        if self.ongoing_episode is None:
            log.warning("end_episode called without active episode")
            return 0

        # Convert lists to arrays
        episode_raw = {}
        for key, values in self.ongoing_episode.items():
            if len(values) > 0:
                episode_raw[key] = np.stack(values, axis=0)

        # Process the episode
        num_samples = self._process_complete_episode(episode_raw, success)

        # Clear ongoing episode
        self.ongoing_episode = None
        self.num_episodes += 1

        return num_samples

    def load_episode_from_disk(
        self,
        episode_name: str,
        raw_data_dir: str,
        success: bool,
        ft_sensor_configuration: str = "handle_on_robot",
        has_correction: bool = True,
    ) -> int:
        """
        Load an episode from raw files saved by ManipServer and convert to RL transitions.

        1. Uses in-memory zarr (no disk persistence for processed data)
        2. Skips VT label computation (not needed for RL)
        3. Directly converts to RL transitions

        Args:
            episode_name: Name of the episode folder (e.g., "episode_1742230408")
            raw_data_dir: Path to the raw data directory containing the episode
                          (e.g., "/path/to/data/raw")
            success: Whether the episode was successful (determines reward)
            ft_sensor_configuration: Force/torque sensor config ("handle_on_robot" or "handle_on_sensor")
            has_correction: Whether the episode has policy_inference.zarr data

        Returns:
            Number of training samples created
        """
        print(f"[ReplayBuffer] Loading episode from disk: {episode_name}")

        # Clear ALL previous episode data from in-memory zarr to prevent memory accumulation
        # Each episode can be ~20MB (RGB frames), so we clear before loading new one
        for existing_episode in list(self._temp_zarr_root["data"].keys()):
            del self._temp_zarr_root["data"][existing_episode]

        # Build config for process_one_episode_into_zarr
        # Note: output_dir is not used since we're using in-memory zarr
        episode_config = {
            "input_dir": raw_data_dir,
            "output_dir": None,  # Not used for in-memory zarr
            "id_list": self.id_list,
            "ft_sensor_configuration": ft_sensor_configuration,
            "num_threads": 10,
            "has_correction": has_correction,
            "save_video": False,
            "max_workers": 16,
        }

        # Convert raw files to zarr format (writes to self._temp_zarr_root in memory)
        try:
            process_one_episode_into_zarr(
                episode_name,
                self._temp_zarr_root,
                episode_config
            )
        except Exception as e:
            log.error(f"Failed to process episode {episode_name}: {e}")
            return 0

        # Load the processed zarr data into a dict
        episode_zarr = self._temp_zarr_root["data"][episode_name]
        episode_raw = {}
        for key in episode_zarr.keys():
            # Load array data into memory (zarr array -> numpy array)
            episode_raw[key] = episode_zarr[key][:]

        print(f"[ReplayBuffer] Loaded episode zarr with {len(episode_raw)} keys: {list(episode_raw.keys())[:5]}...")

        # Convert to RL transitions using existing method
        num_samples = self._process_complete_episode(episode_raw, success)

        self.num_episodes += 1
        log.info(f"Episode {episode_name} processed: {num_samples} samples created")

        return num_samples

    def _process_complete_episode(
        self,
        episode_raw: Dict[str, np.ndarray],
        success: bool
    ) -> int:
        """
        Convert a complete episode to training samples and store them.

        This follows the same logic as VirtualTargetQLearningDataset:
        1. Convert raw data to obs/action using raw_to_obs/raw_to_action9
        2. Sample training transitions using the same sampling logic
        3. Compute MC returns using ceiling-based chunk counting
        4. Encode RGB to features using obs_encoder

        Args:
            episode_raw: Dict of raw observations for the episode
            success: Whether episode succeeded (determines final reward)

        Returns:
            Number of training samples created
        """
        # Step 1: Convert raw to obs/action
        episode_data = {}
        raw_to_obs(episode_raw, episode_data, self.shape_meta)
        raw_to_action9(
            episode_raw,
            episode_data,
            self.id_list,
            self.shape_meta
        )

        # Add timestamps to episode_data["obs"] (raw_to_obs doesn't copy these)
        # These are needed for sampling observations at specific time points
        for id in self.id_list:
            episode_data["obs"][f"rgb_time_stamps_{id}"] = episode_raw[f"rgb_time_stamps_{id}"]
            episode_data["obs"][f"robot_time_stamps_{id}"] = episode_raw[f"robot_time_stamps_{id}"]

        # Get timestamps
        rgb_timestamps = np.squeeze(episode_data["obs"]["rgb_time_stamps_0"])

        # Step 2: Determine valid sample indices
        # Similar to SequenceSampler logic - find valid RGB query indices
        sparse_query_down_sample = self.shape_meta.get(
            "sparse_query_frequency_down_sample_steps", 1
        )

        # Get obs and action horizons
        action_horizon = self.shape_meta["sample"]["action"]["sparse"]["horizon"]
        action_down_sample = self.shape_meta["sample"]["action"]["sparse"]["down_sample_steps"]

        # Find valid range of RGB indices that can serve as query points
        num_rgb_frames = len(rgb_timestamps)

        # Need enough future frames for action chunk
        # Action sampling needs action_id + (horizon-1)*down_sample frames
        action_timestamps = episode_data["action_time_stamps"]

        valid_rgb_indices = []
        for rgb_idx in range(0, num_rgb_frames, sparse_query_down_sample):
            query_time = rgb_timestamps[rgb_idx]

            # Check if we have enough action frames
            action_id = np.searchsorted(action_timestamps, query_time, side="left")
            action_end = action_id + (action_horizon - 1) * action_down_sample

            if action_end < len(action_timestamps):
                valid_rgb_indices.append(rgb_idx)

        if len(valid_rgb_indices) == 0:
            print(f"[ReplayBuffer] Episode too short for any valid samples (num_rgb_frames={num_rgb_frames})")
            return 0

        print(f"[ReplayBuffer] Episode has {num_rgb_frames} RGB frames, {len(valid_rgb_indices)} valid sample indices")

        # Terminal state info
        last_rgb_idx = valid_rgb_indices[-1]
        ts_end_query = rgb_timestamps[last_rgb_idx]

        # Step 3: Create training samples for each valid index
        samples_created = 0

        for rgb_idx in valid_rgb_indices:
            query_time = rgb_timestamps[rgb_idx]

            # Sample obs (current state)
            obs_sample, base_pose = self._sample_obs_at_query(
                episode_data, rgb_idx, query_time
            )

            # Sample next obs
            next_query_time = query_time + self.action_chunk_duration_ms
            next_rgb_idx = np.searchsorted(rgb_timestamps, next_query_time, side="left")
            next_rgb_idx = min(next_rgb_idx, num_rgb_frames - 1)

            next_obs_sample, _ = self._sample_obs_at_query(
                episode_data, next_rgb_idx, rgb_timestamps[next_rgb_idx]
            )

            # Sample action
            action_sample = self._sample_action_at_query(
                episode_data, query_time, base_pose
            )

            # Compute reward and done
            is_terminal = (rgb_idx == last_rgb_idx)
            reward = 1.0 if (is_terminal and success) else 0.0
            done = 1.0 if is_terminal else 0.0

            # Compute MC return using ceiling-based chunk counting
            time_diff_ms = ts_end_query - query_time
            num_chunks_to_end = math.ceil(time_diff_ms / self.action_chunk_duration_ms)
            mc_return = (self.gamma ** num_chunks_to_end) if success else 0.0

            # Encode RGB to features if encoder available
            if self.obs_encoder is not None:
                obs_sample = self._encode_obs_features(obs_sample)
                next_obs_sample = self._encode_obs_features(next_obs_sample)

            # Convert action to normalized tensor (RL operates in normalized action space)
            action_array = action_sample["sparse"]  # (horizon_steps, action_dim)
            action_tensor = torch.from_numpy(action_array).float()
            action_tensor = self.sparse_normalizer["action"].normalize(action_tensor)

            # obs_sample / next_obs_sample are (1, feature_dim) tensors after encoding (cond_steps=1)

            # Create transition
            transition = Transition(
                actions=action_tensor,
                conditions={
                    "state": obs_sample,
                    "next_state": next_obs_sample,
                },
                rewards=torch.tensor([reward], dtype=torch.float32),
                dones=torch.tensor([done], dtype=torch.float32),
                mc_return=torch.tensor([mc_return], dtype=torch.float32),
            )

            # Store sample
            self._store_sample(transition)
            samples_created += 1

        print(f"[ReplayBuffer] Created {samples_created} transitions, buffer size: {self.size}")
        return samples_created

    def _sample_obs_at_query(
        self,
        episode_data: Dict,
        rgb_idx: int,
        query_time: float,
    ) -> Tuple[Dict, List]:
        """
        Sample observation at a query time point.

        Follows the same logic as SequenceSampler for sampling sparse obs.

        Returns:
            Tuple of (processed_obs, base_pose)
        """
        obs = episode_data["obs"]
        sparse_obs = {}

        robot_timestamps = np.squeeze(obs["robot_time_stamps_0"])

        for key, attr in self.shape_meta["sample"]["obs"]["sparse"].items():
            horizon = attr["horizon"]
            down_sample = attr["down_sample_steps"]

            obs_type = self.shape_meta["obs"][key].get("type", "low_dim")

            if obs_type == "rgb":
                # Sample RGB frames
                frames = []
                for h in range(horizon):
                    offset = (horizon - 1 - h) * down_sample
                    idx = max(0, rgb_idx - offset)
                    frames.append(obs[key][idx])
                sparse_obs[key] = np.stack(frames, axis=0)
            else:
                # Sample low-dim data
                samples = []
                for h in range(horizon):
                    offset_time = (horizon - 1 - h) * down_sample * self.robot_dt_ms
                    target_time = query_time - offset_time
                    idx = np.searchsorted(robot_timestamps, target_time, side="left")
                    idx = np.clip(idx, 0, len(robot_timestamps) - 1)
                    samples.append(obs[key][idx])
                sparse_obs[key] = np.stack(samples, axis=0)

        # Process obs to obs_sample (relative pose computation, etc.)
        obs_processed, base_pose = obs_to_obs_sample(
            obs_sparse=sparse_obs,
            obs_dense={},
            shape_meta=self.shape_meta,
            reshape_mode="reshape",
            id_list=self.id_list,
        )

        return obs_processed["sparse"], base_pose

    def _sample_action_at_query(
        self,
        episode_data: Dict,
        query_time: float,
        base_pose: List,
    ) -> Dict:
        """
        Sample action chunk at a query time point.

        Follows the same logic as SequenceSampler for sampling sparse action.
        """
        action = episode_data["action"]
        action_timestamps = episode_data["action_time_stamps"]

        horizon = self.shape_meta["sample"]["action"]["sparse"]["horizon"]
        down_sample = self.shape_meta["sample"]["action"]["sparse"]["down_sample_steps"]

        # Find action index at query time
        action_id = np.searchsorted(action_timestamps, query_time, side="left")

        # Sample action chunk
        action_indices = [
            action_id + i * down_sample
            for i in range(horizon)
        ]
        action_indices = np.clip(action_indices, 0, len(action) - 1)
        sparse_action = action[action_indices]

        # Compute relative action using base_pose from obs
        action_processed = action9_to_action_sample(
            action_sparse=sparse_action,
            action_dense=np.array([]),
            id_list=self.id_list,
            base_pose=base_pose,
            shape_meta=self.shape_meta,
        )

        return action_processed

    def _encode_obs_features(self, obs: Dict) -> torch.Tensor:
        """
        Encode observations to features using the policy's obs_encoder.

        The obs_encoder (TimmObsEncoderWithForce) takes an obs_dict with:
        - RGB keys: (B, T, C, H, W)
        - low_dim keys: (B, T, D)

        And returns a single feature tensor of shape (B, feature_dim).

        Args:
            obs: Dict with RGB and low-dim observations, each with shape (T, ...)

        Returns:
            Feature tensor of shape (1, feature_dim) = (cond_steps, obs_dim) for this single sample
        """
        if self.obs_encoder is None:
            raise RuntimeError("obs_encoder must be set to encode features")

        # Prepare obs_dict with batch dimension for obs_encoder
        obs_dict = {}
        for key, value in obs.items():
            obs_type = self.shape_meta["obs"].get(key, {}).get("type", "low_dim")

            # Skip timestamp keys - obs_encoder doesn't use them
            if obs_type == "timestamp":
                continue

            # Convert to tensor if needed
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value).float()
            else:
                tensor = value.float()

            # Add batch dimension: (T, ...) -> (1, T, ...)
            tensor = tensor.unsqueeze(0).to(self.device)
            obs_dict[key] = tensor

        # Normalize obs before encoding (obs_encoder was trained on normalized inputs)
        # sparse_normalizer.normalize handles: pos→[-1,1], rot→identity, rgb→identity
        obs_dict = self.sparse_normalizer.normalize(obs_dict)

        # Forward through obs_encoder
        with torch.no_grad():
            features = self.obs_encoder.forward(obs_dict)  # (1, feature_dim)

        # Remove batch dim, add cond_steps=1 dim for model.loss() compatibility
        # model.loss() expects state shape (B, cond_steps, obs_dim)
        return features.squeeze(0).unsqueeze(0).cpu()  # (1, feature_dim) = (cond_steps, obs_dim)

    def _store_sample(self, transition: Transition):
        """Store a training sample in the buffer."""
        if self.size < self.max_size:
            self.samples.append(transition)
            self.size += 1
        else:
            # Circular buffer replacement
            self.samples[self.ptr] = transition

        self.ptr = (self.ptr + 1) % self.max_size

    def sample(
        self,
        batch_size: int,
        expert_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions for training.

        Args:
            batch_size: Number of transitions to sample
            expert_ratio: Override for expert data ratio (for RLPD)

        Returns:
            Tuple of (state, action, reward, next_state, done, mc_return, n_steps, data_source)
        """
        if self.use_rlpd and self.expert_dataset is not None:
            return self._sample_rlpd(batch_size, expert_ratio)
        else:
            return self._sample_standard(batch_size)

    def _collate_transitions(
        self, transitions: List[Transition], data_source_labels: List[float]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Collate a list of transitions into batched tensors.

        Args:
            transitions: List of Transition namedtuples
            data_source_labels: List of floats (0.0=online, 1.0=expert)

        Returns:
            Tuple of (state, action, reward, next_state, done, mc_return, n_steps, data_source)
        """
        states = [t.conditions["state"] for t in transitions]
        next_states = [t.conditions["next_state"] for t in transitions]
        actions = [t.actions for t in transitions]
        rewards = [t.rewards for t in transitions]
        dones = [t.dones for t in transitions]
        mc_returns = [t.mc_return for t in transitions]

        state = torch.stack(states).to(self.device)
        action = torch.stack(actions).to(self.device)
        reward = torch.stack(rewards).to(self.device)
        next_state = torch.stack(next_states).to(self.device)
        done = torch.stack(dones).to(self.device)
        mc_return = torch.stack(mc_returns).to(self.device)
        n_steps = torch.ones_like(reward)
        data_source = torch.tensor(
            data_source_labels, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        return state, action, reward, next_state, done, mc_return, n_steps, data_source

    def _sample_standard(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample from online data only."""
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        indices = np.random.randint(0, self.size, size=batch_size)
        transitions = [self.samples[idx] for idx in indices]
        labels = [0.0] * batch_size
        return self._collate_transitions(transitions, labels)

    def _sample_rlpd(
        self,
        batch_size: int,
        expert_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        RLPD sampling: mix expert data with online data.

        Args:
            batch_size: Total batch size
            expert_ratio: Fraction from expert data (default: self.expert_ratio)
        """
        current_ratio = expert_ratio if expert_ratio is not None else self.expert_ratio

        expert_batch_size = int(batch_size * current_ratio)
        online_batch_size = batch_size - expert_batch_size

        transitions = []
        labels = []

        # Sample expert data
        if expert_batch_size > 0 and self.expert_dataset is not None:
            expert_indices = np.random.randint(
                0, len(self.expert_dataset), size=expert_batch_size
            )
            for idx in expert_indices:
                transitions.append(self.expert_dataset[idx])
                labels.append(1.0)

        # Sample online data
        if online_batch_size > 0 and self.size > 0:
            online_indices = np.random.randint(0, self.size, size=online_batch_size)
            for idx in online_indices:
                transitions.append(self.samples[idx])
                labels.append(0.0)
        elif online_batch_size > 0:
            # No online data yet, sample more expert data
            extra_expert_indices = np.random.randint(
                0, len(self.expert_dataset), size=online_batch_size
            )
            for idx in extra_expert_indices:
                transitions.append(self.expert_dataset[idx])
                labels.append(1.0)

        return self._collate_transitions(transitions, labels)

    def __len__(self) -> int:
        """Return number of online samples stored."""
        return self.size

    def get_stats(self) -> Dict[str, Any]:
        """Return buffer statistics."""
        return {
            "online_samples": self.size,
            "num_episodes": self.num_episodes,
            "expert_samples": len(self.expert_dataset) if self.expert_dataset else 0,
            "max_size": self.max_size,
            "expert_ratio": self.expert_ratio,
        }

    def clear(self):
        """Clear the online replay buffer."""
        self.samples = []
        self.ptr = 0
        self.size = 0
        self.ongoing_episode = None
        log.info("HybridReplayBuffer cleared (online data only)")
