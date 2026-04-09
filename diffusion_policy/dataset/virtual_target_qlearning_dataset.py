"""
VirtualTargetQLearningDataset: Extends VirtualTargetDataset for Q-learning / RLPD.

This dataset adds reward, done, and MC return to the base VirtualTargetDataset.
For offline expert demonstrations:
- Only the last valid sample in each episode gets reward=1, done=True
- All other samples get reward=0, done=False
- MC return is computed based on discounted future rewards using ceiling-based chunk counting

Design rationale:
- Reuses existing SequenceSampler infrastructure for obs/action sampling
- Adds reward/done/mc_return tracking per (episode_id, rgb_query_id) index
- Compatible with RLPD (mixing offline expert + online data)

Key insight for MC return:
- Discount is per ACTION CHUNK, not per training sample
- num_chunks_to_end = ceiling((ts_end_query - ts_current) / action_chunk_duration_ms)
- The ceiling ensures that any state not yet at terminal requires at least 1 transition
"""

import math

from collections import namedtuple

import numpy as np
import torch
from tqdm import tqdm

from diffusion_policy.dataset.virtual_target_dataset import VirtualTargetDataset


# Named tuple for Q-learning transitions (matches dppo's Transition)
Transition = namedtuple("Transition", "actions conditions rewards dones mc_return")


class VirtualTargetQLearningDataset(VirtualTargetDataset):
    """
    Extends VirtualTargetDataset to include rewards, dones, and MC returns for Q-learning.

    For offline expert demonstrations:
    - reward = 1.0 only at the last valid sample of each episode (done=True)
    - reward = 0.0 for all other samples (done=False)
    - mc_return = gamma^num_chunks_to_end where num_chunks_to_end is ceiling-based

    Args:
        gamma: Discount factor for MC return computation (default 0.99)
        robot_dt_ms: Robot control period in milliseconds (default 2.0 for 500Hz)
        All other args are passed to VirtualTargetDataset
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        gamma: float = 0.99,
        robot_dt_ms: float = 2.0,  # Robot at 500Hz = 2ms per step
        sparse_query_frequency_down_sample_steps: int = 1,
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        hack_linear_interpolated_dense_action: bool = False,
        normalize_wrench: bool = False,
        weighted_sampling: int = 1,
        correction_horizon: int = 1,
    ):
        # Force include_next_obs=True for Q-learning (need s_{t+1} for bootstrapping)
        super().__init__(
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            sparse_query_frequency_down_sample_steps=sparse_query_frequency_down_sample_steps,
            action_padding=action_padding,
            temporally_independent_normalization=temporally_independent_normalization,
            seed=seed,
            val_ratio=val_ratio,
            hack_linear_interpolated_dense_action=hack_linear_interpolated_dense_action,
            normalize_wrench=normalize_wrench,
            weighted_sampling=weighted_sampling,
            correction_horizon=correction_horizon,
            include_next_obs=True,  # Always True for Q-learning
        )

        self.gamma = gamma
        self.robot_dt_ms = robot_dt_ms

        # Compute action chunk duration from shape_meta
        # Action chunk spans from action_id to action_id + (h-1)*d, next state at action_id + (h-1)*d + 1
        # So duration = ((h-1)*d + 1) * robot_dt_ms
        action_horizon = self.shape_meta["sample"]["action"]["sparse"]["horizon"]
        action_down_sample_steps = self.shape_meta["sample"]["action"]["sparse"]["down_sample_steps"]
        self.action_chunk_duration_ms = ((action_horizon - 1) * action_down_sample_steps + 1) * robot_dt_ms

        # Build reward, done, mc_return arrays indexed by self.sampler.indices
        self._build_reward_done_mc_return()

        print(f"[VirtualTargetQLearningDataset] gamma={gamma}")
        print(f"[VirtualTargetQLearningDataset] action_chunk_duration_ms={self.action_chunk_duration_ms}")
        print(f"[VirtualTargetQLearningDataset] Total transitions: {len(self)}")
        print(f"[VirtualTargetQLearningDataset] Num terminal samples: {int(self.dones.sum())}")

    def _build_reward_done_mc_return(self):
        """
        Build reward, done, and mc_return arrays for all indices in the sampler.

        Key insight: discount is per ACTION CHUNK, not per training sample.
        - num_chunks_to_end = ceiling((ts_end_query - ts_current) / action_chunk_duration_ms)
        - The ceiling ensures any state not at terminal requires at least 1 transition
        - mc_return = gamma^num_chunks_to_end
        """
        indices = self.sampler.indices
        num_samples = len(indices)

        self.rewards = torch.zeros(num_samples, dtype=torch.float32)
        self.dones = torch.zeros(num_samples, dtype=torch.float32)
        self.mc_returns = torch.zeros(num_samples, dtype=torch.float32)

        # Group samples by episode_id
        # episode_to_samples: {episode_id: [(sample_idx, rgb_query_id), ...]}
        episode_to_samples = {}
        for sample_idx, (episode_id, rgb_query_id) in enumerate(indices):
            if episode_id not in episode_to_samples:
                episode_to_samples[episode_id] = []
            episode_to_samples[episode_id].append((sample_idx, rgb_query_id))

        for episode_id, sample_list in tqdm(
            episode_to_samples.items(),
            desc="Building reward/done/mc_return"
        ):
            episode_key = f"episode_{episode_id}"
            data_episode = self.replay_buffer["data"][episode_key]

            # Get RGB timestamps for this episode
            rgb_timestamps = np.squeeze(data_episode["obs"]["rgb_time_stamps_0"])

            # Find the last valid rgb_query_id in this episode's samples
            # (this is the terminal state for this episode)
            sample_list_sorted = sorted(sample_list, key=lambda x: x[1])
            last_sample_idx, last_rgb_query_id = sample_list_sorted[-1]
            ts_end_query = rgb_timestamps[last_rgb_query_id]

            # Last sample gets reward=1, done=True
            self.rewards[last_sample_idx] = 1.0
            self.dones[last_sample_idx] = 1.0

            # Compute mc_return for each sample using ceiling-based chunk counting
            for sample_idx, rgb_query_id in sample_list_sorted:
                ts_current = rgb_timestamps[rgb_query_id]
                time_diff_ms = ts_end_query - ts_current

                # Ceiling: any partial chunk still requires 1 full transition
                num_chunks_to_end = math.ceil(time_diff_ms / self.action_chunk_duration_ms)

                self.mc_returns[sample_idx] = self.gamma ** num_chunks_to_end

        # Print statistics
        print(f"[VirtualTargetQLearningDataset] Reward distribution: "
              f"0.0: {(self.rewards == 0).sum().item()}, 1.0: {(self.rewards == 1).sum().item()}")
        print(f"[VirtualTargetQLearningDataset] MC return range: "
              f"[{self.mc_returns.min().item():.4f}, {self.mc_returns.max().item():.4f}]")

    def __getitem__(self, idx: int) -> Transition:
        """
        Returns a Transition namedtuple containing:
        - actions: action dict with 'sparse' key
        - conditions: dict with 'obs' (current) and 'next_obs'
        - rewards: scalar reward tensor (shape: [1])
        - dones: scalar done flag tensor (shape: [1])
        - mc_return: Monte Carlo return tensor (shape: [1])
        """
        # Get base data from parent class (includes next_obs due to include_next_obs=True)
        torch_data = super().__getitem__(idx)

        # Extract components
        obs = torch_data["obs"]
        action = torch_data["action"]
        next_obs = torch_data["obs_next"]

        # Get reward, done, mc_return for this index
        reward = self.rewards[idx].unsqueeze(0)  # Shape: (1,)
        done = self.dones[idx].unsqueeze(0)      # Shape: (1,)
        mc_return = self.mc_returns[idx].unsqueeze(0)  # Shape: (1,)

        # Build conditions dict (following dppo's convention)
        conditions = {
            "obs": obs,
            "next_obs": next_obs,
        }

        return Transition(
            actions=action,
            conditions=conditions,
            rewards=reward,
            dones=done,
            mc_return=mc_return,
        )

# if __name__ == "__main__":
#     # Config from /home/zhanyi/PyriteML/checkpoints/2025.07.03_18.57.39_belt_angled_150_dp/.hydra/config.yaml
#     # with sparse_action_horizon changed from 32 to 8
#     sparse_obs_rgb_down_sample_steps = 10
#     sparse_obs_rgb_horizon = 1
#     sparse_obs_low_dim_down_sample_steps = 5
#     sparse_obs_low_dim_horizon = 2
#     sparse_action_down_sample_steps = 50
#     sparse_action_horizon = 8  # Changed from 32 to 8
#     robot_dt_ms = 2.0
#     action_chunk_duration_ms = ((sparse_action_horizon - 1) * sparse_action_down_sample_steps + 1) * robot_dt_ms

#     config = {
#         "shape_meta": {
#             "id_list": [0],
#             "raw": {
#                 "rgb_0": {"shape": [3, 224, 224], "type": "rgb"},
#                 "ts_pose_fb_0": {"shape": [7], "type": "low_dim"},
#                 "ts_pose_command_0": {"shape": [7], "type": "low_dim"},
#                 "ts_pose_virtual_target_0": {"shape": [7], "type": "low_dim"},
#                 "stiffness_0": {"shape": [1], "type": "low_dim"},
#                 "wrench_0": {"shape": [6], "type": "low_dim"},
#                 "rgb_time_stamps_0": {"shape": [1], "type": "timestamp"},
#                 "robot_time_stamps_0": {"shape": [1], "type": "timestamp"},
#                 "wrench_time_stamps_0": {"shape": [1], "type": "timestamp"},
#             },
#             "obs": {
#                 "rgb_0": {"shape": [3, 224, 224], "type": "rgb"},
#                 "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
#                 "robot0_eef_rot_axis_angle": {
#                     "shape": [6],
#                     "type": "low_dim",
#                     "rotation_rep": "rotation_6d",
#                 },
#                 "rgb_time_stamps_0": {"shape": [1], "type": "timestamp"},
#                 "robot_time_stamps_0": {"shape": [1], "type": "timestamp"},
#             },
#             "action": {"shape": [9], "rotation_rep": "rotation_6d"},
#             "sample": {
#                 "obs": {
#                     "sparse": {
#                         "rgb_0": {"horizon": sparse_obs_rgb_horizon, "down_sample_steps": sparse_obs_rgb_down_sample_steps},
#                         "robot0_eef_pos": {"horizon": sparse_obs_low_dim_horizon, "down_sample_steps": sparse_obs_low_dim_down_sample_steps},
#                         "robot0_eef_rot_axis_angle": {"horizon": sparse_obs_low_dim_horizon, "down_sample_steps": sparse_obs_low_dim_down_sample_steps},
#                     },
#                 },
#                 "action": {
#                     "sparse": {
#                         "horizon": sparse_action_horizon,
#                         "down_sample_steps": sparse_action_down_sample_steps,
#                     }
#                 },
#                 "training_duration_per_sparse_query": action_chunk_duration_ms,
#             },
#         },
#         "dataset_path": "/home/zhanyi/PyriteML/data/belt_zhanyi_processed",
#         "gamma": 0.99,
#         "robot_dt_ms": robot_dt_ms,
#         "sparse_query_frequency_down_sample_steps": 4,
#         "action_padding": True,
#         "temporally_independent_normalization": False,
#         "seed": 42,
#         "val_ratio": 0.0,  # No validation split for debugging
#         "hack_linear_interpolated_dense_action": False,
#         "normalize_wrench": False,
#         "weighted_sampling": 1,
#         "correction_horizon": 1,
#     }

#     print("=" * 100)
#     print("[DEBUG] Config Summary:")
#     print(f"  sparse_action_horizon: {sparse_action_horizon}")
#     print(f"  sparse_action_down_sample_steps: {sparse_action_down_sample_steps}")
#     print(f"  action_chunk_duration_ms: {action_chunk_duration_ms}")
#     print(f"  sparse_query_frequency_down_sample_steps: {config['sparse_query_frequency_down_sample_steps']}")
#     print(f"  gamma: {config['gamma']}")
#     print(f"  dataset_path: {config['dataset_path']}")
#     print("=" * 100)

#     dataset = VirtualTargetQLearningDataset(
#         shape_meta=config["shape_meta"],
#         dataset_path=config["dataset_path"],
#         gamma=config["gamma"],
#         robot_dt_ms=config["robot_dt_ms"],
#         sparse_query_frequency_down_sample_steps=config["sparse_query_frequency_down_sample_steps"],
#         action_padding=config["action_padding"],
#         temporally_independent_normalization=config["temporally_independent_normalization"],
#         seed=config["seed"],
#         val_ratio=config["val_ratio"],
#         hack_linear_interpolated_dense_action=config["hack_linear_interpolated_dense_action"],
#         normalize_wrench=config["normalize_wrench"],
#         weighted_sampling=config["weighted_sampling"],
#         correction_horizon=config["correction_horizon"],
#     )

#     print(f"\n[main] Dataset length: {len(dataset)}")
#     print(f"[main] Action chunk duration: {dataset.action_chunk_duration_ms} ms")
#     print("\n[main] Debug test completed!")
