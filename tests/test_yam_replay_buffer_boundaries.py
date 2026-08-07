import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dice_rl.replay_buffer.yam_replay_buffer import YAMReplayBuffer


class YAMReplayBufferBoundaryTest(unittest.TestCase):
    @staticmethod
    def _write_expert(path: Path, episode_length: int = 5) -> None:
        states = np.arange(episode_length * 7, dtype=np.float32).reshape(
            episode_length, 7
        )
        np.savez(
            path,
            states=states,
            actions=states.copy(),
            images=np.zeros((episode_length, 6, 2, 2), dtype=np.uint8),
            traj_lengths=np.array([episode_length], dtype=np.int64),
        )

    def test_expert_next_obs_stays_inside_episode(self):
        """The terminal expert transition must not read the next trajectory."""
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            episode_length = 5
            action_horizon = 2
            total = episode_length * 2

            states = np.repeat(
                np.arange(total, dtype=np.float32)[:, None],
                repeats=7,
                axis=1,
            )
            images = np.repeat(
                np.arange(total, dtype=np.uint8)[:, None, None, None],
                repeats=6 * 2 * 2,
                axis=1,
            ).reshape(total, 6, 2, 2)
            expert_path = tmp_path / "expert.npz"
            np.savez(
                expert_path,
                states=states,
                actions=states.copy(),
                images=images,
                traj_lengths=np.array(
                    [episode_length, episode_length],
                    dtype=np.int64,
                ),
            )

            replay = YAMReplayBuffer(
                expert_npz_path=str(expert_path),
                online_data_dir=str(tmp_path / "online"),
                obs_horizon=2,
                action_dim=7,
                action_horizon=action_horizon,
                device="cpu",
            )

            expected_indices = np.array(
                [
                    [0, 0, 2],
                    [1, 0, 2],
                    [2, 0, 2],
                    [5, 5, 7],
                    [6, 5, 7],
                    [7, 5, 7],
                ],
                dtype=np.int64,
            )
            np.testing.assert_array_equal(
                replay._expert_indices,
                expected_indices,
            )

            # Force sampling of the first episode's terminal transition. Its
            # next_obs history must end at frame 4, not frame 5 from the next
            # episode.
            replay._expert_indices = expected_indices[2:3]
            batch = replay._sample_expert(1, torch.device("cpu"))

            self.assertEqual(batch["reward"].item(), 1.0)
            self.assertEqual(batch["done"].item(), 1.0)
            self.assertEqual(
                batch["next_obs"]["joint_pos"][0, -1, 0].item(),
                4.0,
            )
            self.assertEqual(batch["action"][0, -1, 0].item(), 3.0)

    def test_online_transitions_reference_one_compact_episode_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            expert_path = tmp_path / "expert.npz"
            self._write_expert(expert_path)
            replay = YAMReplayBuffer(
                expert_npz_path=str(expert_path),
                online_data_dir=str(tmp_path / "online"),
                obs_horizon=2,
                action_dim=7,
                action_horizon=2,
                max_online_size=10,
                device="cpu",
            )

            states = np.arange(3 * 7, dtype=np.float32).reshape(3, 7)
            actions = 100 + states
            images = np.zeros((3, 6, 2, 2), dtype=np.float32)
            images[1] = 0.5
            images[2] = 1.0
            replay.add_episode({
                "states": states,
                "actions": actions,
                "images": images,
                "rewards": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                "dones": np.array([False, False, True]),
            })

            self.assertEqual(list(replay._online), [(0, 0)])
            self.assertEqual(len(replay._online_episodes), 1)
            stored = replay._online_episodes[0]
            self.assertEqual(stored["images"].dtype, np.uint8)
            self.assertEqual(int(stored["images"][1, 0, 0, 0]), 128)
            # Only unique episode arrays are retained; no materialised obs or
            # next_obs dictionaries live in the transition ring.
            self.assertLess(replay.online_storage_bytes, 2_000)

            batch = replay._sample_online(1, torch.device("cpu"))
            np.testing.assert_array_equal(
                batch["action"][0].numpy(), actions[:2]
            )
            self.assertEqual(batch["reward"].item(), 1.0)
            self.assertEqual(batch["done"].item(), 1.0)
            self.assertEqual(
                batch["next_obs"]["joint_pos"][0, -1, 0].item(),
                states[2, 0],
            )

    def test_expert_image_sidecar_is_preferred(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            expert_path = tmp_path / "expert.npz"
            self._write_expert(expert_path)
            sidecar = tmp_path / "expert_images.npy"
            np.save(sidecar, np.full((5, 6, 2, 2), 123, dtype=np.uint8))

            replay = YAMReplayBuffer(
                expert_npz_path=str(expert_path),
                online_data_dir=str(tmp_path / "online"),
                obs_horizon=2,
                action_dim=7,
                action_horizon=2,
                device="cpu",
            )

            self.assertIsInstance(replay._expert_images, np.memmap)
            self.assertEqual(int(replay._expert_images[0, 0, 0, 0]), 123)

    def test_online_ring_eviction_releases_unreferenced_episode_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            expert_path = tmp_path / "expert.npz"
            self._write_expert(expert_path)
            replay = YAMReplayBuffer(
                expert_npz_path=str(expert_path),
                online_data_dir=str(tmp_path / "online"),
                obs_horizon=2,
                action_dim=7,
                action_horizon=2,
                max_online_size=3,
                device="cpu",
            )

            def episode(offset):
                states = offset + np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
                return {
                    "states": states,
                    "actions": states.copy(),
                    "images": np.full((5, 6, 2, 2), offset, dtype=np.uint8),
                    "rewards": np.zeros(5, dtype=np.float32),
                    "dones": np.array([False, False, False, False, True]),
                }

            replay.add_episode(episode(1))
            self.assertEqual(set(replay._online_episodes), {0})
            replay.add_episode(episode(2))

            self.assertEqual(len(replay._online), 3)
            self.assertEqual(set(replay._online_episodes), {1})
            self.assertEqual(set(ep_id for ep_id, _ in replay._online), {1})
            self.assertEqual(replay._online_episode_refcounts, {1: 3})

    def test_sparse_online_success_keeps_hire_for_failures(self):
        """Success is sparse while a failed episode retains HiRE shaping."""

        class FakeHireShaper:
            @staticmethod
            def is_ready():
                return True

            @staticmethod
            def shape_rewards(
                rewards, images, horizon, adaptive_decay_enabled=True
            ):
                return np.full(len(rewards) - horizon, 7.0, dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            expert_path = tmp_path / "expert.npz"
            self._write_expert(expert_path)
            replay = YAMReplayBuffer(
                expert_npz_path=str(expert_path),
                online_data_dir=str(tmp_path / "online"),
                obs_horizon=2,
                action_dim=7,
                action_horizon=2,
                device="cpu",
                hire_shaper=FakeHireShaper(),
                use_sparse_for_online_success=True,
            )

            def episode(success):
                states = np.zeros((3, 7), dtype=np.float32)
                rewards = np.array([0.0, 0.0, float(success)], dtype=np.float32)
                return {
                    "states": states,
                    "actions": states.copy(),
                    "images": np.zeros((3, 6, 2, 2), dtype=np.uint8),
                    "rewards": rewards,
                    "dones": np.array([False, False, True]),
                }

            replay.add_episode(episode(success=True))
            replay._online = type(replay._online)([(0, 0)], maxlen=replay._max_online)
            success_batch = replay._sample_online(1, torch.device("cpu"))
            self.assertEqual(success_batch["reward"].item(), 1.0)

            replay.add_episode(episode(success=False))
            replay._online = type(replay._online)([(1, 0)], maxlen=replay._max_online)
            failure_batch = replay._sample_online(1, torch.device("cpu"))
            self.assertEqual(failure_batch["reward"].item(), 7.0)

    def test_hire_decay_starts_after_warmup_episode_count(self):
        class TrackingHireShaper:
            adaptive_success_rate_ema = 0.0

            def __init__(self):
                self.shape_decay_flags = []
                self.outcome_decay_flags = []

            @staticmethod
            def is_ready():
                return True

            def shape_rewards(
                self, rewards, images, horizon, adaptive_decay_enabled=True
            ):
                self.shape_decay_flags.append(adaptive_decay_enabled)
                return np.zeros(len(rewards) - horizon, dtype=np.float32)

            @staticmethod
            def current_adaptive_dense_weight(decay_enabled=True):
                return 0.5 if decay_enabled else 1.0

            def observe_episode_outcome(self, success, decay_enabled):
                self.outcome_decay_flags.append(decay_enabled)

        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            expert_path = tmp_path / "expert.npz"
            self._write_expert(expert_path)
            shaper = TrackingHireShaper()
            replay = YAMReplayBuffer(
                expert_npz_path=str(expert_path),
                online_data_dir=str(tmp_path / "online"),
                obs_horizon=2,
                action_dim=7,
                action_horizon=2,
                device="cpu",
                hire_shaper=shaper,
                hire_pbrs_decay_start_episode=2,
            )

            episode = {
                "states": np.zeros((3, 7), dtype=np.float32),
                "actions": np.zeros((3, 7), dtype=np.float32),
                "images": np.zeros((3, 6, 2, 2), dtype=np.uint8),
                "rewards": np.zeros(3, dtype=np.float32),
                "dones": np.array([False, False, True]),
            }
            replay.add_episode(episode)
            replay.add_episode(episode)
            replay.add_episode(episode)

            self.assertEqual(shaper.shape_decay_flags, [False, False, True])
            self.assertEqual(shaper.outcome_decay_flags, [False, False, True])


if __name__ == "__main__":
    unittest.main()
