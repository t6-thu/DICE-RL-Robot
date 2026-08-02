import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dice_rl.replay_buffer.yam_replay_buffer import YAMReplayBuffer


class YAMReplayBufferBoundaryTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
