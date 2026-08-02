import numpy as np

from diffusion_policy.dataset.yam_npz_dataset import YAMNpzDataset


def test_yam_npz_dataset_keeps_terminal_observations_with_padded_actions(tmp_path):
    path = tmp_path / "toy.npz"
    states = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    actions = (100 + np.arange(5 * 7, dtype=np.float32)).reshape(5, 7)
    images = np.zeros((5, 6, 4, 4), dtype=np.uint8)
    traj_lengths = np.array([3, 2], dtype=np.int64)
    np.savez_compressed(
        path,
        states=states,
        actions=actions,
        images=images,
        traj_lengths=traj_lengths,
    )

    ds = YAMNpzDataset(
        str(path),
        obs_horizon=2,
        action_horizon=3,
        image_size=4,
        val_ratio=0.5,
        seed=0,
    )
    val = ds.get_validation_dataset()

    all_indices = sorted(ds._train_indices.tolist() + val._val_indices.tolist())
    assert all_indices == [0, 1, 2, 3, 4]

    terminal_owner = ds if 2 in ds._train_indices else val
    terminal_pos = np.where(
        (terminal_owner._train_indices if terminal_owner is ds else terminal_owner._val_indices)
        == 2
    )[0][0]
    sample = terminal_owner[terminal_pos]
    action = sample["action"]["sparse"].numpy()

    assert action.shape == (3, 7)
    np.testing.assert_array_equal(action[0], actions[2])
    np.testing.assert_array_equal(action[1], actions[2])
    np.testing.assert_array_equal(action[2], actions[2])
