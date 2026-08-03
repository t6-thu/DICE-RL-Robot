import numpy as np
import torch

from dice_rl.env_runner.yam_rl_env_runner import (
    _camera_ages,
    _pack_policy_images,
    _preprocess,
)
from dice_rl.reward.hire_shaper import HireRewardShaper, _as_float01_images
from dice_rl.replay_buffer.yam_replay_buffer import YAMReplayBuffer
from scripts.process_hanoi_hdf5_to_npz import _resize_short_side_and_center_crop


def test_hanoi_wrist_base_packs_wrist_as_rgb0():
    base = np.full((3, 4, 4), 10, dtype=np.float32)
    wrist = np.full((3, 4, 4), 20, dtype=np.float32)

    packed = _pack_policy_images(base, wrist, "wrist_base")

    np.testing.assert_array_equal(packed[:3], wrist)
    np.testing.assert_array_equal(packed[3:], base)


def test_live_preprocess_matches_hanoi_npz_quantization():
    yy, xx = np.mgrid[:480, :640]
    rgb = np.stack(
        [(xx % 256), (yy % 256), ((xx + yy) % 256)], axis=-1
    ).astype(np.uint8)

    # Use the actual offline converter here, rather than duplicating the live
    # crop helper, so this test catches drift between training and deployment.
    crop = _resize_short_side_and_center_crop(rgb, 256)
    expected_t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float()
    expected_t = torch.nn.functional.interpolate(
        expected_t, (224, 224), mode="bilinear", align_corners=False
    )
    expected = (
        expected_t.squeeze(0).clamp(0, 255).to(torch.uint8).numpy().astype(np.float32)
        / 255.0
    )

    actual = _preprocess(rgb)

    np.testing.assert_array_equal(actual, expected)


def test_missing_camera_timestamp_is_stale():
    base_age, wrist_age = _camera_ages(9.9, 0.0, now=10.0)
    assert abs(base_age - 0.1) < 1e-9
    assert wrist_age == float("inf")


def test_hire_uint8_online_images_match_float01_images():
    images = np.arange(2 * 6 * 4 * 4, dtype=np.uint8).reshape(2, 6, 4, 4)
    expected = images.astype(np.float32) / 255.0
    np.testing.assert_array_equal(_as_float01_images(images), expected)
    np.testing.assert_array_equal(_as_float01_images(expected), expected)


def test_online_episode_camera_order_guard_rejects_mismatch():
    replay = YAMReplayBuffer.__new__(YAMReplayBuffer)
    replay.expected_policy_camera_order = "wrist_base"
    episode = {
        "policy_camera_order": np.asarray("base_wrist"),
    }
    try:
        replay.add_episode(episode)
    except ValueError as exc:
        assert "camera order mismatch" in str(exc)
    else:
        raise AssertionError("mismatched online camera order was accepted")


def test_hire_tracks_policy_keys_without_relabeling_physical_cameras():
    class FakeEncoder:
        device = torch.device("cpu")

        def encode(self, images):
            assert float(images.min()) >= 0.0
            assert float(images.max()) <= 1.0
            return images.mean(dim=(2, 3)).unsqueeze(1)

    shaper = HireRewardShaper(
        encoder=FakeEncoder(),
        cameras=("rgb_0", "rgb_1"),
        online_success_frames="all",
    )
    images = np.zeros((2, 6, 4, 4), dtype=np.uint8)
    images[:, :3] = 32
    images[:, 3:] = 224

    shaper.add_episode_to_buffer(images, success=True)

    assert set(shaper.pos_buffer_online) == {"rgb_0", "rgb_1"}
    assert not torch.equal(
        shaper.pos_buffer_online["rgb_0"],
        shaper.pos_buffer_online["rgb_1"],
    )
