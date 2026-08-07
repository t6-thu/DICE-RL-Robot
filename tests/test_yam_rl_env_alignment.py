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


def test_hire_success_rate_decay_matches_reference_formula():
    class FakeEncoder:
        device = torch.device("cpu")

    shaper = HireRewardShaper(
        encoder=FakeEncoder(),
        gamma_pbrs=1.0,
        adaptive_dense_weight_max=0.05,
        adaptive_dense_weight_min=0.0,
        adaptive_dense_weight_alpha=1.0,
        adaptive_success_rate_ema_decay=0.95,
    )
    shaper.is_ready = lambda: True
    shaper._compute_potential = lambda images: np.array(
        [1.0, 2.0, 3.0], dtype=np.float32
    )
    sparse = np.zeros(3, dtype=np.float32)
    images = np.zeros((3, 6, 2, 2), dtype=np.uint8)

    # Warmup: fixed maximum weight; success does not update the EMA.
    warmup = shaper.shape_rewards(
        sparse, images, horizon=1, adaptive_decay_enabled=False
    )
    np.testing.assert_allclose(warmup, [0.05, -0.10], atol=1e-7)
    shaper.observe_episode_outcome(success=True, decay_enabled=False)
    assert shaper.adaptive_success_rate_ema == 0.0

    # First post-warmup success updates EMA to 0.05, so the next episode uses
    # 0.05 * (1 - 0.05) = 0.0475.
    shaper.observe_episode_outcome(success=True, decay_enabled=True)
    assert abs(shaper.adaptive_success_rate_ema - 0.05) < 1e-9
    assert abs(shaper.current_adaptive_dense_weight(True) - 0.0475) < 1e-9
    decayed = shaper.shape_rewards(
        sparse, images, horizon=1, adaptive_decay_enabled=True
    )
    np.testing.assert_allclose(decayed, [0.0475, -0.095], atol=1e-7)
