import numpy as np
import torch

from dice_rl.reward.hire_shaper import HireRewardShaper


class _FakeEncoder:
    def __init__(self):
        self.device = torch.device("cpu")
        self.batch_sizes = []

    @staticmethod
    def features(images_b3hw: torch.Tensor) -> torch.Tensor:
        # Two deterministic patch tokens with D=3.
        mean = images_b3hw.float().mean(dim=(-2, -1))
        return torch.stack((mean, mean * 0.5 + 0.1), dim=1)

    def encode(self, images_b3hw: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(len(images_b3hw))
        return self.features(images_b3hw)


def test_compute_potential_batches_episode_without_changing_scores():
    encoder = _FakeEncoder()
    shaper = HireRewardShaper(
        encoder=encoder,
        cameras=("rgb_0", "rgb_1"),
        sample_K=8,
        online_pos_ratio=1.0,
        encode_batch_size=3,
        reward_weight=0.7,
        contrastive_lambda=0.2,
    )

    rng = np.random.default_rng(7)
    images = rng.integers(0, 256, size=(7, 6, 8, 8), dtype=np.uint8)
    for cam in shaper.cameras:
        shaper.pos_buffer_online[cam] = torch.rand(2, 2, 3)
        shaper.neg_buffer[cam] = torch.rand(2, 2, 3)

    actual = shaper._compute_potential(images)

    images_f = torch.from_numpy(images.astype(np.float32) / 255.0)
    expected_sim = torch.zeros(len(images))
    for cam, channels in (("rgb_0", slice(0, 3)), ("rgb_1", slice(3, 6))):
        current = encoder.features(images_f[:, channels])
        positive = shaper.pos_buffer_online[cam]
        negative = shaper.neg_buffer[cam]
        expected_sim += shaper._sim_to_targets(
            current, positive, beta=shaper.logsumexp_beta_pos
        )
        expected_sim -= shaper.contrastive_lambda * shaper._sim_to_targets(
            current, negative, beta=shaper.logsumexp_beta_neg
        )
    expected = (shaper.reward_weight * expected_sim).numpy().astype(np.float32)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert encoder.batch_sizes == [3, 3, 3, 3, 1, 1]
    assert max(encoder.batch_sizes) == 3
