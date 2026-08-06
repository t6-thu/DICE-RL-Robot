import os
import unittest
from unittest.mock import patch

from dice_rl.config.yam_env_overrides import apply_learner_env_overrides


class YAMLearnerEnvOverridesTest(unittest.TestCase):
    def test_sparse_online_success_override(self):
        with patch.dict(
            os.environ,
            {"YAM_HIRE_SPARSE_ONLINE_SUCCESS": "1"},
            clear=False,
        ):
            training = apply_learner_env_overrides(
                {"use_sparse_for_online_success": False}
            )
        self.assertTrue(training["use_sparse_for_online_success"])

    def test_sparse_online_success_override_can_be_disabled(self):
        with patch.dict(
            os.environ,
            {"YAM_HIRE_SPARSE_ONLINE_SUCCESS": "0"},
            clear=False,
        ):
            training = apply_learner_env_overrides(
                {"use_sparse_for_online_success": True}
            )
        self.assertFalse(training["use_sparse_for_online_success"])


if __name__ == "__main__":
    unittest.main()
