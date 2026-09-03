from __future__ import annotations

import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

try:
    import oopsie_data_tools  # noqa: F401
except ImportError:  # pragma: no cover
    oopsie_data_tools = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

if oopsie_data_tools is not None:
    from convert_yam_npz_to_oopsie import (
        convert_one,
        discover_episodes,
        load_source_episode,
        validate_profile,
    )
    from oopsie_data_tools.utils.conversion_utils import write_episode_annotations
    from oopsie_data_tools.utils.robot_profile.robot_profile import RobotProfile


@unittest.skipIf(oopsie_data_tools is None, "oopsie-data-tools optional dependency not installed")
class TestOopsieYamConversion(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.profile = RobotProfile(
            policy_name="synthetic_test_policy",
            robot_name="synthetic_yam",
            gripper_name="synthetic_gripper",
            is_biarm=False,
            uses_mobile_base=False,
            control_freq=30,
            camera_names=["wrist", "base"],
            robot_state_keys=["joint_position", "gripper_position"],
            robot_state_joint_names=[f"joint_{i}" for i in range(1, 7)],
            action_space=["joint_position", "gripper_position"],
            action_joint_names=[f"joint_{i}" for i in range(1, 7)],
            controller="joint_position",
        )
        self.normalization = {
            "obs_min": np.arange(7, dtype=np.float64),
            "obs_max": np.arange(7, dtype=np.float64) + 2.0,
            "action_min": np.arange(7, dtype=np.float64) + 10.0,
            "action_max": np.arange(7, dtype=np.float64) + 12.0,
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_episode(self, name: str = "episode_0001.npz", order: str = "wrist_base") -> Path:
        path = self.root / name
        images = np.zeros((30, 6, 224, 224), dtype=np.uint8)
        images[:, :3] = 25
        images[:, 3:] = 200
        np.savez_compressed(
            path,
            images=images,
            states=np.zeros((30, 7), dtype=np.float32),
            actions=np.zeros((30, 7), dtype=np.float32),
            rewards=np.zeros(30, dtype=np.float32),
            dones=np.zeros(30, dtype=bool),
            policy_camera_order=np.asarray(order),
        )
        return path

    def test_discovery_excludes_sidecars(self) -> None:
        episode = self.make_episode()
        np.savez(self.root / "episode_0001.robometer_rewards.npz", rewards=[1.0])
        hidden = self.root / ".reward_cache"
        hidden.mkdir()
        np.savez(hidden / "episode_0002.npz", rewards=[1.0])
        self.assertEqual(discover_episodes(self.root, recursive=True), [episode])

    def test_denormalizes_and_resolves_camera_order(self) -> None:
        episode = load_source_episode(
            self.make_episode(order="wrist_base"), self.normalization, 30
        )
        np.testing.assert_allclose(
            episode["states"][0], self.normalization["obs_min"] + 1.0000005
        )
        np.testing.assert_allclose(
            episode["actions"][0], self.normalization["action_min"] + 1.0000005
        )
        self.assertEqual(int(np.asarray(episode["wrist"])[0, 0, 0, 0]), 25)
        self.assertEqual(int(np.asarray(episode["base"])[0, 0, 0, 0]), 200)

    def test_writes_h5_and_two_browser_compatible_videos(self) -> None:
        validate_profile(self.profile)
        source = self.make_episode()
        output = self.root / "out" / "synthetic_episode.h5"
        output.parent.mkdir()
        convert_one(
            source,
            output,
            self.profile,
            self.normalization,
            "synthetic_test_lab",
            "synthetic_test_operator",
            "Move the synthetic test object",
            dry_run=False,
        )
        self.assertTrue(output.is_file())
        with h5py.File(output, "r") as handle:
            self.assertEqual(handle.attrs["schema"], "oopsiedata_format_v1")
            self.assertEqual(handle["observations/robot_states/joint_position"].shape, (30, 6))
            self.assertEqual(handle["observations/robot_states/gripper_position"].shape, (30, 1))
            self.assertEqual(handle["actions/joint_position"].shape, (30, 6))
            self.assertNotIn("episode_annotations", handle)
            for camera in ("wrist", "base"):
                relative = handle[f"observations/video_paths/{camera}"][()].decode()
                self.assertTrue((output.parent / relative).is_file())

        # The converter intentionally leaves the file unannotated. Simulate the UI
        # writing a human result, then exercise the public validator end to end.
        with h5py.File(output, "r+") as handle:
            write_episode_annotations(
                handle,
                annotator_name="synthetic_test_annotator",
                success=1.0,
                outcome="success",
            )
        command = Path(sys.executable).with_name("oopsie-data")
        validated = subprocess.run(
            [str(command), "validate", "--path", str(output), "--json"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)


if __name__ == "__main__":
    unittest.main()
