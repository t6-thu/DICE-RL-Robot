#!/usr/bin/env python3
"""Convert existing DICE/YAM ``episode_*.npz`` files for Oopsie annotation.

The source is never modified. Each source episode becomes one Oopsie HDF5 file
plus separate wrist/base MP4 files. Existing reward labels are deliberately not
copied: the web UI is the source of the new human annotations.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    from oopsie_data_tools.annotation_tool.episode_recorder import write_mp4
    from oopsie_data_tools.utils.contributor_config import read_contributor_config
    from oopsie_data_tools.utils.conversion_utils import (
        resize_frames,
        write_actions,
        write_robot_states,
        write_root_attrs,
        write_video_paths,
    )
    from oopsie_data_tools.utils.robot_profile.robot_profile import (
        RobotProfile,
        load_robot_profile,
    )
except ImportError as exc:  # pragma: no cover - exercised by users without the extra
    raise SystemExit(
        "oopsie-data-tools is unavailable; run: "
        "uv pip install --python .venv/bin/python 'oopsie-data-tools==1.0.1'"
    ) from exc


EPISODE_NAME = re.compile(r"^episode_(\d+)\.npz$")
EXPECTED_STATE_KEYS = {"joint_position", "gripper_position"}
EXPECTED_ACTION_KEYS = {"joint_position", "gripper_position"}
EXPECTED_CAMERAS = {"base", "wrist"}


def discover_episodes(source: Path, recursive: bool = False) -> list[Path]:
    """Find real episodes, excluding reward-cache/sidecar NPZ files."""
    source = source.expanduser().resolve()
    if source.is_file():
        if EPISODE_NAME.fullmatch(source.name) is None:
            raise ValueError(f"Expected a file named episode_<number>.npz, got {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)

    candidates = source.rglob("episode_*.npz") if recursive else source.glob("episode_*.npz")
    episodes = [
        path
        for path in candidates
        if EPISODE_NAME.fullmatch(path.name)
        and not any(part.startswith(".") for part in path.relative_to(source).parts)
    ]
    if not episodes:
        scope = "recursively" if recursive else "at its top level"
        raise ValueError(f"No episode_<number>.npz files found {scope} under {source}")
    return sorted(episodes)


def _nonempty(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"todo", "unknown", "none", "your_lab_id"}:
        raise ValueError(f"{field} must be a confirmed, non-placeholder value")
    return text


def validate_profile(profile: RobotProfile) -> None:
    """Reject profiles that do not describe this converter's fixed NPZ layout."""
    _nonempty(profile.policy_name, "profile.policy_name")
    _nonempty(profile.robot_name, "profile.robot_name")
    _nonempty(profile.gripper_name, "profile.gripper_name")
    if profile.is_biarm or profile.uses_mobile_base:
        raise ValueError("This converter supports the existing single-arm, fixed-base YAM data")
    if set(profile.camera_names) != EXPECTED_CAMERAS or len(profile.camera_names) != 2:
        raise ValueError("profile.camera_names must contain exactly: wrist, base")
    if set(profile.robot_state_keys) != EXPECTED_STATE_KEYS:
        raise ValueError(
            "profile.robot_state_keys must contain exactly joint_position and gripper_position"
        )
    if set(profile.action_space) != EXPECTED_ACTION_KEYS:
        raise ValueError(
            "profile.action_space must contain exactly joint_position and gripper_position"
        )
    if len(profile.robot_state_joint_names) != 6:
        raise ValueError("profile.robot_state_joint_names must contain the six real joint names")
    if profile.action_joint_names is None or len(profile.action_joint_names) != 6:
        raise ValueError("profile.action_joint_names must contain the six real joint names")
    for field, names in (
        ("robot_state_joint_names", profile.robot_state_joint_names),
        ("action_joint_names", profile.action_joint_names),
    ):
        if any(not str(name).strip() or str(name).lower() in {"none", "todo"} for name in names):
            raise ValueError(f"profile.{field} contains a blank or placeholder name")
        if len(set(names)) != len(names):
            raise ValueError(f"profile.{field} contains duplicate names")
    if not np.isfinite(profile.control_freq) or profile.control_freq <= 0:
        raise ValueError("profile.control_freq must be positive")


def load_normalization(path: Path) -> dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    required = ("obs_min", "obs_max", "action_min", "action_max")
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"Normalization file is missing {missing}: {path}")
        values = {key: np.asarray(archive[key], dtype=np.float64) for key in required}
    for key, value in values.items():
        if value.shape != (7,):
            raise ValueError(f"{key} must have shape (7,), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains NaN or infinity")
    if np.any(values["obs_max"] <= values["obs_min"]):
        raise ValueError("obs_max must be greater than obs_min in every dimension")
    if np.any(values["action_max"] <= values["action_min"]):
        raise ValueError("action_max must be greater than action_min in every dimension")
    return values


def denormalize(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Inverse of the DICE ``MinMaxNorm``, including its 1e-6 range epsilon."""
    return ((values + 1.0) * 0.5 * (high - low + 1e-6) + low).astype(np.float64)


def load_source_episode(
    path: Path, normalization: dict[str, np.ndarray], control_freq: float
) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"images", "states", "actions", "policy_camera_order"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"missing arrays: {missing}")
        images = np.asarray(archive["images"])
        states = np.asarray(archive["states"], dtype=np.float64)
        actions = np.asarray(archive["actions"], dtype=np.float64)
        camera_order = str(np.asarray(archive["policy_camera_order"]).item())

    if images.ndim != 4 or images.shape[1] != 6:
        raise ValueError(f"images must have shape (T,6,H,W), got {images.shape}")
    if states.ndim != 2 or states.shape[1] != 7:
        raise ValueError(f"states must have shape (T,7), got {states.shape}")
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"actions must have shape (T,7), got {actions.shape}")
    if not (len(images) == len(states) == len(actions)):
        raise ValueError(
            f"leading dimensions disagree: images={len(images)}, states={len(states)}, "
            f"actions={len(actions)}"
        )
    duration = len(images) / float(control_freq)
    if not 1.0 <= duration <= 600.0:
        raise ValueError(f"episode duration {duration:.3f}s is outside Oopsie's [1, 600]s")
    if camera_order not in {"base_wrist", "wrist_base"}:
        raise ValueError(f"unsupported policy_camera_order={camera_order!r}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("states/actions contain NaN or infinity")
    if images.dtype != np.uint8:
        if not np.issubdtype(images.dtype, np.floating):
            raise ValueError(f"images must be uint8 or floating point, got {images.dtype}")
        images = (np.clip(images, 0.0, 1.0) * 255.0).astype(np.uint8)

    raw_states = denormalize(states, normalization["obs_min"], normalization["obs_max"])
    raw_actions = denormalize(
        actions, normalization["action_min"], normalization["action_max"]
    )
    first, second = ("base", "wrist") if camera_order == "base_wrist" else ("wrist", "base")
    cameras = {
        first: np.transpose(images[:, :3], (0, 2, 3, 1)),
        second: np.transpose(images[:, 3:], (0, 2, 3, 1)),
    }
    return {
        "states": raw_states,
        "actions": raw_actions,
        "base": cameras["base"],
        "wrist": cameras["wrist"],
        "camera_order": camera_order,
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_") or "episode"


def episode_id_for(path: Path, source: Path, prefix: str | None) -> str:
    if source.is_file():
        relative = Path(path.stem)
        base = source.parent.name
    else:
        relative = path.relative_to(source).with_suffix("")
        base = source.name
    parts = ([prefix] if prefix else [base]) + list(relative.parts)
    return _slug("_".join(parts))


def convert_one(
    source_path: Path,
    h5_path: Path,
    profile: RobotProfile,
    normalization: dict[str, np.ndarray],
    lab_id: str,
    operator_name: str,
    language_instruction: str,
    dry_run: bool,
) -> None:
    episode = load_source_episode(source_path, normalization, profile.control_freq)
    if dry_run:
        return

    videos_dir = h5_path.parent / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    tmp_h5 = h5_path.with_suffix(".tmp.h5")
    final_videos = {
        camera: videos_dir / f"{h5_path.stem}_{camera}.mp4" for camera in profile.camera_names
    }
    tmp_videos = {
        camera: path.with_name(path.stem + ".tmp.mp4") for camera, path in final_videos.items()
    }
    created: list[Path] = []
    try:
        for camera in profile.camera_names:
            write_mp4(
                tmp_videos[camera],
                resize_frames(np.asarray(episode[camera])),
                float(profile.control_freq),
            )
            created.append(tmp_videos[camera])

        states = np.asarray(episode["states"])
        actions = np.asarray(episode["actions"])
        with h5py.File(tmp_h5, "w") as handle:
            write_root_attrs(
                handle,
                episode_id=h5_path.stem,
                language_instruction=language_instruction,
                lab_id=lab_id,
                operator_name=operator_name,
                robot_profile=profile,
            )
            write_video_paths(
                handle,
                {camera: str(tmp_videos[camera]) for camera in profile.camera_names},
                tmp_h5,
            )
            write_robot_states(
                handle,
                {
                    "joint_position": states[:, :6],
                    "gripper_position": states[:, 6:7],
                },
                profile.robot_state_keys,
            )
            write_actions(
                handle,
                {
                    "joint_position": actions[:, :6],
                    "gripper_position": actions[:, 6:7],
                },
                profile.action_space,
            )
            # Deliberately omit episode_annotations: a human adds them in the UI.
        created.append(tmp_h5)

        for camera in profile.camera_names:
            os.replace(tmp_videos[camera], final_videos[camera])
        # Rewrite the now-final relative paths before publishing the HDF5 atomically.
        with h5py.File(tmp_h5, "r+") as handle:
            del handle["observations/video_paths"]
            write_video_paths(
                handle,
                {camera: str(final_videos[camera]) for camera in profile.camera_names},
                tmp_h5,
            )
        os.replace(tmp_h5, h5_path)
    except Exception:
        for path in created + list(tmp_videos.values()) + [tmp_h5]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        # Final videos can exist only if a rename succeeded before a later failure.
        for path in final_videos.values():
            if not h5_path.exists():
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--language-instruction", required=True)
    parser.add_argument("--operator-name", required=True)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include exact episode_<number>.npz files in nested eval directories",
    )
    parser.add_argument("--episode-prefix", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="Read and check inputs without writing output"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    operator_name = _nonempty(args.operator_name, "operator_name")
    instruction = _nonempty(args.language_instruction, "language_instruction")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise SystemExit("--max-episodes must be positive")

    profile = load_robot_profile(args.profile.expanduser())
    validate_profile(profile)
    normalization = load_normalization(args.normalization)
    lab_id, _token = read_contributor_config()
    lab_id = _nonempty(lab_id, "lab_id")
    episodes = discover_episodes(source, recursive=args.recursive)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    written = 0
    for index, episode_path in enumerate(episodes, start=1):
        episode_id = episode_id_for(episode_path, source, args.episode_prefix)
        h5_path = output_dir / f"{episode_id}.h5"
        if h5_path.exists():
            print(f"SKIP {episode_path}: output exists: {h5_path}")
            continue
        try:
            convert_one(
                episode_path,
                h5_path,
                profile,
                normalization,
                lab_id,
                operator_name,
                instruction,
                args.dry_run,
            )
            verb = "CHECK" if args.dry_run else "WRITE"
            print(f"{verb} [{index}/{len(episodes)}] {episode_path} -> {h5_path}")
            written += 1
        except Exception as exc:
            failures += 1
            print(f"ERROR [{index}/{len(episodes)}] {episode_path}: {exc}", file=sys.stderr)

    print(
        f"Summary: checked={len(episodes)} successful={written} failures={failures} "
        f"mode={'dry-run' if args.dry_run else 'write'}"
    )
    if not args.dry_run and written:
        free = shutil.disk_usage(output_dir).free / (1024**3)
        print(f"Free space after conversion: {free:.1f} GiB")
        print("Next: run the Oopsie annotation UI; validation is expected to fail until labeling.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
