#!/usr/bin/env python3
"""Read-only readiness check for converting existing DICE/YAM data to Oopsie."""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
from pathlib import Path

import numpy as np

from convert_yam_npz_to_oopsie import (
    discover_episodes,
    load_normalization,
    load_source_episode,
    validate_profile,
)

from oopsie_data_tools.utils.contributor_config import read_contributor_config
from oopsie_data_tools.utils.robot_profile.robot_profile import load_robot_profile


EXPECTED_VERSION = "1.0.1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    version = importlib.metadata.version("oopsie-data-tools")
    print(f"[{'OK' if version == EXPECTED_VERSION else 'FAIL'}] oopsie-data-tools={version}")
    if version != EXPECTED_VERSION:
        failures.append(f"expected oopsie-data-tools {EXPECTED_VERSION}, found {version}")

    ffmpeg = shutil.which("ffmpeg")
    print(f"[{'OK' if ffmpeg else 'FAIL'}] ffmpeg={ffmpeg or 'not found'}")
    if not ffmpeg:
        failures.append("ffmpeg is required for MP4 encoding")

    try:
        lab_id, _token = read_contributor_config()
        print(f"[OK] contributor config: lab_id={lab_id!r}, token=<masked>")
    except Exception as exc:
        print(f"[FAIL] contributor config: {exc}")
        failures.append("run `oopsie-data init` with your registered lab ID and HF token")

    profile = None
    try:
        profile = load_robot_profile(args.profile.expanduser())
        validate_profile(profile)
        print(f"[OK] robot profile={args.profile.expanduser()}")
    except Exception as exc:
        print(f"[FAIL] robot profile: {exc}")
        failures.append("complete and verify the YAM robot profile")

    normalization = None
    try:
        normalization = load_normalization(args.normalization)
        print(f"[OK] normalization={args.normalization.expanduser()}")
    except Exception as exc:
        print(f"[FAIL] normalization: {exc}")
        failures.append("provide the matching normalization.npz")

    episodes: list[Path] = []
    try:
        episodes = discover_episodes(args.source, recursive=args.recursive)
        print(f"[OK] source episodes={len(episodes)} (exact episode_<number>.npz only)")
    except Exception as exc:
        print(f"[FAIL] source: {exc}")
        failures.append("select a directory containing DICE/YAM episodes")

    if episodes and profile is not None and normalization is not None:
        sample_indices = sorted({0, len(episodes) // 2, len(episodes) - 1})
        for index in sample_indices:
            path = episodes[index]
            try:
                episode = load_source_episode(path, normalization, profile.control_freq)
                states = np.asarray(episode["states"])
                print(
                    f"[OK] sample {path.name}: T={len(states)}, "
                    f"duration={len(states) / profile.control_freq:.2f}s, "
                    f"camera_order={episode['camera_order']}"
                )
            except Exception as exc:
                print(f"[FAIL] sample {path}: {exc}")
                failures.append(f"source sample failed: {path}")

    output_parent = args.output_dir.expanduser().resolve().parent
    existing_parent = next((p for p in [output_parent, *output_parent.parents] if p.exists()), None)
    if existing_parent is not None:
        usage = shutil.disk_usage(existing_parent)
        print(f"[INFO] output filesystem free={usage.free / (1024**3):.1f} GiB")
        if usage.free < 10 * 1024**3:
            failures.append("less than 10 GiB free on the output filesystem")

    if failures:
        print("\nNot ready:")
        for failure in dict.fromkeys(failures):
            print(f"  - {failure}")
        return 1
    print("\nReady for a one-episode dry run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
