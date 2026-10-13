"""Prepare unique, complete HiRE v4 rollouts using the existing converter."""
from pathlib import Path
import zipfile

from convert_yam_npz_to_oopsie import (
    discover_episodes, load_normalization, load_robot_profile, validate_profile,
    read_contributor_config, convert_one, episode_id_for,
)


def fingerprint(path):
    with zipfile.ZipFile(path) as archive:
        return tuple((archive.getinfo(key).CRC, archive.getinfo(key).file_size)
                     for key in ('images.npy', 'states.npy', 'actions.npy'))


def main():
    data = Path('/home/haoyi/文档/data/real_processed')
    source = data / 'yam_rl_rollouts_hanoi_hire_all_online_fixed_w1_causal_v4'
    output = data.parent / 'oopsie/hanoi_hire_v4'
    project = Path(__file__).resolve().parents[1]
    profile = load_robot_profile(project / 'robot_profiles/yam_hanoi_hire_v4.yaml')
    validate_profile(profile)
    norm = load_normalization(data / 'stack_green_hanoi_cube_224_recovered_success/normalization.npz')
    lab, _ = read_contributor_config()
    seen = {fingerprint(p) for p in discover_episodes(data / 'yam_rl_rollouts_hanoi_sparse_terminal_v1', True)}
    output.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for path in discover_episodes(source, True):
        try:
            key = fingerprint(path)
        except KeyError:
            print('INCOMPLETE', path, flush=True)
            skipped += 1
            continue
        if key in seen:
            print('DUPLICATE', path, flush=True)
            skipped += 1
            continue
        seen.add(key)
        target = output / (episode_id_for(path, source, 'yam_hire_v4') + '.h5')
        if target.exists():
            print('EXISTS', target, flush=True)
            continue
        convert_one(path, target, profile, norm, lab, 'Haoyi Niu',
                    'Move the green disk from rightmost rod to middle rod of the Towers of Hanoi', False)
        written += 1
        print('WRITE', written, target.name, flush=True)
    print('DONE', 'written=', written, 'excluded=', skipped, flush=True)


if __name__ == '__main__':
    main()
