"""Create annotation-only Oopsie copies of the bimanual lid-insertion session.

The source has no commanded actions. Null action datasets explicitly preserve
that limitation: these copies are for human annotation, not validated uploads.
"""
import json
import argparse
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np


SOURCE = Path('/home/haoyi/文档/data/real/yam_bimanual/lid insertion/session_001.hdf5')
OUTPUT = Path('/home/haoyi/文档/data/oopsie_annotation_only/lid_insertion')
CAMERAS = ['base', 'left_wrist', 'right_wrist']


def video(dataset, indices, destination, fps):
    temporary = destination.with_suffix('.partial.mp4')
    command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt',
               'bgr24', '-s', '512x384', '-r', str(fps), '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
               '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in indices:
            frame = cv2.imdecode(dataset[int(index)], cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f'Undecodable source frame {index}')
            frame = cv2.resize(frame, (512, 384), interpolation=cv2.INTER_AREA)
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f'Video encoding failed: {destination}')
        temporary.replace(destination)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error('Require 0 <= shard < shards')
    cv2.setNumThreads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'videos').mkdir(exist_ok=True)
    with h5py.File(SOURCE, 'r') as source:
        segments = source['segment_ids'][:]
        timestamps = source['timestamps'][:]
        profile = dict(policy_name='human_teleop_yam_bimanual_lid_insertion_v1',
                       robot_name='yam', gripper_name='linear_4310', is_biarm=True,
                       uses_mobile_base=False, control_freq=int(source.attrs['frequency']),
                       camera_names=CAMERAS, robot_state_keys=['joint_position', 'gripper_position'],
                       robot_state_joint_names=[f'{side}_joint{i}' for side in ['left', 'right'] for i in range(1, 7)],
                       action_space=['joint_position', 'gripper_position'],
                       controller='joint_position')
        for number, segment in enumerate(np.unique(segments), 1):
            if (number - 1) % args.shards != args.shard:
                continue
            indices = np.flatnonzero(segments == segment)
            name = f'yam_bimanual_lid_insertion_session001_segment_{int(segment):04d}'
            target = OUTPUT / f'{name}.h5'
            if target.exists():
                print(f'SKIP {number} {name}', flush=True)
                continue
            for i, camera in enumerate(CAMERAS):
                video(source[f'rgb_{i}'], indices, OUTPUT / 'videos' / f'{name}_{camera}.mp4', profile['control_freq'])
            temporary = target.with_suffix('.partial')
            with h5py.File(temporary, 'w') as out:
                out.attrs.update(schema='oopsiedata_format_v1', episode_id=name,
                                 language_instruction='Lid insertion', lab_id='Berkeley_HybridRobotics',
                                 operator_name='Haoyi Niu', robot_profile=json.dumps(profile),
                                 annotation_only=True, submission_ready=False,
                                 missing_data='Commanded actions and original Unix start timestamp were not recorded.',
                                 gripper_units='normalized actuator position, not calibrated opening width',
                                 source_path=str(SOURCE), source_segment_id=int(segment),
                                 source_recording_metadata=str(source.attrs['recording_metadata']))
                q = source['trajectory'][indices]
                out.create_dataset('observations/robot_states/joint_position', data=q[:, [0,1,2,3,4,5,7,8,9,10,11,12]])
                out.create_dataset('observations/robot_states/gripper_position', data=q[:, [6,13]])
                out.create_dataset('source_timestamps', data=timestamps[indices])
                for camera in CAMERAS:
                    out.create_dataset(f'observations/video_paths/{camera}', data=f'videos/{name}_{camera}.mp4')
                for action in profile['action_space']:
                    out.create_dataset(f'actions/{action}', data=h5py.Empty('f8'))
            temporary.replace(target)
            print(f'READY {number} {name} frames={len(indices)}', flush=True)


if __name__ == '__main__':
    main()
