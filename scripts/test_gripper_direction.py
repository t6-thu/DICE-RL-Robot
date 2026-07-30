#!/usr/bin/env python3
"""Drive the linear_4310 gripper to known command values and pause between
each so you can visually identify which command maps to "open" vs "closed".

Run after the YAM is powered, CAN bus is up, and the arm has been calibrated
once today (so motors aren't in a latched-error state).

Usage:
    sudo ip link set can_follower_l up type can bitrate 1000000  # if needed

    source /home/bike/Documents/niu/DICE-RL-Robot/.venv/bin/activate
    python /home/bike/Documents/niu/DICE-RL-Robot/scripts/test_gripper_direction.py \\
        --can_channel can_follower_l --gripper_type linear_4310

Watch the gripper. The script will say "command = X.X" before each move.
Tell me which command value made it OPEN and which made it CLOSED.
"""
import argparse
import os
import sys
import time
import numpy as np

# Make i2rt importable.
_I2RT = os.path.expanduser("~/Documents/niu/i2rt")
if _I2RT not in sys.path:
    sys.path.insert(0, _I2RT)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--can_channel", default="can_follower_l")
    p.add_argument("--gripper_type", default="linear_4310",
                   choices=["linear_4310", "linear_3507", "crank_4310"])
    p.add_argument("--pause_s", type=float, default=2.5)
    args = p.parse_args()

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    gripper_type = GripperType.from_string_name(args.gripper_type)
    print(f"[gripper_test] opening YAM on {args.can_channel} ({gripper_type})")
    robot = get_yam_robot(
        channel=args.can_channel,
        gripper_type=gripper_type,
        zero_gravity_mode=True,
    )

    # Brief settle.
    time.sleep(0.5)

    # Read current joint state so we don't move the arm.
    current = robot.get_joint_pos().astype(np.float64)
    print(f"[gripper_test] current joint state: {np.round(current, 3).tolist()}")
    print("[gripper_test] arm will be held at current pose; only motor 7 (gripper) will move.")
    print()

    # Step through 5 command values in i2rt's gripper command space [0, 1].
    # i2rt's JointMapper maps command -> raw motor angle linearly. command=0
    # corresponds to gripper_limits[0]; command=1 to gripper_limits[1].
    test_commands = [0.0, 0.25, 0.5, 0.75, 1.0]

    for cmd in test_commands:
        target = current.copy()
        target[6] = cmd
        print(f"[gripper_test]  command = {cmd:.2f}   (raw target from i2rt remapper)")
        robot.command_joint_pos(target)
        time.sleep(args.pause_s)
        # Read where the motor actually settled.
        actual = robot.get_joint_pos()[6]
        print(f"[gripper_test]    observed gripper state: {actual:.4f}")
        print(f"[gripper_test]    >>> LOOK AT THE GRIPPER NOW (OPEN or CLOSED?) <<<")
        print()

    # Park at 1.0 before exit (matches i2rt default).
    target = current.copy()
    target[6] = 1.0
    robot.command_joint_pos(target)
    time.sleep(1.0)

    print("[gripper_test] done. tell me which command value matched OPEN and which matched CLOSED.")
    robot.close()


if __name__ == "__main__":
    main()
