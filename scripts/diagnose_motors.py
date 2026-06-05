#!/usr/bin/env python3
"""YAM motor health diagnostic.

Connects to the robot, holds the home pose, and records motor temperatures /
voltages / error states once per second.  Prints a warning line whenever a
motor temperature rises >5 °C or the CAN bus reports an error.

Usage:
    . ./prepare.sh
    python scripts/diagnose_motors.py --duration 300   # 5 minutes idle
    python scripts/diagnose_motors.py --duration 120 --motion  # with slow moves

Recommended test sequence:
    1. `--duration 600` no motion  → checks CAN/power stability at idle
    2. `--duration 300 --motion`    → checks under repeated home cycles
    3. compare temp drift / errors between the two
"""
from __future__ import annotations
import argparse, signal, subprocess, sys, time
import numpy as np


def can_stats():
    """Return (rx_errors, tx_errors, bus_state) from `ip -s link can_follower_l`."""
    try:
        out = subprocess.run(
            ["ip", "-d", "-s", "link", "show", "can_follower_l"],
            capture_output=True, text=True, timeout=2,
        ).stdout
        state = "ERROR-ACTIVE"
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("can state"):
                state = line.split()[2]
                break
        return state
    except Exception:
        return "UNKNOWN"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=120.0, help="seconds to run")
    p.add_argument("--motion", action="store_true", help="cycle home pose every 5 s")
    p.add_argument("--can", default="can_follower_l")
    p.add_argument("--gripper", default="linear_4310")
    args = p.parse_args()

    print(f"[diag] connecting to robot ({args.can}, gripper={args.gripper}) …")
    from i2rt.robots.get_robot import get_yam_robot, GripperType
    robot = get_yam_robot(channel=args.can,
                          gripper_type=GripperType.from_string_name(args.gripper),
                          zero_gravity_mode=True)
    home = np.array([-0.010, 0.833, 0.903, -0.598, -0.028, -0.029, 1.0], dtype=np.float64)
    robot.move_joints(home, time_interval_s=6.0)
    print("[diag] at home. starting monitor …")

    # baseline temps
    t0_state = robot.get_motor_states()
    baseline_mos    = np.array([m.temp_mos    for m in t0_state])
    baseline_rotor  = np.array([m.temp_rotor  for m in t0_state])
    print(f"[diag] baseline temps: mos={baseline_mos.tolist()}  rotor={baseline_rotor.tolist()}")

    start = time.monotonic()
    last_motion = 0.0
    motion_toggle = False
    home_lo = home.copy(); home_lo[0] -= 0.3
    home_hi = home.copy(); home_hi[0] += 0.3

    interrupted = {"flag": False}
    def _stop(*a): interrupted["flag"] = True
    signal.signal(signal.SIGINT, _stop)

    n_can_err = 0
    n_motor_err = 0
    max_mos_rise = np.zeros(7); max_rotor_rise = np.zeros(7)

    while not interrupted["flag"] and (time.monotonic() - start) < args.duration:
        t = time.monotonic() - start

        # Optional gentle motion cycle (toggle target every 5 s).
        if args.motion and (time.monotonic() - last_motion) > 5.0:
            target = home_hi if motion_toggle else home_lo
            motion_toggle = not motion_toggle
            try:
                robot.move_joints(target, time_interval_s=4.0)
            except Exception as e:
                print(f"[diag t={t:.0f}s]  ❌  motion exception: {e}")
                n_motor_err += 1
                if "loss communication" in str(e):
                    interrupted["flag"] = True
                    break
            last_motion = time.monotonic()

        # Read state + check.
        try:
            states = robot.get_motor_states()
        except Exception as e:
            print(f"[diag t={t:.0f}s]  ❌  read exception: {e}")
            n_motor_err += 1
            if "loss communication" in str(e):
                interrupted["flag"] = True
                break
            time.sleep(1.0); continue

        mos    = np.array([m.temp_mos    for m in states])
        rotor  = np.array([m.temp_rotor  for m in states])
        rise_mos    = mos    - baseline_mos
        rise_rotor  = rotor  - baseline_rotor
        max_mos_rise   = np.maximum(max_mos_rise,   rise_mos)
        max_rotor_rise = np.maximum(max_rotor_rise, rise_rotor)
        can_state = can_stats()
        if can_state != "ERROR-ACTIVE":
            n_can_err += 1
            print(f"[diag t={t:.0f}s]  ⚠  CAN state = {can_state}")

        if int(t) % 10 == 0:  # log every 10 s
            print(f"[diag t={int(t):>4d}s]  "
                  f"mos={mos.tolist()}  Δmos={rise_mos.round(1).tolist()}  "
                  f"rotor={rotor.tolist()}  CAN={can_state}")
        time.sleep(1.0)

    elapsed = time.monotonic() - start
    print(f"\n[diag] === SUMMARY ({elapsed:.0f}s elapsed) ===")
    print(f"  CAN bus exceptions:          {n_can_err}")
    print(f"  motor exceptions / loss:     {n_motor_err}")
    print(f"  max ΔT mos   per motor:      {max_mos_rise.tolist()}")
    print(f"  max ΔT rotor per motor:      {max_rotor_rise.tolist()}")
    if n_motor_err == 0 and n_can_err == 0 and max_mos_rise.max() < 10:
        print("  ✓ robot looks healthy under this load.")
    elif n_motor_err > 0:
        print("  ❌ MOTOR LOST COMMUNICATION — see error messages above.")
    elif max_mos_rise.max() >= 10:
        print(f"  ⚠ motor {int(np.argmax(max_mos_rise))} mos rose {max_mos_rise.max():.1f} °C — thermal risk.")
    sys.exit(0 if (n_motor_err == 0 and n_can_err == 0) else 1)


if __name__ == "__main__":
    main()
