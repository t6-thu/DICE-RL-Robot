#!/usr/bin/env python3
"""Quick RealSense camera inspection.

  1. Lists all connected RealSense devices with their serials.
  2. Tries to open `base` and `wrist` per the config and grabs one color frame.
  3. Saves base.jpg / wrist.jpg in cwd and prints a small terminal-friendly summary
     (image shape, mean RGB, USB speed bucket).
  4. Optionally show side-by-side via OpenCV window (--show).

Usage:
    . ./prepare.sh
    python scripts/inspect_cams.py           # just save jpgs
    python scripts/inspect_cams.py --show    # also display window (Esc to close)
"""
import argparse, sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    print("pyrealsense2 not installed in this env", file=sys.stderr)
    sys.exit(1)

from dice_rl.config.yam_rl_config import HARDWARE


def list_devices():
    ctx = rs.context()
    devs = list(ctx.query_devices())
    print(f"=== {len(devs)} RealSense device(s) detected ===")
    usb_by_serial = {}
    for d in devs:
        try:
            sn   = d.get_info(rs.camera_info.serial_number)
            name = d.get_info(rs.camera_info.name)
            usb  = d.get_info(rs.camera_info.usb_type_descriptor)
            fw   = d.get_info(rs.camera_info.firmware_version)
            usb_by_serial[sn] = usb
            print(f"  serial={sn}  {name}  USB={usb}  fw={fw}")
            if not str(usb).startswith("3"):
                print(f"    WARNING: {sn} is not on USB3; D405 color streaming may fail.")
        except Exception as e:
            print(f"  (could not read all info: {e})")
    return [d.get_info(rs.camera_info.serial_number) for d in devs], usb_by_serial


def grab_one_frame(serial: str, w: int = 640, h: int = 480, fps: int = 30):
    """Open the device, wait ~1.5 s for it to settle, grab one color frame."""
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
    profile = pipe.start(cfg)
    try:
        # USB-2 cameras can take several seconds + retries before the first
        # frame arrives. Retry up to ~12s rather than failing after one wait.
        time.sleep(1.0)
        deadline = time.monotonic() + 12.0
        last = None
        while time.monotonic() < deadline:
            try:
                frames = pipe.wait_for_frames(timeout_ms=2000)
                c = frames.get_color_frame()
                if c:
                    last = np.asanyarray(c.get_data())  # (H, W, 3) RGB uint8
            except Exception:
                time.sleep(0.3)
                continue
            # got at least one frame; pop a few more so auto-exposure settles
            if last is not None:
                for _ in range(5):
                    try:
                        frames = pipe.wait_for_frames(timeout_ms=2000)
                        c = frames.get_color_frame()
                        if c:
                            last = np.asanyarray(c.get_data())
                    except Exception:
                        break
                return last
        return last
    finally:
        try: pipe.stop()
        except Exception: pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    serials, usb_by_serial = list_devices()
    cfg_base  = HARDWARE.get("base_cam_serial") if isinstance(HARDWARE, dict) else None
    cfg_wrist = HARDWARE.get("wrist_cam_serial") if isinstance(HARDWARE, dict) else None
    if cfg_base is None or cfg_wrist is None:
        # Fall back to CAMERAS dict for older config layout
        from dice_rl.config import yam_rl_config as cfg_module
        cams = getattr(cfg_module, "CAMERAS", None)
        if cams is not None:
            cfg_base  = cams["base_cam_serial"]
            cfg_wrist = cams["wrist_cam_serial"]
    print(f"\nconfig: base_serial  = {cfg_base}   {'(detected)' if cfg_base  in serials else '(NOT detected)'}")
    print(f"config: wrist_serial = {cfg_wrist}   {'(detected)' if cfg_wrist in serials else '(NOT detected)'}")

    results = {}
    for name, sn in (("base", cfg_base), ("wrist", cfg_wrist)):
        if sn not in serials:
            print(f"\n[{name}] serial {sn} not in detected devices — skipping")
            continue
        print(f"\n[{name}] grabbing frame from serial={sn} …")
        try:
            img = grab_one_frame(sn)
        except Exception as e:
            print(f"[{name}] error: {e}")
            continue
        if img is None:
            usb = usb_by_serial.get(sn, "unknown")
            print(f"[{name}] no frame received")
            if not str(usb).startswith("3"):
                print(
                    f"[{name}] USB={usb}; move this D405 to a USB3 port/cable/hub, "
                    "then rerun this script."
                )
            continue
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out = f"{name}.jpg"
        cv2.imwrite(out, bgr)
        results[name] = bgr
        # quick stats
        means = img.mean(axis=(0, 1)).tolist()
        print(f"[{name}] saved → {os.path.abspath(out)}   shape={img.shape}  mean RGB={[round(v,1) for v in means]}")
        if (np.array(means) < 5).all():
            print(f"[{name}]  ⚠  frame is nearly black — lens cover? bad exposure?")
        elif (np.array(means) > 250).all():
            print(f"[{name}]  ⚠  frame is washed-out white — overexposed / lens uncovered to light?")

    if args.show and results:
        if len(results) == 2:
            side = np.concatenate([results["base"], results["wrist"]], axis=1)
        else:
            side = next(iter(results.values()))
        cv2.imshow("base | wrist", side)
        print("\n(press any key in the window to close)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
