"""
YAM (I2RT) replacement for the C++ ManipServer used by the original tabletop
workcell. Exposes the same Python surface that `dice_rl.env_runner.manip_server_env.ManipServerEnv`
exposes so that `rl_finetuning_env_runner.py` can drive a YAM arm + one
wrist-mounted RealSense.

What it provides:
  - Joint-position control of the YAM 6-DoF chain + 1 gripper motor via `i2rt`.
  - Cartesian waypoint scheduling (pose7 = [x,y,z,qw,qx,qy,qz]) via IK using
    `i2rt.robots.kinematics.Kinematics` (MuJoCo / mink).
  - One RealSense color stream sampled into a ring buffer with monotonic
    hardware timestamps.
  - Ring buffers for joint state, end-effector pose feedback, and gripper.
  - On-disk episode recording compatible with `utils.data_processing.processing_functions`
    (rgb_0/*.jpg + robot_data_0.json + eoat_data_0.json).

What it does NOT provide (intentionally):
  - External force/torque sensor. YAM has no wrist F/T sensor; wrench-related
    fields are filled with zeros so a shape_meta that omits `robot0_eef_wrench`
    works cleanly. A wrench-conditioned BC checkpoint will not run correctly
    here; train a new BC on YAM-collected data without wrench obs.
  - True Cartesian admittance control. We do joint-space PD via i2rt. The
    `stiffness_matrices_6x6` argument is accepted for API compatibility and
    ignored.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from einops import rearrange

from utils.computer_vision import get_image_transform


_INPUT_RES = (224, 224)
_QUAT_EPS = 1e-9


# ---------------------------------------------------------------------------
# Math helpers (kept local so this file does not depend on heavyweight modules
# at import time; FK results are already 4x4 SE(3))
# ---------------------------------------------------------------------------


def _SE3_to_pose7(T: np.ndarray) -> np.ndarray:
    """4x4 SE(3) -> pose7 [x,y,z,qw,qx,qy,qz]."""
    R = T[:3, :3]
    t = T[:3, 3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= max(np.linalg.norm(q), _QUAT_EPS)
    return np.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3]], dtype=np.float64)


def _pose7_to_SE3(pose7: np.ndarray) -> np.ndarray:
    """pose7 [x,y,z,qw,qx,qy,qz] -> 4x4 SE(3)."""
    x, y, z, qw, qx, qy, qz = pose7
    n = max(np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz), _QUAT_EPS)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    T = np.eye(4)
    T[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    T[0, 1] = 2 * (qx * qy - qz * qw)
    T[0, 2] = 2 * (qx * qz + qy * qw)
    T[1, 0] = 2 * (qx * qy + qz * qw)
    T[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    T[1, 2] = 2 * (qy * qz - qx * qw)
    T[2, 0] = 2 * (qx * qz - qy * qw)
    T[2, 1] = 2 * (qy * qz + qx * qw)
    T[2, 2] = 1 - 2 * (qx * qx + qy * qy)
    T[:3, 3] = (x, y, z)
    return T


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------


class _RingBuffer:
    """Fixed-capacity ring buffer of numpy rows + matching float timestamps.

    Single-writer / multi-reader. Reader takes a snapshot under the lock.
    """

    def __init__(self, capacity: int, row_shape: Tuple[int, ...], dtype=np.float64):
        self.capacity = int(capacity)
        self._data = np.zeros((self.capacity, *row_shape), dtype=dtype)
        self._ts_ms = np.zeros((self.capacity,), dtype=np.float64)
        self._n = 0  # total writes (clipped to capacity for reads)
        self._head = 0  # next write index
        self._lock = threading.Lock()

    def push(self, row: np.ndarray, ts_ms: float) -> None:
        with self._lock:
            self._data[self._head] = row
            self._ts_ms[self._head] = ts_ms
            self._head = (self._head + 1) % self.capacity
            if self._n < self.capacity:
                self._n += 1

    def last(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the most recent n rows in chronological order.

        If fewer than n have been written, repeats the oldest sample to pad.
        """
        with self._lock:
            n_avail = self._n
            if n_avail == 0:
                return (
                    np.zeros((n, *self._data.shape[1:]), dtype=self._data.dtype),
                    np.zeros((n,), dtype=np.float64),
                )
            n = int(n)
            if n_avail >= self.capacity:
                # Buffer fully wrapped: ordered = roll so head is end.
                ordered = np.roll(self._data, -self._head, axis=0)
                ordered_ts = np.roll(self._ts_ms, -self._head)
            else:
                ordered = self._data[:n_avail].copy()
                ordered_ts = self._ts_ms[:n_avail].copy()
            if n <= ordered.shape[0]:
                return ordered[-n:].copy(), ordered_ts[-n:].copy()
            pad = n - ordered.shape[0]
            front = np.broadcast_to(ordered[:1], (pad, *ordered.shape[1:])).copy()
            front_ts = np.full((pad,), ordered_ts[0], dtype=np.float64)
            return (
                np.concatenate([front, ordered], axis=0),
                np.concatenate([front_ts, ordered_ts], axis=0),
            )


# ---------------------------------------------------------------------------
# Episode recorder (writes the on-disk schema expected by processing_functions)
# ---------------------------------------------------------------------------


class _EpisodeRecorder:
    """Captures the live streams to disk while an episode is being executed.

    File layout under <root>/episode_<unix_ms>/:
        rgb_0/frame_NNNNN<ts_ms11d>.jpg   (BGR JPEG, char[11:22] = ts_ms)
        robot_data_0.json                 {robot_time_stamps, ts_pose_fb, robot_wrench, mask}
        eoat_data_0.json                  {eoat_time_stamps, eoat_pos_fb}
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False
        self._folder: Optional[str] = None
        self._rgb_dir: Optional[str] = None
        self._frame_idx = 0
        self._robot_rows: List[Dict[str, Any]] = []
        self._eoat_rows: List[Dict[str, Any]] = []

    def start(self, raw_folder: str) -> str:
        os.makedirs(raw_folder, exist_ok=True)
        episode_name = f"episode_{int(time.time())}"
        folder = os.path.join(raw_folder, episode_name)
        rgb_dir = os.path.join(folder, "rgb_0")
        os.makedirs(rgb_dir, exist_ok=True)
        with self._lock:
            self._folder = folder
            self._rgb_dir = rgb_dir
            self._frame_idx = 0
            self._robot_rows.clear()
            self._eoat_rows.clear()
            self._active = True
        return folder

    def stop(self) -> Optional[str]:
        with self._lock:
            if not self._active:
                return None
            self._active = False
            folder = self._folder
            robot_rows = list(self._robot_rows)
            eoat_rows = list(self._eoat_rows)

        if folder is None:
            return None
        with open(os.path.join(folder, "robot_data_0.json"), "w") as f:
            json.dump(_rows_to_columns(robot_rows), f)
        with open(os.path.join(folder, "eoat_data_0.json"), "w") as f:
            json.dump(_rows_to_columns(eoat_rows), f)
        return folder

    def write_frame(self, bgr: np.ndarray, ts_ms: float) -> None:
        with self._lock:
            if not self._active or self._rgb_dir is None:
                return
            idx = self._frame_idx
            self._frame_idx += 1
        # 11-char prefix + 11-digit ms timestamp -> processing_functions parses
        # ts via name[11:22].
        ts_int = max(0, int(round(ts_ms)))
        fname = f"frame_{idx:05d}{ts_int:011d}.jpg"
        cv2.imwrite(os.path.join(self._rgb_dir, fname), bgr)

    def write_robot(self, ts_ms: float, pose7: np.ndarray, robot_wrench6: np.ndarray) -> None:
        with self._lock:
            if not self._active:
                return
            self._robot_rows.append(
                {
                    "robot_time_stamps": float(ts_ms),
                    "ts_pose_fb": pose7.tolist(),
                    "robot_wrench": robot_wrench6.tolist(),
                    "mask": [1] * 6,
                }
            )

    def write_eoat(self, ts_ms: float, eoat: float) -> None:
        with self._lock:
            if not self._active:
                return
            self._eoat_rows.append(
                {
                    "eoat_time_stamps": float(ts_ms),
                    "eoat_pos_fb": [float(eoat)],
                }
            )

    @property
    def folder(self) -> Optional[str]:
        with self._lock:
            return self._folder


def _rows_to_columns(rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    if not rows:
        return {}
    keys = list(rows[0].keys())
    return {k: [r[k] for r in rows] for k in keys}


# ---------------------------------------------------------------------------
# YAM hardware config
# ---------------------------------------------------------------------------


@dataclass
class YamHardwareConfig:
    can_channel: str = "can0"
    gripper_type: str = "yam_teaching_handle"  # i2rt GripperType name (lowercase)
    fk_site: str = "grasp_site"
    camera_serial: Optional[str] = None  # if None, first detected D4xx
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    control_hz: float = 200.0
    state_read_hz: float = 200.0
    ring_buffer_seconds: float = 5.0
    data_folder: Optional[str] = None  # overrides RLFinetuningEnvRunner data_folder_path

    @classmethod
    def from_yaml(cls, path: str) -> "YamHardwareConfig":
        with open(path, "r") as f:
            blob = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in blob.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Camera thread
# ---------------------------------------------------------------------------


class _RealSenseStreamer(threading.Thread):
    def __init__(
        self,
        cfg: YamHardwareConfig,
        rgb_buffer: _RingBuffer,
        recorder: _EpisodeRecorder,
        t0_monotonic: float,
    ) -> None:
        super().__init__(daemon=True, name="yam-rs")
        self._cfg = cfg
        self._rgb_buffer = rgb_buffer
        self._recorder = recorder
        self._t0 = t0_monotonic
        self._stop = threading.Event()
        self._pipeline = None
        self._image_transform = get_image_transform(
            input_res=(cfg.camera_height, cfg.camera_width),
            output_res=_INPUT_RES,
            bgr_to_rgb=True,
        )

    def _open(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if self._cfg.camera_serial is not None:
            config.enable_device(self._cfg.camera_serial)
        config.enable_stream(
            rs.stream.color,
            self._cfg.camera_width,
            self._cfg.camera_height,
            rs.format.bgr8,
            self._cfg.camera_fps,
        )
        pipeline.start(config)
        # Warm-up: wait for auto-exposure to settle.
        for _ in range(5):
            pipeline.wait_for_frames()
        self._pipeline = pipeline

    def run(self) -> None:
        try:
            self._open()
        except Exception as e:
            print(f"[YamServerEnv] RealSense open failed: {e}")
            return
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                bgr = np.asanyarray(color.get_data())
                ts_ms = (time.monotonic() - self._t0) * 1000.0
                # Down-sample to 224x224 RGB for the policy buffer.
                rgb_small = self._image_transform(bgr)
                self._rgb_buffer.push(rgb_small.astype(np.uint8), ts_ms)
                self._recorder.write_frame(bgr, ts_ms)
            except Exception as e:
                # Avoid tight loop on transient errors.
                print(f"[YamServerEnv] camera tick error: {e}")
                time.sleep(0.05)

    def stop(self) -> None:
        self._stop.set()
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Robot state reader + controller
# ---------------------------------------------------------------------------


class _RobotIO(threading.Thread):
    """Reads joint state at state_read_hz and FKs to pose7.

    Also runs the control loop: pops the next scheduled (q_target, t_target)
    from a queue and commands joints when t_target is reached.
    """

    def __init__(
        self,
        cfg: YamHardwareConfig,
        robot,
        kin,
        pose_buffer: _RingBuffer,
        joint_buffer: _RingBuffer,
        eoat_buffer: _RingBuffer,
        recorder: _EpisodeRecorder,
        t0_monotonic: float,
    ) -> None:
        super().__init__(daemon=True, name="yam-io")
        self._cfg = cfg
        self._robot = robot
        self._kin = kin
        self._pose_buffer = pose_buffer
        self._joint_buffer = joint_buffer
        self._eoat_buffer = eoat_buffer
        self._recorder = recorder
        self._t0 = t0_monotonic
        self._stop = threading.Event()
        self._mode = "maintain"  # "maintain" | "free" | "track"
        self._mode_lock = threading.Lock()
        self._traj_lock = threading.Lock()
        self._traj: List[Tuple[float, np.ndarray, Optional[float]]] = []  # (t_ms, q_arm, q_gripper)
        self._last_commanded_q: Optional[np.ndarray] = None  # arm-only (n_arm)

    @property
    def num_arm_dofs(self) -> int:
        return self._robot.num_dofs() - 1 if self._has_gripper else self._robot.num_dofs()

    @property
    def _has_gripper(self) -> bool:
        # We assume the robot was built with a gripper motor unless num_dofs==6.
        return self._robot.num_dofs() > 6

    def set_mode(self, mode: str) -> None:
        with self._mode_lock:
            self._mode = mode
            if mode == "free":
                try:
                    self._robot.zero_torque_mode()
                except Exception as e:
                    print(f"[YamServerEnv] zero_torque_mode failed: {e}")

    def push_trajectory(
        self, ts_ms: np.ndarray, q_arm_seq: np.ndarray, q_grip_seq: Optional[np.ndarray]
    ) -> None:
        traj = []
        for i, t in enumerate(ts_ms):
            qg = float(q_grip_seq[i]) if q_grip_seq is not None else None
            traj.append((float(t), q_arm_seq[i].copy(), qg))
        with self._traj_lock:
            self._traj = traj
        with self._mode_lock:
            self._mode = "track"

    def get_last_arm_q(self) -> np.ndarray:
        if self._last_commanded_q is not None:
            return self._last_commanded_q.copy()
        return self._robot.get_joint_pos()[: self.num_arm_dofs].copy()

    def run(self) -> None:
        dt_state = 1.0 / max(self._cfg.state_read_hz, 1.0)
        dt_ctrl = 1.0 / max(self._cfg.control_hz, 1.0)
        next_state = time.monotonic()
        next_ctrl = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            ts_ms = (now - self._t0) * 1000.0

            # --- state read ---
            if now >= next_state:
                try:
                    q = self._robot.get_joint_pos()
                except Exception as e:
                    print(f"[YamServerEnv] get_joint_pos failed: {e}")
                    q = None
                if q is not None:
                    q_arm = q[: self.num_arm_dofs]
                    T = self._kin.fk(q_arm, self._cfg.fk_site)
                    pose7 = _SE3_to_pose7(T)
                    self._pose_buffer.push(pose7, ts_ms)
                    self._joint_buffer.push(q_arm.copy(), ts_ms)
                    eoat = float(q[-1]) if self._has_gripper else 0.0
                    self._eoat_buffer.push(np.array([eoat], dtype=np.float64), ts_ms)
                    self._recorder.write_robot(ts_ms, pose7, np.zeros(6))
                    if self._has_gripper:
                        self._recorder.write_eoat(ts_ms, eoat)
                    if self._last_commanded_q is None:
                        self._last_commanded_q = q_arm.copy()
                next_state += dt_state

            # --- control tick ---
            if now >= next_ctrl:
                with self._mode_lock:
                    mode = self._mode
                if mode == "track":
                    self._step_track(ts_ms)
                elif mode == "maintain":
                    # Re-issue last commanded q so the controller doesn't drift.
                    if self._last_commanded_q is not None:
                        try:
                            cmd = np.concatenate(
                                [self._last_commanded_q, np.array([self._current_gripper()])]
                            ) if self._has_gripper else self._last_commanded_q
                            self._robot.command_joint_pos(cmd)
                        except Exception as e:
                            print(f"[YamServerEnv] maintain command failed: {e}")
                # "free" mode: nothing to do; zero_torque_mode was set at mode switch.
                next_ctrl += dt_ctrl

            sleep = min(next_state, next_ctrl) - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)

    def _current_gripper(self) -> float:
        rows, _ = self._eoat_buffer.last(1)
        return float(rows[-1, 0])

    def _step_track(self, ts_ms: float) -> None:
        with self._traj_lock:
            # Find the latest waypoint whose timestamp is <= now.
            target = None
            keep = []
            for entry in self._traj:
                if entry[0] <= ts_ms:
                    target = entry
                else:
                    keep.append(entry)
            if target is not None:
                self._traj = keep
        if target is None:
            return
        _, q_arm, q_grip = target
        cmd = q_arm.copy()
        if self._has_gripper:
            grip_target = q_grip if q_grip is not None else self._current_gripper()
            cmd = np.concatenate([cmd, np.array([grip_target])])
        try:
            self._robot.command_joint_pos(cmd)
            self._last_commanded_q = q_arm.copy()
        except Exception as e:
            print(f"[YamServerEnv] track command failed: {e}")

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Public env class (ManipServerEnv-compatible surface)
# ---------------------------------------------------------------------------


class YamServerEnv:
    """YAM-backed drop-in replacement for `ManipServerEnv`.

    Args mirror `ManipServerEnv.__init__`. `hardware_config_path` should point
    at a YAM YAML (see `configs/hardware/yam_workstation.yaml`).
    """

    def __init__(
        self,
        camera_res_hw: List[int],
        hardware_config_path: str,
        query_sizes: Dict[str, Dict],
        compliant_dimensionality: int,  # accepted for API parity, ignored
    ) -> None:
        del compliant_dimensionality
        self._cfg = YamHardwareConfig.from_yaml(hardware_config_path)
        self._t0 = time.monotonic()

        # Lazy-import i2rt so module import does not require CAN hardware.
        from i2rt.robots.get_robot import GripperType, get_yam_robot
        from i2rt.robots.kinematics import Kinematics
        from i2rt.robots.utils import YAM_XML_PATH

        gripper = GripperType[self._cfg.gripper_type.upper()]
        print(
            f"[YamServerEnv] starting i2rt on {self._cfg.can_channel} with gripper={gripper}"
        )
        self._robot = get_yam_robot(channel=self._cfg.can_channel, gripper_type=gripper)
        self._robot.start_server()
        self._kin = Kinematics(YAM_XML_PATH, self._cfg.fk_site)

        # Ring buffers sized to cover query horizons + headroom.
        rb_capacity_pose = max(
            int(self._cfg.state_read_hz * self._cfg.ring_buffer_seconds),
            query_sizes["sparse"]["ts_pose_fb"] * 4,
        )
        n_arm = self._robot.num_dofs() - (1 if self._robot.num_dofs() > 6 else 0)
        self._pose_buffer = _RingBuffer(rb_capacity_pose, (7,))
        self._joint_buffer = _RingBuffer(rb_capacity_pose, (n_arm,))
        self._eoat_buffer = _RingBuffer(rb_capacity_pose, (1,))

        rb_capacity_rgb = max(
            int(self._cfg.camera_fps * self._cfg.ring_buffer_seconds),
            query_sizes["sparse"]["rgb"] * 4,
        )
        self._rgb_buffer = _RingBuffer(rb_capacity_rgb, (_INPUT_RES[0], _INPUT_RES[1], 3), dtype=np.uint8)

        self._recorder = _EpisodeRecorder()
        self._camera = _RealSenseStreamer(self._cfg, self._rgb_buffer, self._recorder, self._t0)
        self._io = _RobotIO(
            self._cfg,
            self._robot,
            self._kin,
            self._pose_buffer,
            self._joint_buffer,
            self._eoat_buffer,
            self._recorder,
            self._t0,
        )

        self._image_transform = get_image_transform(
            input_res=_INPUT_RES, output_res=camera_res_hw, bgr_to_rgb=False
        )
        self._query_sizes = query_sizes
        self._id_list = [0]
        self._output_rgb_buffer = [
            np.zeros(
                (query_sizes["sparse"]["rgb"], camera_res_hw[0], camera_res_hw[1], 3),
                dtype=np.uint8,
            )
        ]

        # Start threads.
        self._camera.start()
        self._io.start()
        # Wait for first samples on both streams.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            rgb, _ = self._rgb_buffer.last(1)
            pose, _ = self._pose_buffer.last(1)
            if rgb.shape[0] > 0 and pose.shape[0] > 0 and self._rgb_buffer._n > 0 and self._pose_buffer._n > 0:
                break
            time.sleep(0.05)
        self._ready = self._pose_buffer._n > 0 and self._rgb_buffer._n > 0

    # ---- Properties --------------------------------------------------------

    @property
    def current_hardware_time_s(self) -> float:
        return time.monotonic() - self._t0

    @property
    def camera_ids(self) -> List[int]:
        return self._id_list

    @property
    def id_list(self) -> List[int]:
        return self._id_list

    @property
    def server(self):
        """Compat shim: rl_finetuning_env_runner reaches into env.server."""
        return self

    # The ManipServerHandleEnv expects these methods on `env.server`.
    def is_bimanual(self) -> bool:
        return False

    def has_eoat(self) -> bool:
        return self._robot.num_dofs() > 6

    def is_ready(self) -> bool:
        return self._ready

    def get_timestamp_now_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000.0

    # ---- High-level mode toggles ------------------------------------------

    def set_high_level_maintain_position(self) -> None:
        self._io.set_mode("maintain")

    def set_high_level_free_jogging(self) -> None:
        self._io.set_mode("free")

    def calibrate_robot_wrench(self, NSamples: int = 100) -> None:  # noqa: N803
        # No F/T sensor on YAM; nothing to calibrate.
        del NSamples

    # ---- Reset / cleanup --------------------------------------------------

    def reset(self) -> None:
        pass

    def cleanup(self) -> None:
        try:
            self._io.stop()
            self._camera.stop()
            self._io.join(timeout=2.0)
            self._camera.join(timeout=2.0)
        finally:
            try:
                self._robot.close()
            except Exception:
                pass

    # ---- Scheduling -------------------------------------------------------

    def schedule_controls(
        self,
        pose7_cmd: np.ndarray,  # 7xN (single arm only)
        timestamps: np.ndarray,  # ms, shape (N,)
        stiffness_matrices_6x6: Optional[np.ndarray] = None,  # ignored
        eoat_cmd: Optional[np.ndarray] = None,  # 2xN or 1xN; we use row 0
    ) -> bool:
        del stiffness_matrices_6x6
        if pose7_cmd.shape[0] != 7:
            raise ValueError(
                f"YamServerEnv expects pose7_cmd of shape (7, N); got {pose7_cmd.shape}"
            )
        if timestamps.shape[0] != pose7_cmd.shape[1]:
            raise ValueError("timestamps length must equal pose7_cmd N")
        if not self._ready:
            return False

        # IK each waypoint. Use the previous solution as init for warm-start.
        N = pose7_cmd.shape[1]
        init_q = self._io.get_last_arm_q()
        q_arm_seq = np.zeros((N, init_q.shape[0]), dtype=np.float64)
        for i in range(N):
            target_T = _pose7_to_SE3(pose7_cmd[:, i])
            ok, q = self._kin.ik(target_T, self._cfg.fk_site, init_q=init_q)
            if not ok:
                # Hold previous q rather than diverging.
                q_arm_seq[i] = init_q
            else:
                q_arm_seq[i] = q[: init_q.shape[0]]
                init_q = q_arm_seq[i]

        q_grip_seq = None
        if eoat_cmd is not None and self.has_eoat():
            q_grip_seq = np.asarray(eoat_cmd[0], dtype=np.float64).reshape(-1)
            if q_grip_seq.shape[0] != N:
                q_grip_seq = None  # malformed; ignore

        self._io.push_trajectory(np.asarray(timestamps, dtype=np.float64), q_arm_seq, q_grip_seq)
        return True

    # ---- Observation queries (match ManipServerEnv) ------------------------

    def get_sparse_observation_from_buffer(self) -> Dict[str, np.ndarray]:
        qs = self._query_sizes["sparse"]
        rgb_small, rgb_ts_ms = self._rgb_buffer.last(qs["rgb"])
        pose_rows, pose_ts_ms = self._pose_buffer.last(qs["ts_pose_fb"])
        wrench_ts_ms = np.linspace(
            max(pose_ts_ms[0] - 10.0, 0.0), pose_ts_ms[-1], qs["wrench"]
        )
        wrench = np.zeros((qs["wrench"], 6), dtype=np.float64)
        # Re-resize stored 224x224 RGBs to whatever the policy wants.
        out_rgb = self._output_rgb_buffer[0]
        for i in range(qs["rgb"]):
            out_rgb[i] = self._image_transform(rgb_small[i])

        result = {
            "rgb_0": out_rgb,
            "rgb_time_stamps_0": rgb_ts_ms / 1000.0,
            "ts_pose_fb_0": pose_rows,
            "robot_time_stamps_0": pose_ts_ms / 1000.0,
            "wrench_0": wrench,
            "wrench_time_stamps_0": wrench_ts_ms / 1000.0,
            "robot_wrench_0": np.zeros((500, 6), dtype=np.float64),
            "robot_wrench_time_stamps_0": np.linspace(
                pose_ts_ms[0] / 1000.0, pose_ts_ms[-1] / 1000.0, 500
            ),
        }
        if self.has_eoat():
            if "eoat" in qs:
                eoat_rows, _ = self._eoat_buffer.last(qs["eoat"])
                result["eoat_pos_0"] = eoat_rows
        return result

    def get_dense_observation_from_buffer(self) -> Dict[str, np.ndarray]:
        # Reuse current sparse buffers; the env_runner currently uses sparse.
        qs_dense = self._query_sizes.get("dense", self._query_sizes["sparse"])
        pose_rows, pose_ts_ms = self._pose_buffer.last(qs_dense["ts_pose_fb"])
        wrench = np.zeros((qs_dense["wrench"], 6), dtype=np.float64)
        wrench_ts_ms = np.linspace(pose_ts_ms[0], pose_ts_ms[-1], qs_dense["wrench"])
        return {
            "rgb_0": self._output_rgb_buffer[0],
            "rgb_time_stamps_0": self._rgb_buffer.last(self._query_sizes["sparse"]["rgb"])[1] / 1000.0,
            "ts_pose_fb_0": pose_rows,
            "robot_time_stamps_0": pose_ts_ms / 1000.0,
            "wrench_0": wrench,
            "wrench_time_stamps_0": wrench_ts_ms / 1000.0,
        }

    # ---- Data saving (called from ManipServerHandleEnv-style API) ----------

    def start_saving_data_for_a_new_episode(self, raw_folder: str) -> str:
        return self._recorder.start(raw_folder)

    def stop_saving_data(self) -> Optional[str]:
        return self._recorder.stop()

    def get_episode_folder(self) -> Optional[str]:
        return self._recorder.folder

    def start_listening_key_events(self) -> None:
        pass

    def stop_listening_key_events(self) -> None:
        pass
