"""YAM handle env: same role as `ManipServerHandleEnv` but backed by `YamServerEnv`.

The DICE env_runner expects this single-handle wrapper to expose:
  - get_observation_from_buffer() -> dict of obs (sparse)
  - start_saving_data_for_a_new_episode(raw_folder)
  - stop_saving_data()
  - get_episode_folder()
plus every method on the underlying ManipServerEnv (current_hardware_time_s,
schedule_controls, calibrate_robot_wrench, set_high_level_*, cleanup, ...).
"""

from __future__ import annotations

from dice_rl.env_runner.yam_server_env import YamServerEnv


class YamServerHandleEnv(YamServerEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_observation_from_buffer(self):
        return self.get_sparse_observation_from_buffer()
