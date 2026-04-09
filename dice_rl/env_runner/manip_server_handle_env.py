import numpy as np

from dice_rl.env_runner.manip_server_env import ManipServerEnv

class ManipServerHandleEnv(ManipServerEnv):
    """
    This class is a wrapper for the ManipServerEnv class.
    It wraps the get observation function for the stow robot with handle.

    """
    def __init__(self, *args, **kwargs):
        super(ManipServerHandleEnv, self).__init__(*args, **kwargs)

    def get_observation_from_buffer(self):
        obs = super(ManipServerHandleEnv, self).get_sparse_observation_from_buffer()
        # for id in self.id_list:
        #     robot_wrench = obs[f"robot_wrench_{id}"]
        #     robot_wrench_timestamps = obs[f"robot_wrench_time_stamps_{id}"]
        #     wrench = obs[f"wrench_{id}"]
        #     wrench_timestamps = obs[f"wrench_time_stamps_{id}"]

        #     robot_wrench_id = np.searchsorted(robot_wrench_timestamps, wrench_timestamps)
        #     Nrobot = len(robot_wrench_timestamps)
        #     robot_wrench_id = np.minimum(robot_wrench_id, Nrobot - 1)
        #     wrench_net = robot_wrench[robot_wrench_id] - wrench
            
        #     obs[f"wrench_{id}"] = wrench_net
        
        return obs

    def start_saving_data_for_a_new_episode(self, episode_name = ""):
        self.server.start_listening_key_events()
        self.server.start_saving_data_for_a_new_episode(episode_name)

    def stop_saving_data(self):
        self.server.stop_saving_data()
        self.server.stop_listening_key_events()


    def get_episode_folder(self):
        return self.server.get_episode_folder()
    