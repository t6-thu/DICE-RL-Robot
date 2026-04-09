import os
import psutil
from einops import rearrange, reduce
import time
import torch
import copy
import numpy as np
from typing import Optional, Callable

from utils import common_type_conversions as common_task
from utils import umift_type_conversions as umift_task

from utils.common import dict_apply
from utils.data_plotting import plot_ts_action, plot_js_action
import utils.spatial_math as su


def printOrNot(verbose, *args):
    if verbose >= 0:
        print(*args)


# fmt: off
class ModelPredictiveControllerHybrid():
    """Class that maintains IO buffering of a MPC controller with hybrid policy.
    args:
        shape_meta: dict
        policy: torch.nn.Module
        action_to_trajectory: A function that converts waypoints to a continuous trajectory
        execution_horizon: MPC execution horizon in number of steps
        obs_queue_size: size of the observation queue. Should make sure that the queue is large
                        enough for downsampling. Should not be too large to avoid waiting for
                        too long before MPC starts.
        low_dim_obs_upsample_ratio: The real feedback frequency is usually lower than that used in training. 
            for example, training may uses 500hz data, but real feedback is 100hz. In this case, we need to
            set low_dim_obs_upsample_ratio to 5. Then if shape_meta sparse_obs_low_dim_down_sample_steps = 5,
            the MPC will downsample feedback at step = 1, instead of step = 5.

    Important internal variables:
        obs_queue: stores observations specified in shape_meta['obs']
                   Data in this queue is supposed to be at raw frequency, and need to be downsampled
                   according to 'down_sample_steps' in shape_meta.
    """
    def __init__(self,
        shape_meta,
        id_list,
        policy,
        action_to_trajectory: Callable[[np.ndarray], Callable],
        sparse_execution_horizon=10,
        dense_execution_horizon=2,
        test_sparse_action=False,
        fix_orientation=False,
        dense_execution_offset=0.0,
        rl_actor=None,
    ):
        print("[MPC] Initializing")
        self.shape_meta = shape_meta
        self.id_list = id_list

        action_type = "pose9" # "pose9" or "pose9pose9s1"
        if shape_meta['action']['shape'][0] == 9:
            action_type = "pose9"
        elif shape_meta['action']['shape'][0] == 10:
            action_type = "pose9g1"
        elif shape_meta['action']['shape'][0] == 19:
            action_type = "pose9pose9s1"
        elif shape_meta['action']['shape'][0] == 38:
            action_type = "pose9pose9s1"
        elif shape_meta["action"]["shape"][0] == 21:
            action_type = "pose9pose9s1a2"
        elif shape_meta["action"]["shape"][0] == 42:
            action_type = "pose9pose9s1a2"
        else:
            raise RuntimeError('unsupported')

        if action_type == "pose9":
            action_postprocess = common_task.action9_postprocess
        elif action_type == "pose9g1":
            action_postprocess = common_task.action10_postprocess
        elif action_type == "pose9pose9s1":
            action_postprocess = common_task.action19_postprocess
        elif action_type == "pose9pose9s1a2":
            action_postprocess = common_task.action21_postprocess
        else:
            raise RuntimeError('unsupported')
        
        if "wrench_0" in shape_meta["obs"].keys():
            self.obs_has_wrench = True
        else:
            self.obs_has_wrench = False

        self.action_type = action_type
        self.action_postprocess = action_postprocess

        self.policy = policy
        self.rl_actor = rl_actor
        self.sparse_execution_horizon_time_step = sparse_execution_horizon
        self.dense_execution_horizon_time_step = dense_execution_horizon
        self.action_to_trajectory = action_to_trajectory
        self.test_sparse_action = test_sparse_action
        self.fix_orientation = fix_orientation
        self.dense_execution_offset = dense_execution_offset

        # internal variables
        self.time_offset = None
        self.sparse_obs_data = {}
        self.sparse_obs_last_timestamps = {}
        self.horizon_start_time_step = -np.inf
        self.dense_horizon_start_time_step = -np.inf
        self.dense_action_traj = []
        self.sparse_action_traj = []
        self.SE3_WBase = None
        self.verbose_level = -1

        self.sparse_target_mats = None
        self.sparse_vt_mats = None
        self.stiffness = None

        # added just for debugging
        self.sparse_action = None

        print("[MPC] Done initializing")


    def set_time_offset(self, hardware):
        '''
        Set time offset such that timing in this controller is aligned with hardware time.
        hardware: a class with a property 'current_hardware_time_s'
        '''
        self.time_offset = hardware.current_hardware_time_s - time.perf_counter()

    def set_observation(self, obs_task):
        for key, attr in self.shape_meta['sample']['obs']['sparse'].items():
            data = obs_task[key]
            horizon = attr['horizon']
            down_sample_steps = attr['down_sample_steps']
            # sample 'horizon' number of latest obs from the queue
            assert len(data) >= (horizon-1) * down_sample_steps + 1
            self.sparse_obs_data[key] = data[-(horizon-1) * down_sample_steps - 1::down_sample_steps]

        # for id in self.id_list:
        #     self.sparse_obs_last_timestamps[f"rgb_time_stamps_{id}"] = obs_task[f"rgb_time_stamps_{id}"][-1]
        #     self.sparse_obs_last_timestamps[f"robot_time_stamps_{id}"] = obs_task[f"robot_time_stamps_{id}"][-1]
        #     self.sparse_obs_last_timestamps[f"wrench_time_stamps_{id}"] = obs_task[f"wrench_time_stamps_{id}"][-1]

    def compute_sparse_control(self, device):
        """ Run sparse model inference once. Does not output control.
        """
        process = psutil.Process(os.getpid())

        with torch.no_grad():
            s = time.time()
            obs_sample_np = {}
            obs_sample_np['sparse'], SE3_WBase = common_task.sparse_obs_to_obs_sample(
                obs_sparse=self.sparse_obs_data,
                shape_meta=self.shape_meta,
                reshape_mode='reshape',
                id_list=self.id_list,
                ignore_rgb=False,
            )
            self.SE3_WBase = SE3_WBase
            # add batch dimension
            obs_sample_np = dict_apply(obs_sample_np,
                lambda x: rearrange(x, '... -> 1 ...'))
            # convert to torch tensor
            obs_sample = dict_apply(obs_sample_np,
                lambda x: torch.from_numpy(x).to(device))
            # ==== Reset and sync GPU for clean measurement ====
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            cpu_before = process.memory_info().rss / (1024 ** 2)  # in MB
            
            start_time = time.time()
            if self.rl_actor is not None:
                print('Using residual RL policy')
                # RL finetuning path: pretrained + residual actor
                action_dim = self.shape_meta["action"]["shape"][0]
                noise = torch.randn(
                    1, self.policy.sparse_action_horizon, action_dim, device=device
                )
                result = self.policy.predict_action(
                    obs_sample, init_noise=noise, unnormalize_result=False
                )
                pretrained_normalized = result["sparse_normalized"]
                features = result["features"]
                # Actor expects (B, cond_steps, obs_dim)
                state = features.unsqueeze(1)
                residual = self.rl_actor(state, noise)
                total_normalized = pretrained_normalized + residual
                raw_action = self.policy.sparse_normalizer["action"].unnormalize(
                    total_normalized
                )[0].detach().cpu().numpy()
            else:
                print('Using pretrained policy only')
                result = self.policy.predict_action(obs_sample)
                raw_action = result['sparse'][0].detach().to('cpu').numpy()
            elapsed_time = time.time() - start_time
            # ==== Measure after inference ====
            gpu_peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
            cpu_after = process.memory_info().rss / (1024 ** 2)  # in MB

            print(f"Inference time: {elapsed_time:.6f} seconds")
            # ==== Logging ====
            print(f"Peak GPU memory usage: {gpu_peak_mem:.2f} MB")
            print(f"CPU RAM usage increase: {cpu_after - cpu_before:.2f} MB")

            # print('raw action shape:', raw_action.shape)
            # raw_action[:, 2] += 0.1 * np.arange(0.5, 0.5 + 0.01 * 31, 0.01)
            
            
            action = self.action_postprocess(raw_action, SE3_WBase, self.id_list, self.fix_orientation)
            printOrNot(self.verbose_level, 'Sparse inference latency:', time.time() - s)
            return action

    def compute_dense_control(self, time_step, device):
        # Gather dense obs data
        dense_obs_data = {}
        for key, attr in self.shape_meta['sample']['obs']['dense'].items():
            key_queue = self.obs_queue[key]
            horizon = attr['horizon']
            down_sample_steps = attr['down_sample_steps']
            assert len(key_queue) > (horizon-1) * down_sample_steps + 1
            # sample 'horizon' number of latest obs from the queue
            dense_obs_data[key] = key_queue[-(horizon-1) * down_sample_steps - 1::down_sample_steps]
            dense_obs_data[key] = np.array(dense_obs_data[key]) # list -> ndarray
            dense_obs_data[key] = rearrange(dense_obs_data[key], 't d -> 1 t d')

        with torch.no_grad():
            s = time.time()
            obs_sample_np = {}
            obs_sample_np['dense'] = dense_obs_to_obs_sample(
                    obs_dense = dense_obs_data,
                    shape_meta = self.shape_meta,
                    SE3_WBase = self.SE3_WBase,
            )
            # add batch dimension
            obs_sample_np = dict_apply(obs_sample_np,
                lambda x: rearrange(x, '... -> 1 ...'))
            # convert to torch tensor
            obs_sample = dict_apply(obs_sample_np,
                lambda x: torch.from_numpy(x).to(device))

            time_step = time_step - self.horizon_start_time_step # might be a bug?
            result = self.policy.predict_dense_action(obs_sample['dense'], time_step, unnormalize_result=True)
            if isinstance(result, torch.Tensor):
                raw_action = result.detach().to('cpu').numpy()
            else:
                raw_action = result
            action = self.action_postprocess(raw_action, self.obs_last_current)
            printOrNot(self.verbose_level, '    Dense inference latency:', time.time() - s)
            return action

    def update_control(self, time_step, device):
        """ Update control per time step.
            A ValueError is raised if no IK solution is found.
        """
        sparse_action_horizon = self.shape_meta['sample']['action']['sparse']['horizon']
        sparse_action_down_sample_steps = self.shape_meta['sample']['action']['sparse']['down_sample_steps']

        if time_step > self.horizon_start_time_step + self.sparse_execution_horizon_time_step:
            # time for a new horizon
            
            # TODO: replace this mechanism with multi-threading, remove these two variables
            time0 = time.perf_counter()
            p_timestep_s = 0.01 # 0.002

            if self.action_type == "pose9":
                self.sparse_action = self.compute_sparse_control(device)
            elif self.action_type == "pose9pose9s1":
                self.sparse_target_mats, self.sparse_vt_mats, self.stiffness = self.compute_sparse_control(device)
            else:
                raise RuntimeError('unsupported')
            
            if self.test_sparse_action:
                sparse_action_timesteps = np.arange(0, sparse_action_horizon) * sparse_action_down_sample_steps
                if self.action_type == "pose9":
                    self.sparse_action_traj = self.action_to_trajectory(action_mats = self.sparse_action, time_steps = sparse_action_timesteps)
                elif self.action_type == "pose9pose9s1":
                    self.sparse_action_traj = self.action_to_trajectory(target_mats = self.sparse_target_mats,
                                                                        vt_mats = self.sparse_target_mats, # debug self.sparse_vt_mats,
                                                                        stiffness = self.stiffness,
                                                                        time_steps = sparse_action_timesteps)

            time1 = time.perf_counter()
            
            self.horizon_start_time_step = time_step + (time1 - time0 - 0.012) / p_timestep_s

            print("[MPC debug] new horizon start time step: ", self.horizon_start_time_step, ", computation takes: ", time1 - time0, "s")
            # print("[MPC debug] vt mats: ")
            # for vt in self.sparse_vt_mats:
            #     print(vt)
            # print("[MPC debug] time_steps: ", sparse_action_timesteps)
            return None # indicates a new horizon has started

            # # debug sparse action
            # timesteps_sparse_local = np.arange(0, sparse_action_horizon) * sparse_action_down_sample_steps
            # plot_ts_action(timesteps_sparse_local,
            #                su.SE3_to_pose9(self.sparse_action), title='sparse action in mpc')
            # print('press Enter to continue')
            # input()

        if self.test_sparse_action:
            time_now = time_step - self.horizon_start_time_step
            print("[MPC debug] time_step in horizon: ", time_now)
            time_now = max(time_now, 0)
            target_now = self.sparse_action_traj(time_now)
            if (np.linalg.norm(target_now[1].reshape([4, 4])[:3, 3]) < 1e-3):
                print('Warning: sparse action is zero at time_now:', time_now)
            return target_now

        dense_action_horizon = self.shape_meta['sample']['action']['dense']['horizon']
        dense_action_down_sample_steps = self.shape_meta['sample']['action']['dense']['down_sample_steps']
        if time_step > self.dense_horizon_start_time_step + self.dense_execution_horizon_time_step - 1e-6:
            # time for a new dense horizon
            dense_action = self.compute_dense_control(time_step, device)
            dense_action = rearrange(dense_action, '1 t d1 d2 -> t d1 d2')
            dense_action_timesteps = np.arange(0, dense_action_horizon+1) * dense_action_down_sample_steps
            self.dense_action_traj = self.action_to_trajectory(action_mats = dense_action, time_steps = dense_action_timesteps)
            self.dense_horizon_start_time_step = time_step

            # # debug dense/sparse action
            # timesteps_sparse_local = np.arange(0, sparse_action_horizon) * sparse_action_down_sample_steps
            # timesteps_dense_local = np.arange(0, dense_action_horizon+1) * dense_action_down_sample_steps
            # timesteps_dense_offset = time_step - self.horizon_start_time_step
            # plot_ts_action(timesteps_sparse_local,
            #                su.SE3_to_pose9(self.sparse_action),
            #                timesteps_dense_local+timesteps_dense_offset,
            #                su.SE3_to_pose9(dense_action), title='action in mpc')
            # print('press Enter to continue')
            # input()

            # # debug dense/sparse traj
            # # need to change code above to compute sparse_action_traj no matter what test_sparse_action is
            # times_sparse_local = np.arange(0, self.sparse_execution_horizon_time_step, 2)
            # times_sparse = self.horizon_start_time_step + times_sparse_local
            # joints_sparse = np.array([self.sparse_action_traj(t) for t in times_sparse])
            # times_dense_local = np.arange(0, self.dense_execution_horizon_time_step, 1)
            # times_dense = self.dense_horizon_start_time_step + times_dense_local
            # joints_dense = np.array([self.dense_action_traj(t) for t in times_dense_local])
            # plot_js_action(times_sparse, joints_sparse, times_dense, joints_dense, title='traj in mpc')
            # print('press Enter to continue')
            # input()

        return self.dense_action_traj(time_step + self.dense_execution_offset - self.dense_horizon_start_time_step)
    
    def get_SE3_targets(self):
        return self.sparse_target_mats, self.sparse_vt_mats

# fmt: on

class ModelPredictiveController():
    """Class that maintains IO buffering of a MPC controller.
    args:
        shape_meta: dict
        id_list: list of robot ids. [0] or [0, 1]
        policy: torch.nn.Module
        action_to_trajectory: A function that converts waypoints to a continuous trajectory
        execution_horizon: MPC execution horizon in number of steps
        fix_orientation: whether to only execute xyz in the action
    """
    def __init__(self,
        shape_meta,
        id_list,
        policy,
        action_to_trajectory: Callable[[np.ndarray], Callable],
        execution_horizon=10,
        fix_orientation=False,
    ):
        print("[MPC] Initializing")
        self.shape_meta = shape_meta
        self.id_list = id_list

        action_type = "pose9" # "pose9" or "pose9pose9s1"
        if shape_meta['action']['shape'][0] == 9:
            action_type = "pose9"
        elif shape_meta['action']['shape'][0] == 19:
            action_type = "pose9pose9s1"
        elif shape_meta['action']['shape'][0] == 38:
            action_type = "pose9pose9s1"
        elif shape_meta['action']['shape'][0] == 21:
            action_type = "pose9pose9s1a2"
        elif shape_meta['action']['shape'][0] == 42:
            action_type = "pose9pose9s1a2"
        else:
            raise RuntimeError('unsupported')

        if action_type == "pose9":
            action_postprocess = umift_task.action9_postprocess
        elif action_type == "pose9pose9s1":
            action_postprocess = umift_task.action19_postprocess
        elif action_type == "pose9pose9s1a2":
            action_postprocess = umift_task.action21_postprocess
        else:
            raise RuntimeError('unsupported')
        self.action_type = action_type
        self.action_postprocess = action_postprocess

        self.policy = policy
        self.execution_horizon_time_step = execution_horizon
        self.action_to_trajectory = action_to_trajectory
        self.fix_orientation = fix_orientation

        # internal variables
        self.time_offset = None
        self.obs_data = {}
        self.obs_last_timestamps = {}
        self.SE3_WBase = None
        self.verbose_level = -1

        print("[MPC] Done initializing")


    def set_time_offset(self, hardware):
        '''
        Set time offset such that timing in this controller is aligned with hardware time.
        hardware: a class with a property 'current_hardware_time_s'
        '''
        self.time_offset = hardware.current_hardware_time_s - time.perf_counter()

    def compute_one_horizon_action(self, obs_task, device):
        # sample the data per down_sample_steps and horizon
        obs_sampled = {}
        for key, attr in self.shape_meta['sample']['obs']['sparse'].items():
            data = obs_task[key]
            horizon = attr['horizon']
            down_sample_steps = attr['down_sample_steps']
            # sample 'horizon' number of latest obs from the queue
            assert len(data) >= (horizon-1) * down_sample_steps + 1
            obs_sampled[key] = data[-(horizon-1) * down_sample_steps - 1::down_sample_steps]

        # # report latency
        # time_now = time.perf_counter() + self.time_offset
        # for id in self.id_list:
        #     dt_rgb = time_now - obs_task[f"rgb_time_stamps_{id}"][-1]
        #     dt_ts_pose = time_now - obs_task[f"robot_time_stamps_{id}"][-1]
        #     dt_wrench = time_now - obs_task[f"wrench_time_stamps_{id}"][-1]
        #     dt_gripper = time_now - obs_task[f"gripper_time_stamps_{id}"][-1]
            # print(f'[MPC] obs lagging for robot {id}: dt_rgb: {dt_rgb}, dt_ts_pose: {dt_ts_pose}, dt_wrench: {dt_wrench}, dt_gripper: {dt_gripper}')

        # run inference
        with torch.no_grad():
            s = time.time()
            obs_sample_np = {}
            obs_sample_np['sparse'], SE3_WBase = umift_task.sparse_obs_to_obs_sample(
                obs_sparse=obs_sampled,
                shape_meta=self.shape_meta,
                reshape_mode='reshape',
                id_list=self.id_list,
                ignore_rgb=False,
            )
            self.SE3_WBase = SE3_WBase
            # add batch dimension
            obs_sample_np = dict_apply(obs_sample_np,
                lambda x: rearrange(x, '... -> 1 ...'))
            # convert to torch tensor
            obs_sample = dict_apply(obs_sample_np,
                lambda x: torch.from_numpy(x).to(device))

            result = self.policy.predict_action(obs_sample)
            raw_action = result['sparse'][0].detach().to('cpu').numpy()
            
            action = self.action_postprocess(raw_action, SE3_WBase, self.id_list, self.fix_orientation)
            printOrNot(self.verbose_level, 'Sparse inference latency:', time.time() - s)
            return action
    