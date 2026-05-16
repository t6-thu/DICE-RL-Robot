"""
RL Finetuning Env Runner for Real Robot.

This env runner:
1. Runs a diffusion policy on the real robot
2. Collects episodes and saves raw data via ManipServer
3. Saves policy_inference.zarr with policy actions
4. Sends episode_name to learner via ZMQ
5. Receives updated policy weights from learner

Based on residual_online_learning_env_runner.py but simplified for RL finetuning.
"""

import sys
import os
import time
import shutil
import pickle

import numpy as np
import torch
import zarr

def _make_env(backend, **kwargs):
    """Construct the env-runner backend.

    backend == "manip_server": original C++ ManipServer (UR/ARX/etc).
    backend == "yam":          i2rt-backed YamServerHandleEnv (single YAM arm).
    """
    if backend == "yam":
        from dice_rl.env_runner.yam_server_handle_env import YamServerHandleEnv
        return YamServerHandleEnv(**kwargs)
    from dice_rl.env_runner.manip_server_handle_env import ManipServerHandleEnv
    return ManipServerHandleEnv(**kwargs)
from dice_rl.env_runner.env_utils import (
    ts_to_js_traj,
    pose9g1_to_traj,
    pose9pose9s1_to_traj,
    get_real_obs_resolution,
)
from utils.common_type_conversions import (
    raw_to_obs,
)
from utils import spatial_math as su
from utils.mpc import ModelPredictiveControllerHybrid
from utils.model_io import load_policy
from utils.common import GracefulKiller
from dice_rl.communication.actor_node import Actor
from dice_rl.model.distill_rl import DistilledActor


def print_env_status(env):
    """Print current environment sensor status (skips wrench when absent)."""
    obs_raw = env.get_observation_from_buffer()
    for id in env.id_list:
        if f"robot_wrench_{id}" in obs_raw:
            print(
                f"[Env Status] robot_wrench_{id} average: ",
                np.mean(obs_raw[f"robot_wrench_{id}"], axis=0),
            )
        if f"wrench_{id}" in obs_raw:
            print(
                f"[Env Status] ati_wrench_{id} average: ",
                np.mean(obs_raw[f"wrench_{id}"], axis=0),
            )


class RLFinetuningEnvRunner:
    """
    Environment runner for RL finetuning of diffusion policy on real robot.

    This class handles:
    - Loading and running the diffusion policy
    - Collecting episodes on the real robot
    - Saving raw data and policy inference outputs
    - Communicating with the learner via ZMQ
    """

    def __init__(
        self,
        # Policy config
        policy_ckpt_path: str,
        device: str = "cuda",
        # Hardware config
        hardware_config_path: str = None,
        # "manip_server" (C++ ManipServer) or "yam" (i2rt YamServerEnv).
        backend: str = "manip_server",
        # Control config
        raw_time_step_s: float = 0.002,
        slow_down_factor: float = 1.0,
        sparse_execution_horizon: int = 24,
        delay_tolerance_s: float = 0.3,
        max_duration_s: float = 30.0,
        fix_orientation: bool = False,
        # Stiffness (for compliant control)
        translational_stiffness: list = None,
        rotational_stiffness: float = 50.0,
        # Data saving
        data_folder_path: str = None,
        # ZMQ communication
        network_server_endpoint: str = "ipc:///tmp/feeds/2",
        network_weight_topic: str = "network_weights_topic",
        transitions_server_endpoint: str = "ipc:///tmp/feeds/3",
        transitions_topic: str = "transitions_topic",
        transitions_topic_expire_time_s: float = 1200.0,
        # Optional features
        send_data_to_learner: bool = True,
        receive_weights_from_learner: bool = False,
        # DDIM inference steps override (None = use checkpoint default)
        rl_num_inference_steps: int = None,
        # RL checkpoint directory for auto-loading latest weights on restart
        rl_checkpoint_dir: str = None,
        actor_hidden_dims: list = None,
        critic_hidden_dims: list = None,
        actor_activation_type: str = "GELU",
        actor_use_layernorm: bool = True,
        reset_pose: list = None,
        resume_rl: bool = False,
    ):
        """Initialize the RL finetuning env runner."""
        self.device = torch.device(device)
        self.resume_rl = resume_rl
        self.rl_checkpoint_dir = rl_checkpoint_dir
        self.actor_hidden_dims = actor_hidden_dims or [1024, 1024, 1024]
        self.critic_hidden_dims = critic_hidden_dims or [1024, 1024, 1024]
        self.actor_activation_type = actor_activation_type
        self.actor_use_layernorm = actor_use_layernorm
        self.raw_time_step_s = raw_time_step_s
        self.slow_down_factor = slow_down_factor
        self.sparse_execution_horizon = sparse_execution_horizon
        self.delay_tolerance_s = delay_tolerance_s
        self.max_duration_s = max_duration_s
        self.data_folder_path = data_folder_path
        self.send_data_to_learner = send_data_to_learner
        self.receive_weights_from_learner = receive_weights_from_learner
        if reset_pose is None:
            raise ValueError("reset_pose must be provided (pose7: x, y, z, qx, qy, qz, qw)")
        self.reset_pose = np.array(reset_pose)

        if translational_stiffness is None:
            translational_stiffness = [1500, 1500, 500]
        self.translational_stiffness = translational_stiffness
        self.rotational_stiffness = rotational_stiffness

        # Initialize ZMQ communication
        print("[Env Runner] Initializing ZMQ communication...")
        self.actor_node = Actor(
            network_server_endpoint=network_server_endpoint,
            network_weight_topic=network_weight_topic,
            transitions_server_endpoint=transitions_server_endpoint,
            transitions_topic=transitions_topic,
            transitions_topic_expire_time_s=transitions_topic_expire_time_s,
        )

        # Load policy
        print(f"[Env Runner] Loading policy from: {policy_ckpt_path}")
        self.policy, self.shape_meta, self.policy_cfg = load_policy(
            policy_ckpt_path, self.device
        )
        self.policy.eval()
        if rl_num_inference_steps is not None:
            print(f"[Env Runner] Overriding num_inference_steps: {self.policy.num_inference_steps} -> {rl_num_inference_steps}")
            self.policy.num_inference_steps = rl_num_inference_steps
        print(f"[Env Runner] Final num_inference_steps: {self.policy.num_inference_steps}")

        # Get image size
        (self.image_width, self.image_height) = get_real_obs_resolution(self.shape_meta)

        # Compute timing variables first (needed by stiffness matrix and controller)
        self.sparse_action_down_sample_steps = self.shape_meta["sample"]["action"]["sparse"][
            "down_sample_steps"
        ]
        self.sparse_action_horizon = self.shape_meta["sample"]["action"]["sparse"]["horizon"]
        self.sparse_action_timesteps_s = (
            np.arange(0, self.sparse_action_horizon)
            * self.sparse_action_down_sample_steps
            * raw_time_step_s
            * slow_down_factor
        )
        self.execution_duration_s = (
            self.sparse_execution_horizon
            * self.sparse_action_down_sample_steps
            * raw_time_step_s
            * slow_down_factor
        )

        # Compute query sizes for observation buffers
        self._setup_query_sizes()

        # Determine action type
        self._setup_action_type()

        # Set up stiffness matrix (sized for sparse_execution_horizon - what we actually send)
        self._setup_stiffness_matrix()

        # Initialize environment
        print(f"[Env Runner] Initializing environment (backend={backend})...")
        self.backend = backend
        self.env = _make_env(
            backend,
            camera_res_hw=(self.image_height, self.image_width),
            hardware_config_path=hardware_config_path,
            query_sizes=self.query_sizes,
            compliant_dimensionality=6,
        )
        self.env.reset()

        # Set up controller
        # Note: controller expects sparse_execution_horizon in raw timesteps
        print("[Env Runner] Creating controller...")
        self.controller = ModelPredictiveControllerHybrid(
            shape_meta=self.shape_meta,
            id_list=self.id_list,
            policy=self.policy,
            action_to_trajectory=self.action_to_trajectory,
            sparse_execution_horizon=self.sparse_execution_horizon
            * self.sparse_action_down_sample_steps,
            test_sparse_action=True,
            fix_orientation=fix_orientation
        )
        self.controller.set_time_offset(self.env)

        # Episode counter
        self.num_episodes = self._count_existing_episodes()
        print(f"[Env Runner] Found {self.num_episodes} existing episodes")

        print("[Env Runner] Initialization complete")

    def _setup_query_sizes(self):
        """Compute query sizes for observation buffers."""
        base_rgb_query_size = (
            self.shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["horizon"] - 1
        ) * self.shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["down_sample_steps"] + 1

        base_ts_pose_query_size = (
            self.shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["horizon"] - 1
        ) * self.shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["down_sample_steps"] + 1

        # Wrench query size - check if in shape_meta, otherwise use default
        if "robot0_eef_wrench" in self.shape_meta["sample"]["obs"]["sparse"]:
            wrench_query_size = (
                self.shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"]["horizon"] - 1
            ) * self.shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"]["down_sample_steps"] + 1
        else:
            # Default fallback: (32 - 1) * 4 + 1 = 125
            wrench_query_size = (32 - 1) * 4 + 1

        self.query_sizes = {
            "sparse": {
                "rgb": base_rgb_query_size,
                "ts_pose_fb": base_ts_pose_query_size,
                "wrench": wrench_query_size,
            }
        }

    def _setup_action_type(self):
        """Determine action type from shape_meta."""
        action_dim = self.shape_meta["action"]["shape"][0]
        self.id_list = [0]

        if action_dim == 9:
            self.action_type = "pose9"
            self.action_to_trajectory = ts_to_js_traj
        elif action_dim == 10:
            self.action_type = "pose9g1"
            self.action_to_trajectory = pose9g1_to_traj
        elif action_dim == 19:
            self.action_type = "pose9pose9s1"
            self.action_to_trajectory = pose9pose9s1_to_traj
        elif action_dim == 38:
            self.action_type = "pose9pose9s1"
            self.action_to_trajectory = pose9pose9s1_to_traj
            self.id_list = [0, 1]
        else:
            raise RuntimeError(f"Unsupported action dimension: {action_dim}")

        print(f"[Env Runner] Action type: {self.action_type}, id_list: {self.id_list}")

    def _setup_stiffness_matrix(self):
        """Set up stiffness matrix for compliant control.

        Sized for sparse_execution_horizon (the number of waypoints we actually send),
        not sparse_action_horizon (the full policy output).
        """
        stiffness_matrix = np.eye(6)
        stiffness_matrix[0, 0] = self.translational_stiffness[0]
        stiffness_matrix[1, 1] = self.translational_stiffness[1]
        stiffness_matrix[2, 2] = self.translational_stiffness[2]
        stiffness_matrix[3:, 3:] *= self.rotational_stiffness

        # Size for sparse_execution_horizon (what we send to robot)
        stiffness_matrix_all = np.zeros((6, 6 * self.sparse_execution_horizon))
        for i in range(self.sparse_execution_horizon):
            stiffness_matrix_all[:, 6 * i : 6 * i + 6] = stiffness_matrix

        self.stiffness_matrix = stiffness_matrix_all

    def _count_existing_episodes(self) -> int:
        """Count existing episodes in data folder."""
        if self.data_folder_path is None:
            return 0

        raw_folder = os.path.join(self.data_folder_path, "raw")
        if not os.path.exists(raw_folder):
            os.makedirs(raw_folder, exist_ok=True)
            return 0

        data_files = os.listdir(raw_folder)
        return len(data_files)

    def receive_and_update_weights(self) -> bool:
        """
        Check for and apply new weights from learner.

        The learner sends:
        - actor_state_dict: state dict for the RL actor
        - actor_config: dict with obs_dim, action_dim, cond_steps, horizon_steps,
                        hidden_dims, activation_type (only needed on first receive)
        - training_step: current training step

        Returns:
            True if weights were updated, False otherwise
        """
        if not self.receive_weights_from_learner:
            return False

        retrieved_data, timestamp = self.actor_node.network_weight_client.pop_data(
            topic=self.actor_node.network_weight_topic, order="latest", n=1
        )
        if not retrieved_data:
            return False

        weight_data = pickle.loads(retrieved_data[0])
        actor_state_dict = weight_data["actor_state_dict"]
        training_step = weight_data.get("training_step", -1)

        if self.controller.rl_actor is None:
            # First time: create actor from config
            actor_config = weight_data["actor_config"]
            self.controller.rl_actor = DistilledActor(
                obs_dim=actor_config["obs_dim"],
                action_dim=actor_config["action_dim"],
                cond_steps=actor_config.get("cond_steps", 1),
                horizon_steps=actor_config["horizon_steps"],
                hidden_dims=actor_config.get("hidden_dims", [1024, 1024, 1024]),
                activation_type=actor_config.get("activation_type", "GELU"),
                use_layernorm=actor_config.get("use_layernorm", True),
            ).to(self.device)
            self.controller.rl_actor.eval()
            print(f"[Env Runner] Created RL actor: {actor_config}")

        self.controller.rl_actor.load_state_dict(actor_state_dict)
        print(f"[Env Runner] Updated RL actor weights (training_step={training_step})")
        return True

    def _try_load_latest_checkpoint(self) -> bool:
        """
        Try to load the latest RL checkpoint from disk.

        This allows the env_runner to resume with the most recent RL actor
        weights after a restart, without waiting for the learner to push
        weights via ZMQ.

        Returns:
            True if checkpoint was loaded, False otherwise.
        """
        if self.rl_checkpoint_dir is None or not os.path.exists(self.rl_checkpoint_dir):
            return False

        # Find all checkpoint files
        ckpt_files = [f for f in os.listdir(self.rl_checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
        if not ckpt_files:
            print(f"[Env Runner] No RL checkpoints found in {self.rl_checkpoint_dir}")
            return False

        # Parse step numbers and find the latest
        def parse_step(fname):
            try:
                return int(fname.replace("checkpoint_", "").replace(".pt", ""))
            except ValueError:
                return -1

        latest_file = max(ckpt_files, key=parse_step)
        latest_step = parse_step(latest_file)
        ckpt_path = os.path.join(self.rl_checkpoint_dir, latest_file)

        print(f"[Env Runner] Found RL checkpoint: {latest_file} (training_step={latest_step})")

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_state_dict = checkpoint["model_state_dict"]
        training_step = checkpoint.get("training_step", latest_step)

        # Extract actor state dict (keys prefixed with "actor.")
        actor_prefix = "actor."
        actor_state_dict = {
            k[len(actor_prefix):]: v
            for k, v in model_state_dict.items()
            if k.startswith(actor_prefix)
        }

        if not actor_state_dict:
            print(f"[Env Runner] WARNING: No actor keys found in checkpoint")
            return False

        # Build actor config from pretrained policy dims + constructor params
        obs_dim = self.policy.obs_feature_dim
        action_dim = self.shape_meta["action"]["shape"][0]
        horizon_steps = self.shape_meta["sample"]["action"]["sparse"]["horizon"]
        cond_steps = self.shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["horizon"]

        actor_config = {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "cond_steps": cond_steps,
            "horizon_steps": horizon_steps,
            "hidden_dims": self.actor_hidden_dims,
            "activation_type": self.actor_activation_type,
            "use_layernorm": self.actor_use_layernorm,
        }

        # Create and load actor
        self.controller.rl_actor = DistilledActor(
            obs_dim=actor_config["obs_dim"],
            action_dim=actor_config["action_dim"],
            cond_steps=actor_config["cond_steps"],
            horizon_steps=actor_config["horizon_steps"],
            hidden_dims=actor_config["hidden_dims"],
            activation_type=actor_config["activation_type"],
            use_layernorm=actor_config["use_layernorm"],
        ).to(self.device)
        self.controller.rl_actor.load_state_dict(actor_state_dict)
        self.controller.rl_actor.eval()

        print(f"[Env Runner] Loaded RL actor from checkpoint (training_step={training_step})")
        print(f"[Env Runner] Actor config: {actor_config}")
        return True

    def run_episode(self) -> dict:
        """
        Run a single episode on the robot.

        Returns:
            Dict with episode info:
            - episode_folder: path to saved episode data
            - episode_name: name of the episode (e.g., "episode_1742230408")
            - success: whether episode was successful
            - num_steps: number of control steps
            - aborted: whether episode was aborted (delay/user)
        """
        # Start saving data in ManipServer
        raw_folder = os.path.join(self.data_folder_path, "raw")
        self.env.start_saving_data_for_a_new_episode(raw_folder)

        # Storage for policy inference outputs
        policy_targets_all = {id: [] for id in self.id_list}
        policy_grippers_all = {id: [] for id in self.id_list}
        policy_timestamps_all = []

        num_steps = 0
        horizon_count = 0
        flag_large_delay = False
        episode_initial_time_s = self.env.current_hardware_time_s

        killer = GracefulKiller()

        while not killer.kill_now:
            horizon_initial_time_s = self.env.current_hardware_time_s
            print(f"[Episode] Starting horizon {horizon_count} at {horizon_initial_time_s:.3f}")

            # Get observation
            obs_raw = self.env.get_observation_from_buffer()
            obs_task = dict()
            raw_to_obs(obs_raw, obs_task, self.shape_meta)

            # Run policy inference
            self.controller.set_observation(obs_task["obs"])
            action_output = self.controller.compute_sparse_control(self.device)

            # Process action based on type
            if self.action_type == "pose9":
                action_sparse_target_mats = action_output
                action_sparse_eoats = None
            elif self.action_type == "pose9g1":
                action_sparse_target_mats, action_sparse_eoats = action_output
            elif self.action_type == "pose9pose9s1":
                action_sparse_target_mats, action_sparse_eoats = action_output
            else:
                action_sparse_target_mats = action_output
                action_sparse_eoats = None

            # Convert to pose7 commands for saving and execution
            policy_action = {}
            for id in self.id_list:
                # Convert SE3 to pose7 for execution
                pose7_cmd = su.SE3_to_pose7(
                    action_sparse_target_mats[id].reshape([-1, 4, 4])
                )[: self.sparse_execution_horizon]
                policy_action[f"policy_pose_command_{id}"] = pose7_cmd

                # Compute timestamps for this action chunk
                timestamps_ms = (
                    obs_raw["robot_time_stamps_0"][-1] + self.sparse_action_timesteps_s
                )[: self.sparse_execution_horizon] * 1000.0
                policy_action[f"policy_time_stamps_{id}"] = timestamps_ms

                if self.action_type == "pose9g1" and action_sparse_eoats is not None:
                    policy_action[f"policy_gripper_command_{id}"] = action_sparse_eoats[id].reshape(
                        [-1, 1]
                    )[: self.sparse_execution_horizon]

                # Store for saving
                policy_targets_all[id].append(policy_action[f"policy_pose_command_{id}"])
                if f"policy_gripper_command_{id}" in policy_action:
                    policy_grippers_all[id].append(policy_action[f"policy_gripper_command_{id}"])

            policy_timestamps_all.append(policy_action["policy_time_stamps_0"])

            # Prepare action for execution
            outputs_ts_targets = self._prepare_action_for_execution(
                action_sparse_target_mats
            )
            outputs_gripper = None
            if self.action_type == "pose9g1" and action_sparse_eoats is not None:
                outputs_gripper = self._prepare_gripper_for_execution(action_sparse_eoats)

            # Check timing/delay
            flag_large_delay = self._check_delay(obs_raw)
            if flag_large_delay:
                print("[Episode] Large delay detected, terminating episode")
                break

            # Send action to robot
            action_start_time_s = obs_raw["robot_time_stamps_0"][-1]
            self.env.schedule_controls(
                pose7_cmd=outputs_ts_targets,
                eoat_cmd=outputs_gripper,
                stiffness_matrices_6x6=self.stiffness_matrix,
                timestamps=(self.sparse_action_timesteps_s[: self.sparse_execution_horizon]
                           + action_start_time_s) * 1000,
            )

            num_steps += 1
            horizon_count += 1

            # Wait for execution
            time_s = self.env.current_hardware_time_s
            sleep_duration_s = horizon_initial_time_s + self.execution_duration_s - time_s
            time.sleep(max(0, sleep_duration_s))

            # Check max duration
            if time_s - episode_initial_time_s > self.max_duration_s:
                print("[Episode] Max duration reached")
                break

        # Stop saving data
        self.env.stop_saving_data()
        self.env.set_high_level_maintain_position()

        # Get episode folder path
        episode_folder = self.env.get_episode_folder()
        episode_name = os.path.basename(episode_folder)

        # Handle aborted episodes (large delay)
        if flag_large_delay:
            print("[Episode] Deleting aborted episode due to delay")
            if os.path.exists(episode_folder):
                shutil.rmtree(episode_folder)
            return {
                "episode_folder": None,
                "episode_name": None,
                "success": False,
                "num_steps": num_steps,
                "aborted": True,
            }

        # Save policy inference data
        self._save_policy_inference(
            episode_folder,
            policy_targets_all,
            policy_grippers_all,
            policy_timestamps_all,
        )

        print(f"[Episode] Saved episode: {episode_folder}")

        return {
            "episode_folder": episode_folder,
            "episode_name": episode_name,
            "success": True,  # Will be set by user
            "num_steps": num_steps,
            "aborted": False,
        }

    def _prepare_action_for_execution(self, action_sparse_target_mats) -> np.ndarray:
        """Prepare action targets for robot execution."""
        if len(self.id_list) == 1:
            # Single arm: shape (7, N)
            targets = su.SE3_to_pose7(
                action_sparse_target_mats[0].reshape([-1, 4, 4])
            )[: self.sparse_execution_horizon]
            return targets.T
        else:
            # Dual arm: shape (14, N)
            targets = []
            for id in self.id_list:
                t = su.SE3_to_pose7(
                    action_sparse_target_mats[id].reshape([-1, 4, 4])
                )[: self.sparse_execution_horizon]
                targets.append(t)
            return np.hstack(targets).T

    def _prepare_gripper_for_execution(self, action_sparse_eoats) -> np.ndarray:
        """Prepare gripper commands for robot execution."""
        if len(self.id_list) == 1:
            return action_sparse_eoats[0].reshape([-1, 1])[: self.sparse_execution_horizon]
        else:
            grippers = []
            for id in self.id_list:
                g = action_sparse_eoats[id].reshape([-1, 1])[: self.sparse_execution_horizon]
                grippers.append(g)
            return np.hstack(grippers)

    def _check_delay(self, obs_raw) -> bool:
        """Check if observation delay exceeds tolerance.

        Wrench delay is only checked when a wrench stream is present (YAM has
        no F/T sensor and reports zero-filled wrench without timestamps).
        """
        for id in self.id_list:
            dt_rgb = self.env.current_hardware_time_s - obs_raw[f"rgb_time_stamps_{id}"][-1]
            dt_ts_pose = self.env.current_hardware_time_s - obs_raw[f"robot_time_stamps_{id}"][-1]
            wrench_key = f"wrench_time_stamps_{id}"
            dt_wrench = (
                self.env.current_hardware_time_s - obs_raw[wrench_key][-1]
                if wrench_key in obs_raw
                else 0.0
            )

            if (dt_rgb > self.delay_tolerance_s or
                dt_ts_pose > self.delay_tolerance_s or
                dt_wrench > self.delay_tolerance_s):
                print(f"[Delay] robot {id}: rgb={dt_rgb:.3f}, pose={dt_ts_pose:.3f}, wrench={dt_wrench:.3f}")
                return True
        return False

    def _save_policy_inference(
        self,
        episode_folder: str,
        policy_targets_all: dict,
        policy_grippers_all: dict,
        policy_timestamps_all: list,
    ):
        """Save policy inference data to zarr."""
        zarr_path = os.path.join(episode_folder, "policy_inference.zarr")
        policy_inference_group = zarr.group(
            store=zarr.DirectoryStore(zarr_path), overwrite=True
        )

        for id in self.id_list:
            policy_inference_group.create_dataset(
                f"ts_targets_{id}",
                data=np.array(policy_targets_all[id])
            )
            if len(policy_grippers_all[id]) > 0:
                policy_inference_group.create_dataset(
                    f"ts_grippers_{id}",
                    data=np.array(policy_grippers_all[id])
                )

        # Note: timestamps are in seconds (divide by 1000)
        policy_inference_group.create_dataset(
            "timestamps_s",
            data=np.array(policy_timestamps_all) / 1000.0
        )

    def send_episode_to_learner(self, episode_info: dict, success: bool):
        """Send episode data to learner via ZMQ and save metadata to disk."""
        episode_name = episode_info.get("episode_name")
        if episode_name is None:
            return

        # Always save metadata to disk so the learner can discover episodes
        # even if ZMQ connection is broken (e.g., after env_runner restart)
        import json
        episode_folder = episode_info.get("episode_folder")
        if episode_folder and os.path.exists(episode_folder):
            metadata = {"episode_name": episode_name, "success": success}
            metadata_path = os.path.join(episode_folder, "rl_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

        if not self.send_data_to_learner:
            return

        online_data = {
            "episode_name": episode_name,
            "success": success,
        }
        self.actor_node.send_transitions(online_data)
        print(f"[Env Runner] Sent episode to learner: {episode_name}, success={success}")

    def cleanup(self):
        """Clean up resources."""
        self.env.cleanup()

    def run(self):
        """
        Main loop for running the env runner.

        Collects episodes interactively with user confirmation.
        """
        print("[Env Runner] Starting main loop...")
        print_env_status(self.env)

        # Try to load latest RL checkpoint from disk (only if resume_rl is True)
        if self.resume_rl and self.controller.rl_actor is None:
            self._try_load_latest_checkpoint()

        while True:
            input(f"[Env Runner] Press Enter to start episode #{self.num_episodes}")

            # Check for weight updates
            if self.receive_and_update_weights():
                print("[Env Runner] Applied updated weights from learner")

            # Run episode
            episode_info = self.run_episode()

            if episode_info["aborted"]:
                print("[Env Runner] Episode aborted, not counting")
                print_env_status(self.env)
                continue

            # User evaluation
            print(f"[Env Runner] Episode saved: {episode_info['episode_folder']}")
            print("Do you want to keep this episode?")
            print("    d: delete the episode")
            print("    s: keep as success")
            print("    f: keep as failure")
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            c = input("Select option: ").strip().lower()

            if c == "d":
                print("[Env Runner] Deleting episode")
                if os.path.exists(episode_info["episode_folder"]):
                    shutil.rmtree(episode_info["episode_folder"])
            elif c == "s":
                self.send_episode_to_learner(episode_info, success=True)
                self.num_episodes += 1
            elif c == "f":
                self.send_episode_to_learner(episode_info, success=False)
                self.num_episodes += 1
            else:
                # Default to success
                self.send_episode_to_learner(episode_info, success=True)
                self.num_episodes += 1

            print_env_status(self.env)

            # Options for next step
            print("Options:")
            print("     c: continue to next episode.")
            print("     j: reset, jog, calibrate, then continue.")
            print("     r: reset to default pose, then continue.")
            print("     b: reset to default pose, then quit the program.")
            print("     others: quit the program.")
            c = input("Please select an option: ").strip().lower()

            if c in ("r", "b", "j"):
                self._reset_to_default_pose()
            elif c == "c":
                pass
            else:
                print("Quitting the program.")
                break

            if c == "b":
                input("Press Enter to quit program.")
                break

            if c == "j":
                input("Once robot is stopped, leave the robot free, Press Enter to run calibration.")
                print("---- Calibrating the robot. ----")
                self.env.calibrate_robot_wrench(NSamples=100)
                print("---- Calibration done. ----")
                print_env_status(self.env)
                input("Hold the handle, Press Enter to enter a 1 second jog mode.")
                self.env.set_high_level_free_jogging()
                time.sleep(1)
                self.env.set_high_level_maintain_position()
                input("Jogging is done. Press Enter to continue.")

            print("Continuing to execution.")

        self.cleanup()
        print("[Env Runner] Finished")

    def _reset_to_default_pose(self):
        """Reset robot to default pose."""
        print("[Env Runner] Resetting to default pose...")

        fixed_target_pose = self.reset_pose

        obs_raw = self.env.get_observation_from_buffer()
        N = 100
        duration_s = 5
        sample_indices = np.linspace(0, 1, N)
        timestamps = sample_indices * duration_s

        homing_ts_targets = np.zeros([7 * len(self.id_list), N])
        for id in self.id_list:
            ts_pose_fb = obs_raw[f"ts_pose_fb_{id}"][-1]
            pose7_waypoints = su.pose7_interp(ts_pose_fb, fixed_target_pose, sample_indices)
            homing_ts_targets[0 + id * 7 : 7 + id * 7, :] = pose7_waypoints.T

        time_now_s = self.env.current_hardware_time_s
        self.env.schedule_controls(
            pose7_cmd=homing_ts_targets,
            timestamps=(timestamps + time_now_s) * 1000,
        )

        # Wait for motion to complete
        time.sleep(duration_s + 0.5)
        print("[Env Runner] Reset complete")


def main():
    """Main function using shared RL finetuning config."""
    from dice_rl.config.rl_finetuning_config import (
        control_para,
        hardware_para,
        model_para,
        online_learning_para,
        checkpoint_folder_path,
    )

    runner = RLFinetuningEnvRunner(
        # Policy
        policy_ckpt_path=model_para["pretrained_flow_policy_path"],
        device=control_para.get("device", "cuda"),
        # Hardware
        hardware_config_path=hardware_para["hardware_config_path"],
        backend=hardware_para.get("backend", "manip_server"),
        # Control
        raw_time_step_s=control_para["raw_time_step_s"],
        slow_down_factor=control_para["slow_down_factor"],
        sparse_execution_horizon=control_para["sparse_execution_horizon"],
        delay_tolerance_s=control_para["delay_tolerance_s"],
        max_duration_s=control_para["max_duration_s"],
        fix_orientation=control_para.get("fix_orientation", False),
        translational_stiffness=control_para.get("translational_stiffness", [1500, 1500, 500]),
        rotational_stiffness=control_para.get("rotational_stiffness", 50.0),
        # Data saving
        data_folder_path=online_learning_para["data_folder_path"],
        # ZMQ (env_runner is the client for weights, server for transitions)
        network_server_endpoint=online_learning_para["network_server_endpoint"],
        network_weight_topic=online_learning_para["network_weight_topic"],
        transitions_server_endpoint=online_learning_para["transitions_server_endpoint"],
        transitions_topic=online_learning_para["transitions_topic"],
        transitions_topic_expire_time_s=online_learning_para.get("transitions_topic_expire_time_s", 1200.0),
        # Control flags (always send episodes and receive weights for RL finetuning)
        send_data_to_learner=True,
        receive_weights_from_learner=True,
        rl_num_inference_steps=model_para.get("rl_num_inference_steps", None),
        # RL checkpoint auto-loading on restart
        rl_checkpoint_dir=os.path.join(checkpoint_folder_path, "rl_finetuning"),
        actor_hidden_dims=model_para.get("actor_hidden_dims", [1024, 1024, 1024]),
        critic_hidden_dims=model_para.get("critic_hidden_dims", [1024, 1024, 1024]),
        actor_activation_type=model_para.get("activation_type", "GELU"),
        actor_use_layernorm=model_para.get("use_layernorm", True),
        reset_pose=control_para["reset_pose"],
        resume_rl=control_para.get("resume_rl", False),
    )
    runner.run()


if __name__ == "__main__":
    main()