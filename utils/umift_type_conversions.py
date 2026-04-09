import utils.spatial_math as su
from utils.computer_vision import get_image_transform_with_border
from utils.common import dict_apply

import numpy as np
from typing import Union, Dict, Optional
import zarr
import torch

##
## raw: keys used in the dataset. Each key contains data for a whole episode
## obs: keys used in inference. Needs some pre-processing before sending to the policy NN.
## obs_preprocessed: obs with normalized rgb keys. len = whole episode
## obs_sample: len = obs horizon, pose computed relative to current pose (id = -1)
## action: pose command in world frame. len = whole episode
## action_sample: len = action horizon, pose computed relative to current pose (id = 0)


def raw_to_obs(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    shape_meta: dict,
):
    """convert shape_meta.raw data to shape_meta.obs.

    This function keeps image data as compressed zarr array in memory, while loads and decompresses
    low dim data.

    Args:
      raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
      episode_data: output dictionary that matches shape_meta.obs
    """
    episode_data["obs"] = {}
    # obs.rgb: keep entry, keep as compressed zarr array in memory
    for key, attr in shape_meta["raw"].items():
        type = attr.get("type", "low_dim")
        if type == "rgb":
            # obs.rgb: keep as compressed zarr array in memory
            episode_data["obs"][key] = raw_data[key]

    # obs.low_dim: load entry, convert to obs.low_dim
    for id in shape_meta["id_list"]:
        pose7_fb = raw_data[f"ts_pose_fb_{id}"]
        pose9_fb = su.SE3_to_pose9(su.pose7_to_SE3(pose7_fb))

        episode_data["obs"][f"robot{id}_eef_pos"] = pose9_fb[..., :3]
        episode_data["obs"][f"robot{id}_eef_rot_axis_angle"] = pose9_fb[..., 3:]

        # optional: wrench
        if "robot0_eef_wrench_left" in shape_meta["obs"].keys():
            wrench_left = raw_data[f"wrench_left_{id}"] 
            wrench_right = raw_data[f"wrench_right_{id}"]
            episode_data["obs"][f"robot{id}_eef_wrench_left"] = wrench_left[:]
            episode_data["obs"][f"robot{id}_eef_wrench_right"] = wrench_right[:] 
            episode_data["obs"][f"wrench_time_stamps_{id}"] = raw_data[f"wrench_time_stamps_{id}"][:]

        # optional: abs
        if "robot0_abs_eef_pos" in shape_meta["obs"].keys():
            episode_data["obs"][f"robot{id}_abs_eef_pos"] = pose9_fb[..., :3]
            episode_data["obs"][f"robot{id}_abs_eef_rot_axis_angle"] = pose9_fb[..., 3:]

        # optional: gripper
        if "robot0_gripper" in shape_meta["obs"].keys():
            episode_data["obs"][f"robot{id}_gripper"] = raw_data[f"gripper_{id}"][:]
            episode_data["obs"][f"gripper_time_stamps_{id}"] = raw_data[
                f"gripper_time_stamps_{id}"
            ][:]

        # timestamps
        episode_data["obs"][f"rgb_time_stamps_{id}"] = raw_data[
            f"rgb_time_stamps_{id}"
        ][:]
        episode_data["obs"][f"robot_time_stamps_{id}"] = raw_data[
            f"robot_time_stamps_{id}"
        ][:]

        # indices

        if "ultrawide_0" in shape_meta["obs"].keys():
            episode_data["obs"][f"map_to_uw_idx_{id}"] = raw_data[
                f"map_to_uw_idx_{id}"
            ][:]

        if "depth_0" in shape_meta["obs"].keys():   
            episode_data["obs"][f"map_to_d_idx_{id}"] = raw_data[
                f"map_to_d_idx_{id}"
            ][:]

def raw_to_obs_inference(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    shape_meta: dict,
):
    """convert shape_meta.raw data to shape_meta.obs.

    This function keeps image data as compressed zarr array in memory, while loads and decompresses
    low dim data.

    Args:
      raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
      episode_data: output dictionary that matches shape_meta.obs
    """
    episode_data["obs"] = {}
    # obs.rgb: keep entry, keep as compressed zarr array in memory
    for key, attr in shape_meta["raw"].items():
        type = attr.get("type", "low_dim")
        if type == "rgb":
            # obs.rgb: keep as compressed zarr array in memory
            episode_data["obs"][key] = raw_data[key]

    # obs.low_dim: load entry, convert to obs.low_dim
    for id in shape_meta["id_list"]:
        pose7_fb = raw_data[f"ts_pose_fb_{id}"]
        pose9_fb = su.SE3_to_pose9(su.pose7_to_SE3(pose7_fb))

        episode_data["obs"][f"robot{id}_eef_pos"] = pose9_fb[..., :3]
        episode_data["obs"][f"robot{id}_eef_rot_axis_angle"] = pose9_fb[..., 3:]

        # optional: wrench
        if "robot0_eef_wrench_left" in shape_meta["obs"].keys():
            wrench_left = raw_data[f"wrench_left_{id}"] 
            wrench_right = raw_data[f"wrench_right_{id}"]
            episode_data["obs"][f"robot{id}_eef_wrench_left"] = wrench_left[:]
            episode_data["obs"][f"robot{id}_eef_wrench_right"] = wrench_right[:] 
            episode_data["obs"][f"wrench_time_stamps_{id}"] = raw_data[f"wrench_time_stamps_{id}"][:]

        # optional: abs
        if "robot0_abs_eef_pos" in shape_meta["obs"].keys():
            episode_data["obs"][f"robot{id}_abs_eef_pos"] = pose9_fb[..., :3]
            episode_data["obs"][f"robot{id}_abs_eef_rot_axis_angle"] = pose9_fb[..., 3:]

        # optional: gripper
        if "robot0_gripper" in shape_meta["obs"].keys():
            episode_data["obs"][f"robot{id}_gripper"] = raw_data[f"gripper_{id}"][:]
            episode_data["obs"][f"gripper_time_stamps_{id}"] = raw_data[
                f"gripper_time_stamps_{id}"
            ][:]

        # timestamps
        episode_data["obs"][f"rgb_time_stamps_{id}"] = raw_data[
            f"rgb_time_stamps_{id}"
        ][:]
        episode_data["obs"][f"robot_time_stamps_{id}"] = raw_data[
            f"robot_time_stamps_{id}"
        ][:]


        

def raw_to_action9(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    id_list: list,
):
    """Convert shape_meta.raw data to shape_meta.action.
    Note: if relative action is used, the relative pose still needs to be computed every time a sample
    is made. This function only converts the whole episode, and does not know what pose to be relative to.

    Args:
        raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
        episode_data: output dictionary that has an 'action' field that matches shape_meta.action
    """
    action = []
    action_lens = []
    for id in id_list:
        # action: assemble from low_dim
        action_lens.append(raw_data[f"ts_pose_command_{id}"].shape[0])
        ts_pose7_command = raw_data[f"ts_pose_command_{id}"][:]
        ts_pose9_command = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_command))
        action.append(ts_pose9_command)

    action_len = min(action_lens)
    action = [x[:action_len] for x in action]

    episode_data["action"] = np.concatenate(action, axis=-1)
    assert episode_data["action"].shape[1] == 9 or episode_data["action"].shape[1] == 18

    # action timestamps is set according to robot 0
    episode_data["action_time_stamps"] = raw_data["robot_time_stamps_0"][:action_len]


def raw_to_action19(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    id_list: list,
):
    """Convert shape_meta.raw data to shape_meta.action.
    Note: if relative action is used, the relative pose still needs to be computed every time a sample
    is made. This function only converts the whole episode, and does not know what pose to be relative to.

    Args:
        raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
        episode_data: output dictionary that has an 'action' field that matches shape_meta.action
    """
    action = []
    action_lens = []
    for id in id_list:
        # action: assemble from low_dim
        action_lens.append(raw_data[f"ts_pose_command_{id}"].shape[0])
        ts_pose7_command = raw_data[f"ts_pose_command_{id}"][:]
        ts_pose9_command = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_command))
        ts_pose7_virtual_target = raw_data[f"ts_pose_virtual_target_{id}"][:]
        ts_pose9_virtual_target = su.SE3_to_pose9(
            su.pose7_to_SE3(ts_pose7_virtual_target)
        )
        stiffness = raw_data[f"stiffness_{id}"][:][:, np.newaxis]
        action.append(
            np.concatenate(
                [ts_pose9_command, ts_pose9_virtual_target, stiffness], axis=-1
            )
        )
    # action: trim to the shortest length
    action_len = min(action_lens)
    action = [x[:action_len] for x in action]

    episode_data["action"] = np.concatenate(action, axis=-1)
    assert (
        episode_data["action"].shape[1] == 19 or episode_data["action"].shape[1] == 38
    )

    # action timestamps is set according to robot 0
    episode_data["action_time_stamps"] = raw_data["robot_time_stamps_0"][:action_len]


def raw_to_action21(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    id_list: list,
):
    """Convert shape_meta.raw data to shape_meta.action.
    Note: if relative action is used, the relative pose still needs to be computed every time a sample
    is made. This function only converts the whole episode, and does not know what pose to be relative to.

    Args:
        raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
        episode_data: output dictionary that has an 'action' field that matches shape_meta.action
    """
    action = []
    action_lens = []
    for id in id_list:
        # action: assemble from low_dim
        action_lens.append(raw_data[f"ts_pose_command_{id}"].shape[0])
        ts_pose7_command = raw_data[f"ts_pose_command_{id}"][:]
        ts_pose9_command = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_command))
        ts_pose7_virtual_target = raw_data[f"ts_pose_virtual_target_{id}"][:]
        ts_pose9_virtual_target = su.SE3_to_pose9(
            su.pose7_to_SE3(ts_pose7_virtual_target)
        )
        stiffness = raw_data[f"stiffness_{id}"][:][:, np.newaxis]  # (T,) -> (T, 1)
        gripper = raw_data[f"gripper_{id}"][:]  # (T, 1)
        # Calculate grasp force. Average of contact force from both fingers. Everything is defined in the TCP frame. Grasp force is assumed to be positive under compression.
        contact_force_left = -raw_data[f"wrench_left_{id}"][:, 0:1]
        contact_force_right = raw_data[f"wrench_right_{id}"][:, 0:1]
        grasping_force = (contact_force_left + contact_force_right) / 2
        # Downsample wrench data to match the action length based on timestamps
        robot_timestamps = raw_data[f"robot_time_stamps_{id}"][:].reshape(-1)
        wrench_timestamps = raw_data[f"wrench_time_stamps_{id}"][:].reshape(-1)
        # Find closest matching timestamps
        indices = np.searchsorted(wrench_timestamps, robot_timestamps, side="left")
        indices = np.clip(indices, 0, len(wrench_timestamps) - 1)  # Ensure indices are valid
        downsampled_grasping_force = grasping_force[indices]
        action.append(
            np.concatenate(
                [ts_pose9_command, ts_pose9_virtual_target, stiffness, gripper, downsampled_grasping_force], axis=-1
            )
        )
    # action: trim to the shortest length
    action_len = min(action_lens)
    action = [x[:action_len] for x in action]

    episode_data["action"] = np.concatenate(action, axis=-1)
    assert (
        episode_data["action"].shape[1] == 21 or episode_data["action"].shape[1] == 42
    )

    # action timestamps is set according to robot 0
    episode_data["action_time_stamps"] = raw_data["robot_time_stamps_0"][:action_len]


def obs_rgb_preprocess(
    obs: dict,
    obs_output: dict,
    reshape_mode: str,
    shape_meta: dict,
):
    """Pre-process the rgb data in the obs dictionary as inputs to policy network.

    This function does the following to the rgb keys in the obs dictionary:
    * Unpack/unzip it, if the rgb data is still stored as a compressed zarr array (not recommended)
    * Reshape the rgb image, or just check its shape.
    * Convert uint8 (0~255) to float32 (0~1)
    * Move its axes from THWC to TCHW.
    Since this function unpacks the whole key, it should only be used for online inference.
    If used in training, so the data length is the obs horizon instead of the whole episode len.

    Args:
        obs: dict with keys from shape_meta.obs
        obs_output: dict with the same keys but processed images
        reshape_mode: One of 'reshape', 'check', or 'none'.
        shape_meta: the shape_meta from task.yaml
    """
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = attr.get("shape")
        if type == "rgb":
            this_imgs_in = obs[key]
            t, hi, wi, ci = this_imgs_in.shape
            co, ho, wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi):
                if reshape_mode == "reshape":
                    tf = get_image_transform_with_border(
                        input_res=(wi, hi), output_res=(wo, ho), bgr_to_rgb=False
                    )
                    out_imgs = np.stack([tf(x) for x in this_imgs_in])
                elif reshape_mode == "check":
                    print(
                        f"[obs_rgb_preprocess] shape check failed! Require {ho}x{wo}, get {hi}x{wi}"
                    )
                    assert False
            if this_imgs_in.dtype == np.uint8 or this_imgs_in.dtype == np.int32:
                out_imgs = out_imgs.astype(np.float32) / 255

            # THWC to TCHW
            obs_output[key] = np.moveaxis(out_imgs, -1, 1)


def obs_rgb_depth_preprocess(
    obs: dict,
    obs_output: dict,
    reshape_mode: str,
    shape_meta: dict,
):
    """Pre-process the rgb and depth data in the obs dictionary as inputs to policy network.

    This function does the following to the rgb keys in the obs dictionary:
    * Unpack/unzip it, if the rgb data is still stored as a compressed zarr array (not recommended)
    * Reshape the rgb image, or just check its shape.
    * Convert uint8 (0~255) to float32 (0~1)
    * Move its axes from THWC to TCHW.
    Since this function unpacks the whole key, it should only be used for online inference.
    If used in training, so the data length is the obs horizon instead of the whole episode len.

    This function does the following to the depth keys in the obs dictionary:
    * Unpack/unzip it, if the depth data is still stored as a compressed zarr array (not recommended)
    * Reshape the depth image, or just check its shape.
    * The range of the raw data is 0.0 ~ {clip}. Normalize it to 0.0 ~ 1.0. 
    * Move its axes from THWC to TCHW.

    Args:
        obs: dict with keys from shape_meta.obs
        obs_output: dict with the same keys but processed images
        reshape_mode: One of 'reshape', 'check', or 'none'.
        shape_meta: the shape_meta from task.yaml
    """
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = attr.get("shape")
        if type == "rgb" and ("rgb" in key or "ultrawide" in key):
            this_imgs_in = obs[key]
            t, hi, wi, ci = this_imgs_in.shape
            co, ho, wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi):
                if reshape_mode == "reshape":
                    tf = get_image_transform_with_border(
                        input_res=(wi, hi), output_res=(wo, ho), bgr_to_rgb=False
                    )
                    out_imgs = np.stack([tf(x) for x in this_imgs_in])
                elif reshape_mode == "check":
                    print(
                        f"[obs_rgb_depth_preprocess] shape check failed! Require {ho}x{wo}, get {hi}x{wi}"
                    )
                    assert False
            if this_imgs_in.dtype == np.uint8 or this_imgs_in.dtype == np.int32:
                out_imgs = out_imgs.astype(np.float32) / 255

            # THWC to TCHW
            obs_output[key] = np.moveaxis(out_imgs, -1, 1)

        elif type == "rgb" and "depth" in key:
            this_depth_in = obs[key]
            t, hi, wi, ci = this_depth_in.shape
            co, ho, wo = shape
            assert ci == co
            out_depth = this_depth_in
            if (ho != hi) or (wo != wi):
                if reshape_mode == "reshape":
                    tf = get_image_transform_with_border(
                        input_res=(wi, hi), output_res=(wo, ho), bgr_to_rgb=False
                    )
                    out_depth = np.stack([tf(x) for x in this_depth_in])
                elif reshape_mode == "check":
                    print(
                        f"[obs_rgb_depth_preprocess] depth shape check failed! Require {ho}x{wo}, get {hi}x{wi}"
                    )
                    assert False
            assert this_depth_in.dtype == np.float16

            # depth is assumed to be clipped. Normalize it to 0.0 ~ 1.0, just like rgb.
            depth_clip = obs_shape_meta[key]["clip"]

            assert np.all(out_depth <= depth_clip), f"Depth values are greater than the clip value: {np.max(out_depth)}"
            assert np.all(out_depth >= 0.0), f"Depth values are less than 0.0: {np.min(out_depth)}"

            out_depth = out_depth / depth_clip

            obs_output[key] = np.moveaxis(out_depth, -1, 1)






def sparse_obs_to_obs_sample(
    obs_sparse: dict,  # each key: (T, D)
    shape_meta: dict,
    reshape_mode: str,
    id_list: list,
    ignore_rgb: bool = False,
):
    """Prepare a sample of sparse obs as inputs to policy network.

    After packing an obs dictionary with keys according to shape_meta.sample.obs.sparse, with
    length corresponding to the correct horizons, pass it to this function to get it ready
    for the policy network.

    It does two things:
        1. RGB: unpack, reshape, normalize, turn into float
        2. low dim: convert pose to relative pose, turn into float

    Args:
        obs_sparse: dict with keys from shape_meta['sample']['obs']['sparse']
        shape_meta: the shape_meta from task.yaml
        reshape_mode: One of 'reshape', 'check', or 'none'.
        ignore_rgb: if True, skip the rgb keys. Used when computing normalizers.
    return:
        sparse_obs_processed: dict with keys from shape_meta['sample']['obs']['sparse']
        base_SE3: the initial pose used for relative pose calculation
    """
    sparse_obs_processed = {}
    assert len(obs_sparse) > 0
    if not ignore_rgb:
        # obs_rgb_preprocess(obs_sparse, sparse_obs_processed, reshape_mode, shape_meta)
        obs_rgb_depth_preprocess(obs_sparse, sparse_obs_processed, reshape_mode, shape_meta)

    # copy all low dim keys
    for key, attr in shape_meta["obs"].items():
        type = attr.get("type", "low_dim")
        if type == "low_dim":
            sparse_obs_processed[key] = obs_sparse[key].astype(
                np.float32
            )  # astype() makes a copy

    # generate relative pose
    base_SE3_WT = []
    for id in id_list:
        # convert pose to mat
        SE3_WT = su.pose9_to_SE3(
            np.concatenate(
                [
                    sparse_obs_processed[f"robot{id}_eef_pos"],
                    sparse_obs_processed[f"robot{id}_eef_rot_axis_angle"],
                ],
                axis=-1,
            )
        )

        # HC TODO: fully understand. Why last pose instead of first?
        # solve relative obs 
        base_SE3_WT.append(SE3_WT[-1])
        SE3_base_i = su.SE3_inv(base_SE3_WT[id]) @ SE3_WT

        pose9_relative = su.SE3_to_pose9(SE3_base_i)
        sparse_obs_processed[f"robot{id}_eef_pos"] = pose9_relative[..., :3]
        sparse_obs_processed[f"robot{id}_eef_rot_axis_angle"] = pose9_relative[..., 3:]

        # HC TODO: double check
        # solve relative wrench
        # Note:
        #   The correct way to compute relative wrench requires
        #   using a different adjoint matrix for each time step of wrench.
        #   This can be expensive when the number of wrench samples is large.
        #   As an approximation, we use the adjoint matrix of the last pose.
        #   When the wrench is reported in tool frame, SE3_i_base is the identity matrix.
        SE3_i_base = su.SE3_inv(SE3_base_i)[-1]

        if "robot0_eef_wrench_left" in shape_meta["obs"].keys():
            wrench_left = su.transpose(su.SE3_to_adj(SE3_i_base)) @ np.expand_dims(
                obs_sparse[f"robot{id}_eef_wrench_left"], -1
            )
            wrench_right = su.transpose(su.SE3_to_adj(SE3_i_base)) @ np.expand_dims(
                obs_sparse[f"robot{id}_eef_wrench_right"], -1
            )
            sparse_obs_processed[f"robot{id}_eef_wrench_left"] = np.squeeze(wrench_left)
            sparse_obs_processed[f"robot{id}_eef_wrench_right"] = np.squeeze(wrench_right)

        # double check the shape
        for key, attr in shape_meta["sample"]["obs"]["sparse"].items():
            sparse_obs_horizon = attr["horizon"]
            if shape_meta["obs"][key]["type"] == "low_dim":
                assert len(sparse_obs_processed[key].shape) == 2  # (T, D)
                assert sparse_obs_processed[key].shape[0] == sparse_obs_horizon
            else:
                if not ignore_rgb:
                    assert len(sparse_obs_processed[key].shape) == 4  # (T, C, H, W)
                    assert sparse_obs_processed[key].shape[0] == sparse_obs_horizon

    return sparse_obs_processed, base_SE3_WT


def dense_obs_to_obs_sample(
    obs_dense: dict,  # each key: (H, T, D) (training) or (T, D) (testing)
    shape_meta: dict,
    SE3_WBase: list,
    id_list: list,
):
    """Prepare a sample of obs as inputs to policy network.

    After packing an obs dictionary with keys according to shape_meta.sample.obs.dense, with
    length corresponding to the correct horizons, pass it to this function to get it ready
    for the policy network.

    Since dense obs only contains low dim data, it only does the low dim part:
        low dim: convert pose to relative pose about the initial pose of the SPARSE horizon

    Args:
        obs_dense: dict with keys from shape_meta['sample']['obs']['dense']
        shape_meta: the shape_meta from task.yaml
        SE3_WBase: a list of current pose SE3s, one per robot. The initial pose used for relative pose calculation
    """
    dense_obs_processed = {}
    for key in shape_meta["sample"]["obs"]["dense"].keys():
        dense_obs_processed[key] = obs_dense[key].astype(
            np.float32
        )  # astype() makes a copy
    # get the length of the first key in the dictionary obs_dense

    data_shape = next(iter(obs_dense.values())).shape
    assert len(data_shape) == 3
    H = data_shape[0]

    # convert each dense horizon to the same relative pose
    for id in id_list:
        for step in range(H):
            # generate relative pose. Everything is (T, D)
            # convert pose to mat
            SE3_WT = su.pose9_to_SE3(
                np.concatenate(
                    [
                        obs_dense[f"robot{id}_eef_pos"][step],
                        obs_dense[f"robot{id}_eef_rot_axis_angle"][step],
                    ],
                    axis=-1,
                )
            )

            # solve relative obs
            SE3_BaseT = np.linalg.inv(SE3_WBase[id]) @ SE3_WT

            pose9_relative = su.SE3_to_pose9(SE3_BaseT).astype(np.float32)
            dense_obs_processed[f"robot{id}_eef_pos"][step] = pose9_relative[..., :3]
            dense_obs_processed[f"robot{id}_eef_rot_axis_angle"][step] = pose9_relative[
                ..., 3:
            ]

            # solve relative wrench
            # Note:
            #   The correct way to compute relative wrench requires
            #   using a different adjoint matrix for each time step of wrench.
            #   This can be expensive when the number of wrench samples is large.
            #   As an approximation, we use the adjoint matrix of the last pose.
            #   When the wrench is reported in tool frame, SE3_i_base is the identity matrix.
            SE3_i_base = su.SE3_inv(SE3_BaseT[-1])

            wrench_left_0 = su.transpose(su.SE3_to_adj(SE3_i_base)) @ np.expand_dims(
                obs_dense[f"robot{id}_eef_wrench_left"][step], -1
            )
            wrench_right_0 = su.transpose(su.SE3_to_adj(SE3_i_base)) @ np.expand_dims(
                obs_dense[f"robot{id}_eef_wrench_right"][step], -1
            )
            dense_obs_processed[f"robot{id}_eef_wrench_left"][step] = np.squeeze(
                wrench_left_0
            ).astype(np.float32)
            dense_obs_processed[f"robot{id}_eef_wrench_right"][step] = np.squeeze(
                wrench_right_0
            ).astype(np.float32)

    # double check the shape
    for key in shape_meta["sample"]["obs"]["dense"].keys():
        assert dense_obs_processed[key].shape[0] == H
        assert len(dense_obs_processed[key].shape) == 3  # (H, T, D)

    return dense_obs_processed


def obs_to_obs_sample(
    obs_sparse: dict,  # each key: (T, D)
    obs_dense: dict,  # each key: (H, T, D)
    shape_meta: dict,
    reshape_mode: str,
    id_list: list,
    ignore_rgb: bool = False,
):
    """Prepare a sample of obs as inputs to policy network.

    After packing an obs dictionary with keys according to shape_meta.obs, with
    length corresponding to the correct horizons, pass it to this function to get it ready
    for the policy network.

    It does two things:
        1. RGB: unpack, reshape, normalize, turn into float
        2. low dim: convert pose to relative pose, turn into float
    For sparse obs, it does both. For dense obs, it only does the low dim part, and all poses are
    computed relative to the same current pose (id = 0).

    Args:
        obs_sparse: dict with keys from shape_meta['sample']['obs']['sparse']
        obs_dense: dict with keys from shape_meta['sample']['obs']['dense']
        shape_meta: the shape_meta from task.yaml
        reshape_mode: One of 'reshape', 'check', or 'none'.
        ignore_rgb: if True, skip the rgb keys. Used when computing normalizers.
    """
    obs_processed = {"sparse": {}, "dense": {}}
    obs_processed["sparse"], base_pose_mat = sparse_obs_to_obs_sample(
        obs_sparse, shape_meta, reshape_mode, id_list, ignore_rgb
    )
    if len(obs_dense) > 0:
        obs_processed["dense"] = dense_obs_to_obs_sample(
            obs_dense, shape_meta, base_pose_mat, id_list
        )

    return obs_processed, base_pose_mat


def action9_to_action_sample(
    action_sparse: np.ndarray,  # (T, D), D = 9
    action_dense: np.ndarray,  # (H, T, D), D = 9
    id_list: list,
    base_pose: list,
):
    """Prepare a sample of actions as labels for the policy network.

    This function is used in training. It takes a sample of actions (len = action_horizon)
    and convert the poses in it to be relative to the current pose (id = 0).

    """
    action_processed = {"sparse": {}, "dense": {}}
    if len(action_sparse) > 0:
        T, D = action_sparse.shape
        if len(id_list) == 1:
            assert D == 9
        else:
            assert D == 18
    if len(action_dense) > 0:
        H, T, D = action_dense.shape
        if len(id_list) == 1:
            assert D == 9
        else:
            assert D == 18

    def action9_preprocess(action: np.ndarray, SE3_WBase: np.ndarray):
        # generate relative pose
        # convert pose to mat
        pose9 = action
        SE3 = su.pose9_to_SE3(pose9)

        # solve relative obs
        SE3_WBase_inv = su.SE3_inv(SE3_WBase)
        SE3_relative = SE3_WBase_inv @ SE3

        pose9_relative = su.SE3_to_pose9(SE3_relative)

        return pose9_relative

    if len(action_sparse) > 0:
        if len(id_list) == 1:
            action_processed["sparse"] = action9_preprocess(action_sparse, base_pose[0])
        else:
            action_processed["sparse"] = np.concatenate(
                [
                    action9_preprocess(action_sparse[:, :9], base_pose[0]),
                    action9_preprocess(action_sparse[:, 9:18], base_pose[1]),
                ],
                axis=-1,
            )

    if len(action_dense) > 0:
        action_processed["dense"] = np.zeros_like(action_dense)
        H = action_dense.shape[0]
        for step in range(H):
            if len(id_list) == 1:
                # generate relative pose
                # convert pose to mat
                pose9 = action_dense[step]  # Tx9
                SE3 = su.pose9_to_SE3(pose9)  # Tx4x4

                # solve relative obs
                SE3_relative = su.SE3_inv(base_pose[0]) @ SE3
                pose9_relative = su.SE3_to_pose9(SE3_relative)
                action_processed["dense"][step] = pose9_relative
            else:
                # generate relative pose
                # convert pose to mat
                pose90 = action_dense[step][:, :9]  # Tx9
                pose91 = action_dense[step][:, 9:18]  # Tx9
                SE30 = su.pose9_to_SE3(pose90)
                SE31 = su.pose9_to_SE3(pose91)

                # solve relative obs
                SE3_relative0 = su.SE3_inv(base_pose[0]) @ SE30
                SE3_relative1 = su.SE3_inv(base_pose[1]) @ SE31
                pose9_relative0 = su.SE3_to_pose9(SE3_relative0)
                pose9_relative1 = su.SE3_to_pose9(SE3_relative1)
                action_processed["dense"][step] = np.concatenate(
                    [
                        pose9_relative0,
                        pose9_relative1,
                    ],
                    axis=-1,
                )

    # double check the shape
    if len(action_sparse) > 0:
        assert action_processed["sparse"].shape == (T, D)
    if len(action_dense) > 0:
        assert action_processed["dense"].shape == action_dense.shape

    return action_processed


def action19_to_action_sample(
    action_sparse: np.ndarray,  # (T, D), D = 19 or 38
    action_dense: np.ndarray,  # (H, T, D) not used
    id_list: list,
    base_pose: list,
):
    """Prepare a sample of actions as labels for the policy network.

    This function is used in training. It takes a sample of actions (len = action_horizon)
    and convert the poses in it to be relative to the current pose (id = 0).

    """
    action_processed = {"sparse": {}, "dense": {}}
    T, D = action_sparse.shape
    if len(id_list) == 1:
        assert D == 19
    else:
        assert D == 38

    def action19_preprocess(action: np.ndarray, SE3_WBase: np.ndarray):
        # generate relative pose
        # convert pose to mat
        pose9 = action[:, 0:9]
        pose9_vt = action[:, 9:18]
        stiffness = action[:, 18:19]
        SE3 = su.pose9_to_SE3(pose9)
        SE3_vt = su.pose9_to_SE3(pose9_vt)

        # solve relative obs
        SE3_WBase_inv = su.SE3_inv(SE3_WBase)
        SE3_relative = SE3_WBase_inv @ SE3
        SE3_vt_relative = SE3_WBase_inv @ SE3_vt
        pose9_relative = su.SE3_to_pose9(SE3_relative)
        pose9_vt_relative = su.SE3_to_pose9(SE3_vt_relative)

        return np.concatenate([pose9_relative, pose9_vt_relative, stiffness], axis=-1)

    if len(id_list) == 1:
        action_processed["sparse"] = action19_preprocess(action_sparse, base_pose[0])
    else:
        action_processed["sparse"] = np.concatenate(
            [
                action19_preprocess(action_sparse[:, :19], base_pose[0]),
                action19_preprocess(action_sparse[:, 19:38], base_pose[1]),
            ],
            axis=-1,
        )

    if len(action_dense) > 0:
        # not implemented properly
        raise NotImplementedError

    # double check the shape
    assert action_processed["sparse"].shape == (T, D)
    if len(action_dense) > 0:
        assert action_processed["dense"].shape == action_dense.shape

    return action_processed


def action21_to_action_sample(
    action_sparse: np.ndarray,  # (T, D), D = 21 or 42
    action_dense: np.ndarray,  # (H, T, D) not used
    id_list: list,
    base_pose: list,
):
    """Prepare a sample of actions as labels for the policy network.

    This function is used in training. It takes a sample of actions (len = action_horizon)
    and convert the poses in it to be relative to the current pose (id = 0).

    """
    action_processed = {"sparse": {}, "dense": {}}
    T, D = action_sparse.shape
    if len(id_list) == 1:
        assert D == 21
    else:
        assert D == 42

    def action21_preprocess(action: np.ndarray, SE3_WBase: np.ndarray):
        # generate relative pose
        # convert pose to mat
        pose9 = action[:, 0:9]
        pose9_vt = action[:, 9:18]
        stiffness = action[:, 18:19]
        gripper = action[:, 19:20]
        grasping_force = action[:, 20:21]
        SE3 = su.pose9_to_SE3(pose9)
        SE3_vt = su.pose9_to_SE3(pose9_vt)

        # HC TODO: double check
        # solve relative obs
        SE3_WBase_inv = su.SE3_inv(SE3_WBase)
        SE3_relative = SE3_WBase_inv @ SE3
        SE3_vt_relative = SE3_WBase_inv @ SE3_vt
        pose9_relative = su.SE3_to_pose9(SE3_relative)
        pose9_vt_relative = su.SE3_to_pose9(SE3_vt_relative)

        return np.concatenate(
            [pose9_relative, pose9_vt_relative, stiffness, gripper, grasping_force], axis=-1
        )

    if len(id_list) == 1:
        action_processed["sparse"] = action21_preprocess(action_sparse, base_pose[0])
    else:
        action_processed["sparse"] = np.concatenate(
            [
                action21_preprocess(action_sparse[:, :21], base_pose[0]),
                action21_preprocess(action_sparse[:, 21:42], base_pose[1]),
            ],
            axis=-1,
        )

    if len(action_dense) > 0:
        # not implemented properly
        raise NotImplementedError

    # double check the shape
    assert action_processed["sparse"].shape == (T, D)
    if len(action_dense) > 0:
        assert action_processed["dense"].shape == action_dense.shape

    return action_processed


def action9_postprocess(
    action: np.ndarray, current_SE3: list, id_list: list, fix_orientation=False, delta_pos_limit=None
):
    """Convert policy outputs from relative pose to world frame pose
    Used in online inference
    """

    action_SE3_absolute = [np.array] * len(id_list)
    for id in id_list:
        action_pose9 = action[..., 19 * id + 0 : 19 * id + 9]

        # TODO: apply limit here
        if delta_pos_limit is not None:
            delta_pos = action_pose9[:, :3]
            delta_pos = np.clip(delta_pos, -delta_pos_limit, delta_pos_limit)
            action_pose9[:, :3] = delta_pos

        action_SE3 = su.pose9_to_SE3(action_pose9)

        action_SE3_absolute[id] = current_SE3[id] @ action_SE3

        if fix_orientation:
            action_SE3_absolute[id][:, :3, :3] = current_SE3[id][:3, :3]

    # return pose matrices
    return action_SE3_absolute


def action19_postprocess(
    action: np.ndarray, current_SE3: list, id_list: list, fix_orientation=False
):
    """Convert policy outputs from relative pose to world frame pose
    Used in online inference
    """

    action_SE3_absolute = [np.array] * len(id_list)
    action_SE3_vt_absolute = [np.array] * len(id_list)
    stiffness = [0] * len(id_list)

    for id in id_list:
        action_pose9 = action[..., 19 * id + 0 : 19 * id + 9]
        action_pose9_vt = action[..., 19 * id + 9 : 19 * id + 18]
        stiffness[id] = action[..., 19 * id + 18]
        action_SE3 = su.pose9_to_SE3(action_pose9)
        action_SE3_vt = su.pose9_to_SE3(action_pose9_vt)

        action_SE3_absolute[id] = current_SE3[id] @ action_SE3
        action_SE3_vt_absolute[id] = current_SE3[id] @ action_SE3_vt

        if fix_orientation:
            action_SE3_absolute[id][:, :3, :3] = current_SE3[:3, :3]
            action_SE3_vt_absolute[id][:, :3, :3] = current_SE3[:3, :3]

    # return pose matrices
    return action_SE3_absolute, action_SE3_vt_absolute, stiffness

def action21_postprocess(
    action: np.ndarray, current_SE3: list, id_list: list, fix_orientation=False
):
    """Convert policy outputs from relative pose to world frame pose
    Used in online inference
    """
    # HC TODO: double check
    
    action_SE3_absolute = [np.array] * len(id_list)
    action_SE3_vt_absolute = [np.array] * len(id_list)
    stiffness = [0] * len(id_list)
    eoat = [np.array] * len(id_list)

    for id in id_list:
        action_pose9 = action[..., 21 * id + 0 : 21 * id + 9]
        action_pose9_vt = action[..., 21 * id + 9 : 21 * id + 18]
        stiffness[id] = action[..., 21 * id + 18]
        eoat[id] = action[..., 21 * id + 19: 21 * id + 21]
        action_SE3 = su.pose9_to_SE3(action_pose9)
        action_SE3_vt = su.pose9_to_SE3(action_pose9_vt)

        action_SE3_absolute[id] = current_SE3[id] @ action_SE3
        action_SE3_vt_absolute[id] = current_SE3[id] @ action_SE3_vt

        if fix_orientation:
            action_SE3_absolute[id][:, :3, :3] = current_SE3[:3, :3]
            action_SE3_vt_absolute[id][:, :3, :3] = current_SE3[:3, :3]

    # return pose matrices
    return action_SE3_absolute, action_SE3_vt_absolute, stiffness, eoat