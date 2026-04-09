# DICE-RL-Robot

RL finetuning for real robot manipulation policies. This system takes a pretrained behavior cloning (BC) policy (diffusion policy) and finetunes it online using reinforcement learning on a real robot.

## Architecture

Two-process architecture communicating via [RobotMQ](https://pypi.org/project/robotmq/) (ZMQ-based):

```
+-------------------------+          RobotMQ/ZMQ          +---------------------------+
|   Learner (GPU)         |<------------------------------|  Env Runner (Robot)       |
|                         |    episode metadata (ZMQ)     |                           |
|  - Receive episodes     |------------------------------>|  - Load pretrained policy |
|  - Process into replay  |    actor weights (ZMQ)        |  - Execute policy on robot|
|    buffer (online +     |                               |  - Save raw episode data  |
|    expert via RLPD)     |                               |  - Send episode to learner|
|  - Train actor & critic |                               |  - Receive & apply updated|
|  - Send updated weights |                               |    actor weights          |
+-------------------------+                               +---------------------------+
```

- **Learner** (GPU server): Receives episode metadata from the env runner, loads raw episode data from disk, processes it into a hybrid replay buffer (mixing online episodes with offline expert demonstrations via RLPD), trains a residual RL actor and critic ensemble, and sends updated actor weights back.
- **Env Runner** (Robot PC): Loads the pretrained BC policy (diffusion policy), runs it on the real robot to collect episodes (RGB, proprioception, wrench), saves raw data to disk, sends episode metadata to the learner via ZMQ, and applies updated actor weights when received.

---

## Table of Contents

1. [Installation](#installation)
2. [Environment Setup](#environment-setup)
3. [Step 1: Data Collection (Pretraining)](#step-1-data-collection-pretraining)
4. [Step 2: Diffusion Policy Training (BC)](#step-2-diffusion-policy-training-bc)
5. [Step 3: RL Finetuning](#step-3-rl-finetuning)
6. [Configuration Reference](#configuration-reference)
7. [Repository Structure](#repository-structure)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### 1. Clone the repository

```bash
git clone --recurse-submodules <repo-url>
cd DICE-RL-Robot
```

### 2. Create the conda environment

```bash
conda env create -f environment.yaml
conda activate dice-rl-robot
```

### 3. Install the Python package

**Learner PC (GPU server) -- Python only:**

```bash
pip install -e ".[learner]"
```

**Robot PC -- Python + C++ hardware drivers:**

```bash
pip install -e ".[robot]"
```

### 4. Build hardware interfaces (Robot PC only)

The `hardware_interfaces/` directory is a git submodule containing C++ drivers for the robot arm, gripper, cameras, and force/torque sensor. See [`hardware_interfaces/README.md`](hardware_interfaces/README.md) for build instructions and dependencies.

---

## Environment Setup

Before running any command, source `prepare.sh` to set environment variables and activate the conda env:

```bash
cd DICE-RL-Robot
. ./prepare.sh
```

Edit `prepare.sh` to match your local paths. The key variables are:

| Variable | Description | Example |
|----------|-------------|---------|
| `LD_LIBRARY_PATH` | Must include conda env's `lib/` dir for hardware binaries | `$HOME/miniforge3/envs/dice-rl-robot/lib/` |
| `DICE_RAW_DATASET_FOLDERS` | Where raw collected data is saved | `$HOME/data/real` |
| `DICE_DATASET_FOLDERS` | Where postprocessed zarr data lives | `$HOME/data/real_processed` |
| `DICE_CHECKPOINT_FOLDERS` | Where training outputs (BC and RL) are saved | `$HOME/training_outputs` |
| `DICE_HARDWARE_INTERFACES_ROOT` | Root of `hardware_interfaces/` | `$HOME/DICE-RL-Robot/hardware_interfaces` |
| `DICE_HARDWARE_CONFIG_FOLDERS` | Path to hardware YAML configs | `$DICE_HARDWARE_INTERFACES_ROOT/workcell/table_top_manip/config` |
| `DICE_CONTROL_LOG_FOLDERS` | Logging directory | `$HOME/data/control_log` |

---

## Step 1: Data Collection (Pretraining)

Collect kinesthetic demonstration episodes on the real robot. The robot runs in compliant (free-jogging) mode while a human guides it through the task.

### 1.1 Configure hardware

Edit the hardware config YAML for your setup. An example is at:
```
hardware_interfaces/workcell/table_top_manip/config/single_arm_data_collection.yaml
```

Key fields to set:
- `data_folder`: Output directory for raw episodes (e.g., `$HOME/data/real/my_task`)
- `reset_pose`: Home pose `[x, y, z, qw, qx, qy, qz]`
- Robot IP, gripper IP, camera device paths, F/T sensor IP
- Admittance controller parameters (stiffness, damping, inertia)

### 1.2 Run the data collection program

```bash
. ./prepare.sh
./hardware_interfaces/build/applications/manipulation_data_collection/manipulation_data_collection
```

**Controls:** Use an external keyboard/button device to control episode recording:
- Press the configured key to **start recording** an episode
- Press again to **stop recording** and save the episode
- The robot automatically returns to the reset pose between episodes

Each episode is saved as a folder (e.g., `episode_1742230408/`) containing:
```
episode_XXXXXXXXXX/
  rgb_0/               # RGB images (JPEG), one per camera frame
    0000.jpg, 0001.jpg, ...
  robot_data_0.json    # Robot pose feedback, timestamps
  wrench_data_0.json   # Force/torque sensor readings
  eoat_data_0.json     # Gripper state
  key_data.json        # Keyboard/button events
```

### 1.3 Process raw data into zarr format

The raw data must be converted into a zarr DirectoryStore before training. Edit `scripts/process_raw_data.py` to set your input/output directories:

```python
input_dir = pathlib.Path(os.environ.get("DICE_RAW_DATASET_FOLDERS") + "/your_task")
output_dir = pathlib.Path(os.environ.get("DICE_DATASET_FOLDERS") + "/your_task_processed")
id_list = [0]  # [0] for single-arm, [0, 1] for bimanual
```

Then run:
```bash
. ./prepare.sh
python scripts/process_raw_data.py
```

This reads raw images and JSON files, aligns timestamps, compresses RGB with JpegXL, and saves everything to zarr format.

The resulting zarr store has this structure:
```
my_task_processed/
  data/
    episode_0/
      rgb_0                     # (N, H, W, 3) uint8
      rgb_time_stamps_0         # (N,) float, milliseconds
      ts_pose_fb_0              # (T, 7) [x,y,z,qw,qx,qy,qz]
      ts_pose_command_0         # (T, 7)
      robot_time_stamps_0       # (T,) float
      wrench_0                  # (T, 6)
      wrench_time_stamps_0      # (T,)
      gripper_0                 # (T, 1)
      ...
    episode_1/
      ...
  meta/
    episode_rgb0_len            # Array of per-episode RGB frame counts
    episode_robot0_len          # Array of per-episode robot data lengths
    ...
```

---

## Step 2: Diffusion Policy Training (BC)

Train a diffusion policy on the processed zarr dataset using behavior cloning.

### 2.1 Create or edit a task config

Task configs live in `diffusion_policy/config/task/`. Each defines:
- Task name, dataset path
- Observation structure (RGB resolution, proprioception dims)
- Action structure (pose representation, action horizon)
- Sampling parameters (downsampling steps, horizons)

Example: `diffusion_policy/config/task/gear_insertion_no_force.yaml`

Key parameters to set:
```yaml
sparse_action_horizon: 20        # Number of action steps per chunk
sparse_action_down_sample_steps: 50  # Temporal downsampling of actions
dataset_path: /path/to/your/processed_zarr
```

### 2.2 Select the task in the workspace config

Edit `diffusion_policy/config/train_dp_workspace.yaml`:
```yaml
defaults:
  - _self_
  - task: your_task_config.yaml   # <-- point to your task yaml
```

### 2.3 Launch training

```bash
. ./prepare.sh
HYDRA_FULL_ERROR=1 accelerate launch train.py --config-name=train_dp_workspace
```

Multi-GPU training:
```bash
HYDRA_FULL_ERROR=1 accelerate launch --gpu_ids 0,1 --num_processes=2 --main_process_port=28888 \
    train.py --config-name=train_dp_workspace
```

Training outputs are saved to `$DICE_CHECKPOINT_FOLDERS/<timestamp>_<task_name>_dp/`:
- `checkpoints/` -- model checkpoints (`.ckpt`)
- `.hydra/config.yaml` -- full resolved config (important for RL finetuning)
- Weights & Biases logging (if configured)

---

## Step 3: RL Finetuning

Once you have a pretrained BC policy checkpoint, you can finetune it online on the real robot using residual RL with RLPD.

### 3.1 Configure RL finetuning

Edit `dice_rl/config/rl_finetuning_config.py`. This single config file is shared between the learner and env runner.

**Required settings (must be set before running):**

```python
# Path to pretrained BC checkpoint (.ckpt file)
model_para["pretrained_flow_policy_path"] = "/path/to/checkpoints/epoch=XXXX-train_loss=X.XXX.ckpt"

# Must match sparse_action_horizon from the BC checkpoint's config
control_para["sparse_execution_horizon"] = 20

# Path to expert dataset zarr (same data used for BC training), for RLPD
learner_para["expert_dataset_path"] = "/path/to/processed_zarr"

# Hardware config yaml (use the RL config, NOT the data collection config)
hardware_para["hardware_config_path"] = hardware_config_folder_path + "/belt_assembly.yaml"

# Reset pose for the robot between episodes
control_para["reset_pose"] = [x, y, z, qx, qy, qz, qw]

# Directory for online rollout data (use a fresh directory for each run)
online_learning_para["data_folder_path"] = data_folder_path + "/rl_run_01/"
```

**Important:** `sparse_execution_horizon` must exactly match the `sparse_action_horizon` in the BC checkpoint. Check the BC checkpoint's `.hydra/config.yaml` to verify.

### 3.2 Run RL finetuning

Open two terminals on the same machine (or on learner/robot PCs if running distributed).

**Terminal 1 -- Learner (GPU server):**
```bash
. ./prepare.sh
bash scripts/run_learner.sh
```

**Terminal 2 -- Env Runner (Robot PC):**
```bash
. ./prepare.sh
bash scripts/run_env_runner.sh
```

### 3.3 RL finetuning workflow

1. The **env runner** loads the pretrained BC policy and starts an interactive loop
2. Press Enter to start each episode. The robot executes the current policy
3. After each episode, you are asked to label it as success/failure
4. Episode data is sent to the **learner** via ZMQ (or saved to disk as fallback)
5. After `num_episodes_before_first_training` episodes, the learner begins training
6. The learner trains for `gradient_steps` gradient updates, mixing online data with expert data (RLPD)
7. Updated actor weights are sent back to the env runner
8. Subsequent training rounds trigger every `update_every_x_episode` episodes

### 3.4 Network topology

| Mode | When to use | Config |
|------|-------------|--------|
| **IPC** (default) | Learner and env runner on the same machine | `run_learner_on_server = False` |
| **TCP** | Learner and env runner on different machines | `run_learner_on_server = True`, set TCP endpoints |

### 3.5 Resuming RL training

To resume from a previous RL checkpoint:

1. Set `control_para["resume_rl"] = True` in the config
2. Ensure `$DICE_CHECKPOINT_FOLDERS/rl_finetuning/` contains the checkpoint files
3. Restart both learner and env runner

The env runner will auto-load the latest `checkpoint_XXXXX.pt` from the RL checkpoint directory.

**Important:** When starting a new RL run with a different BC checkpoint, set `resume_rl = False` and use a fresh `data_folder_path` to avoid loading stale checkpoints or reprocessing old episodes.

### 3.6 Key RL hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learner_para.num_episodes_before_first_training` | 20 | Episodes to collect before first training round |
| `learner_para.gradient_steps` | 2000 | Gradient updates per training round |
| `learner_para.update_every_x_episode` | 10 | Episodes between training rounds |
| `learner_para.batch_size` | 256 | Training batch size |
| `learner_para.expert_ratio` | 0.5 | Fraction of batch from expert data |
| `model_para.bc_loss_weight` | 100 | BC regularization weight |
| `model_para.rl_num_inference_steps` | 8 | DDIM steps during RL (can be fewer than BC for speed) |

---

## Configuration Reference

### Config files

| File | Purpose |
|------|---------|
| `dice_rl/config/rl_finetuning_config.py` | All RL finetuning parameters (shared between learner and env runner) |
| `diffusion_policy/config/train_dp_workspace.yaml` | BC training workspace config |
| `diffusion_policy/config/task/*.yaml` | Task-specific configs (dataset, shape_meta, horizons) |
| `hardware_interfaces/workcell/.../single_arm_data_collection.yaml` | Hardware config for **data collection** (low damping, low force limits for teleoperation) |
| `hardware_interfaces/workcell/.../belt_assembly.yaml` | Hardware config for **RL finetuning** (higher damping/stiffness for autonomous execution) |
| `prepare.sh` | Environment variables and conda activation |

**Data collection vs. RL finetuning hardware configs:** Data collection uses low damping and low force limits so the human can easily guide the robot. RL finetuning uses higher damping and higher force limits for autonomous execution. Create separate hardware configs for each mode — see the two examples above.

---

## Repository Structure

```
DICE-RL-Robot/
├── train.py                         # BC training entry point (Hydra)
├── prepare.sh                       # Environment setup script
├── pyproject.toml                   # Python package config
├── environment.yaml                 # Conda environment spec
│
├── dice_rl/                         # RL finetuning package
│   ├── config/                      # RL finetuning configuration
│   │   └── rl_finetuning_config.py
│   ├── learner/                     # RL learner (GPU server)
│   │   ├── distill_rl_learner.py    #   Training loop
│   │   └── run_learner.py           #   Entry point
│   ├── env_runner/                  # Environment runner (robot PC)
│   │   ├── rl_finetuning_env_runner.py  #   Policy execution, data collection
│   │   ├── manip_server_env.py      #   Robot environment abstraction
│   │   ├── manip_server_handle_env.py
│   │   └── run_env_runner.py        #   Entry point
│   ├── model/                       # RL model components
│   │   ├── distill_rl.py            #   DistilledActor, Critic
│   │   ├── distill_rl_img.py        #   Image-conditioned branch of RL model
│   │   └── common/                  #   MLP, critic ensembles, etc.
│   ├── replay_buffer/               #  replay buffer 
│   │   └── hybrid_replay_buffer.py
│   └── communication/               # ZMQ-based learner-actor communication
│       ├── learner_node.py
│       └── actor_node.py
│
├── utils/                           # Shared utilities (used by dice_rl + diffusion_policy)
│   ├── spatial_math.py              #   SE(3), rotation conversions
│   ├── model_io.py                  #   Load pretrained BC checkpoints
│   ├── mpc.py                       #   Model predictive controller
│   ├── common_type_conversions.py   #   Raw-to-obs-action conversion
│   ├── imagecodecs_numcodecs.py     #   JpegXL codec for zarr
│   └── data_processing/             #   Raw data to zarr pipeline
│
├── diffusion_policy/                # Vendored diffusion policy (BC backbone)
│   ├── config/                      #   Hydra configs (workspace, task)
│   ├── workspace/                   #   Training workspaces
│   ├── policy/                      #   Policy networks
│   ├── dataset/                     #   Dataset loaders (VirtualTargetDataset)
│   └── model/                       #   UNet, vision encoders, etc.
│
├── hardware_interfaces/             # C++ hardware drivers (git submodule)
│   ├── applications/               #   Data collection, calibration binaries
│   └── workcell/                   #   Hardware YAML configs
│
├── configs/                         # User-facing configs
│   ├── hardware/                   #   example_workcell.yaml
│   └── tasks/                      #   Task-specific parameters
│
├── scripts/                         # Shell and data processing scripts
    ├── process_raw_data.py          #   Raw episodes -> zarr conversion
    ├── run_learner.sh
    └── run_env_runner.sh

```

---


## Citation

If you find this codebase useful, consider citing:

```bibtex
@article{sun2026prior,
  title={From Prior to Pro: Efficient Skill Mastery via Distribution Contractive RL Finetuning},
  author={Sun, Zhanyi and Song, Shuran},
  journal={arXiv preprint arXiv:2603.10263},
  year={2026}
}
```

# Contact
If you have any questions, please feel free to contact [Zhanyi Sun](mailto:zhanyis@stanford.edu). If you leave an issue, please send me an accompanying email!

## License

MIT
