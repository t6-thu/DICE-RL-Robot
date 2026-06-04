# Robometer-4B reward for YAM DICE-RL finetune

## 真机当前用哪路相机？

| 用途 | 代码里的 key | episode `images` 通道 | RealSense serial (`yam_rl_config.py`) |
|------|-------------|----------------------|--------------------------------------|
| 固定第三人称 / 台面 | `rgb_0` | `[:, 0:3]` | `CAMERAS["base_cam_serial"]` → `HARDWARE["base_cam_serial"]` |
| 腕部 | `rgb_1` | `[:, 3:6]` | `CAMERAS["wrist_cam_serial"]` |

- **策略 / critic**：两路都用（`rgb_0` + `rgb_1`）。
- **HiRE dense reward**：两路 DINO 特征都算。
- **Robometer dense reward（默认）**：只用 **`base`**（= `rgb_0` = 固定相机），与 HiRE-Dice 里 `sideview_image` / `agentview_image` 一类第三人称视角对应；**不是** `wrist`。
- **Robometer 公式（默认 relative）**：`reward_weight * (progress_{t+H} - progress_t)`（LIBERO `use_relative_rewards`）；绝对 progress PBRS 需设 `robometer_use_relative_rewards=False`。

Episode 写入见 `yam_rl_env_runner.py`：`np.concatenate([base_preprocess, wrist_preprocess], axis=0)` → `(6, 224, 224)`。

## HiRE-Dice `FINETUNE_GLOBAL_OVERRIDES` → 真机怎么设

仿真（Hydra）示例：

```bash
FINETUNE_GLOBAL_OVERRIDES="expert_dataset.max_n_episodes=20 \
  env.wrappers.robomimic_image.robometer_query_every_n_chunks=4 \
  env.wrappers.robomimic_image.robometer_query_fill_mode=hold \
  env.wrappers.robomimic_image.robometer_max_batch_size=4 \
  env.wrappers.robomimic_image.robometer_camera_key=sideview_image"
```

| HiRE-Dice (Hydra) | 真机 YAM |
|-------------------|----------|
| `expert_dataset.max_n_episodes=20` | `HIRE_EXPERT_CURATION_PATH` 的 JSON `include` 列表，或改 `expert.npz` |
| `train.eval_freq=200` | 真机 pipeline **无** 在线 sim eval；可忽略 |
| `robometer_query_every_n_chunks=4` | `yam_rl_config.py` → `robometer_query_every_n_chunks`，或 env `YAM_ROBOMETER_QUERY_EVERY_N_CHUNKS=4` |
| `robometer_query_fill_mode=hold` | `robometer_query_fill_mode` / `YAM_ROBOMETER_QUERY_FILL_MODE=hold` |
| `robometer_max_batch_size=4` | `robometer_max_batch_size` / `YAM_ROBOMETER_MAX_BATCH_SIZE=4` |
| `robometer_batch_env_queries` | 真机按 **episode 批量** HTTP，无 multi-env；已由 `max_batch_size` 控制 |
| `robometer_parallel_requests=4` | 真机 episode shaper 串行 checkpoint 查询（episode 数少，一般够用） |
| `robometer_camera_key=sideview_image` | **`YAM_ROBOMETER_CAMERA=base`**（或 `sideview_image` / `agentview_image` 别名，见下） |
| `robometer_use_relative_rewards=true` | **默认已开**；关闭：`YAM_ROBOMETER_USE_RELATIVE_REWARDS=false` |

### 相机 ID（RealSense serial）

在 `dice_rl/config/yam_rl_config.py` 的 `CAMERAS` / `HARDWARE` 里改，或启动 env runner 时用环境变量：

```bash
export YAM_BASE_CAM_SERIAL=218622278369
export YAM_WRIST_CAM_SERIAL=218622271309
python scripts/yam_rl_run_env_runner.py
```

列出本机 serial：`rs-enumerate-devices -s`

### Robometer 只看哪路 + 其它超参（learner / GPU）

编辑 `dice_rl/config/yam_rl_config.py` 的 `TRAINING`，或：

```bash
export YAM_REWARD_MODE=robometer
export ROBOMETER_SERVER_URL=http://127.0.0.1:8000
export YAM_ROBOMETER_CAMERA=base          # 默认；腕部进度用 wrist
export YAM_ROBOMETER_QUERY_EVERY_N_CHUNKS=4
export YAM_ROBOMETER_QUERY_FILL_MODE=hold
export YAM_ROBOMETER_MAX_BATCH_SIZE=4
export YAM_ROBOMETER_REWARD_WEIGHT=1.0
export YAM_ROBOMETER_TASK_INSTRUCTION="Pick up the Arizona bottle and place it in the target location."

bash scripts/launch_yam_rl_robometer.sh learner
```

`YAM_ROBOMETER_CAMERA` 合法值：`base`, `wrist`, `rgb_0`, `rgb_1`, `sideview_image`, `agentview_image`（后两者映射到 `base`）。

### 一键对齐你那条 HiRE-Dice overrides

```bash
export YAM_REWARD_MODE=robometer
export YAM_ROBOMETER_QUERY_EVERY_N_CHUNKS=4
export YAM_ROBOMETER_QUERY_FILL_MODE=hold
export YAM_ROBOMETER_MAX_BATCH_SIZE=4
export YAM_ROBOMETER_CAMERA=base   # sideview_image 在真机 ≡ base / rgb_0
# 相机 serial 在机器人端：
export YAM_BASE_CAM_SERIAL=你的固定相机serial
export YAM_WRIST_CAM_SERIAL=你的腕部serial
```

## Quick start

```bash
. ./prepare.sh
bash scripts/robometer/start_eval_server.sh
bash scripts/launch_yam_rl_robometer.sh learner
bash scripts/launch_yam_rl_robometer.sh env
```
