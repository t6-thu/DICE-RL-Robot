# Hanoi Diffusion Policy + DICE-RL Finetuning 部署手册

本文档对应本仓库当前已经在 YAM 左臂上验证过的 Hanoi 流程：以 diffusion policy 为冻结的 base policy，使用 DICE-RL residual actor 做在线 finetuning，并使用 HiRE 生成 dense reward。

> 机器人会真实运动。启动 env runner 前，应确认急停可用、机械臂运动范围内无人且无障碍物，并安排一人始终观察机器人。不要在 learner 或其他程序仍占用机械臂时启动第二个控制进程。

## 1. 当前任务的固定配置

| 项目 | 当前值 |
| --- | --- |
| CAN channel | `can_follower_l` |
| Gripper | `linear_4310` |
| Wrist camera | `218622271309` |
| Base camera | `218622278369` |
| Spare camera | `218622274562` |
| Policy camera order | `wrist_base`，即 `rgb_0=wrist`、`rgb_1=base` |
| Image preprocessing | `center_crop` 到 224×224 |
| Observation horizon | 2 |
| Action horizon | 16 |
| Base DP checkpoint | `~/training_outputs/stack_green_hanoi_cube_dp_npz_retrain/checkpoints/latest.ckpt` |
| Expert dataset | `~/文档/data/real_processed/stack_green_hanoi_cube_224/train.npz` |
| Expert curation | 默认关闭，使用 `train.npz` 中全部 expert trajectories |
| Normalization | `~/文档/data/real_processed/stack_green_hanoi_cube_224/normalization.npz` |
| 默认 RL run | `hanoi_hire_npz_epoch0500_wristbase_v2` |
| Online episodes | `~/文档/data/real_processed/yam_rl_rollouts_<RUN_NAME>/` |
| RL checkpoints/log | `~/training_outputs/yam_rl_finetuning_<RUN_NAME>/` |

这些参数由 [launch_hanoi_hire.sh](scripts/launch_hanoi_hire.sh) 和 [yam_rl_config.py](dice_rl/config/yam_rl_config.py) 共同设置。Hanoi 应始终通过 `launch_hanoi_hire.sh` 启动，避免继承旧终端中的 Robometer reward 或错误的相机顺序。

## 2. 软件安装（新机器只需做一次）

推荐目录结构：

```text
~/文档/
├── DICE-RL-Robot/
└── i2rt/
```

安装系统依赖：

```bash
sudo apt update
sudo apt install -y build-essential python3-dev linux-headers-$(uname -r) can-utils
```

在本仓库创建 Python 3.11 环境并安装两个 editable package：

```bash
cd ~/文档/DICE-RL-Robot
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[learner,robot]"
uv pip install -e ../i2rt
```

验证环境：

```bash
cd ~/文档/DICE-RL-Robot
source ./prepare.sh

python - <<'PY'
import torch, i2rt, pyrealsense2, can
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("i2rt:", i2rt.__file__)
PY
```

第一次启动 HiRE learner 时，DINOv2 可能通过 `torch.hub` 下载一次模型，因此首次运行需要联网。之后会使用本机缓存。

## 3. 每次开机后的完整 preflight

以下检查应在启动 learner/env runner 之前完成。

### 3.1 检查 GPU、内存和残留进程

```bash
nvidia-smi
free -h
ps -eo pid,rss,%mem,cmd --sort=-rss | \
  grep -E 'yam_rl_run_(learner|env_runner)' | grep -v grep || true
```

不要同时运行两份 learner 或两份 env runner。当前 compact online replay 会按 episode/index 引用图像，不再为每个 transition 重复复制 observation；恢复 60 个 episode 时 learner 的空闲 RSS 实测约 12 GB。若启动后 RSS 很快持续超过 30 GB，应先停止并检查是否运行了旧代码。

可验证 compact replay 的回归测试：

```bash
cd ~/文档/DICE-RL-Robot
source ./prepare.sh
python -m unittest tests.test_yam_replay_buffer_boundaries -v
```

### 3.2 加载 CAN 驱动并启动接口

```bash
sudo modprobe -a can can_raw can_dev gs_usb

ip -brief link | grep -E 'can|yam'
sudo ip link set can_follower_l down 2>/dev/null || true
sudo ip link set can_follower_l up type can bitrate 1000000

ip -details -statistics link show can_follower_l
```

正确状态应包含：

```text
state UP
can state ERROR-ACTIVE
bitrate 1000000
bus-off 0
```

如果没有 `can_follower_l`，先运行 `ip -brief link` 查看 CANable 实际名字，并修复 udev 的稳定命名；不要临时把训练配置指向不确定的接口。

检查 CANable 是否和高带宽相机共用 USB hub：

```bash
bash scripts/check_can_isolation.sh
```

这是一项辅助检查；机器连接多个 CANable 时，还应确认脚本打印的 USB device 确实对应 `can_follower_l`，不能用另一个 CANable 的结果代替。

出现 `loss communication` 时应立即停止 env runner。这通常是 CAN/USB 通信或供电问题，并不等同于电机过热；先检查 `ip -details -statistics link show can_follower_l` 的错误计数，再重新插拔 CANable、重新启动接口，必要时给机械臂断电重启。

### 3.3 确认相机身份和画面

```bash
cd ~/文档/DICE-RL-Robot
source ./prepare.sh

python - <<'PY'
import pyrealsense2 as rs
for d in rs.context().query_devices():
    print(d.get_info(rs.camera_info.serial_number),
          d.get_info(rs.camera_info.usb_type_descriptor))
PY

python scripts/inspect_cams.py
```

检查生成的 `base.jpg` 和 `wrist.jpg`：

- `218622271309` 必须是 wrist view。
- `218622278369` 必须是 base view。
- 两张图都应曝光正常、没有镜头盖、画面方向正确。
- 优先让两个工作相机运行在 USB 3.x。若某个设备显示 2.1 但能稳定提供新帧，可以暂时运行；一旦出现 stale camera 或 CAN timing 问题，应先调整 USB 端口/控制器。

不要因为交换 USB 插口而修改 serial mapping；RealSense serial 属于相机本体，不属于 USB 端口。

### 3.4 确认 base policy、expert data 和 normalization

```bash
for p in \
  "$HOME/training_outputs/stack_green_hanoi_cube_dp_npz_retrain/checkpoints/latest.ckpt" \
  "$HOME/文档/data/real_processed/stack_green_hanoi_cube_224/train.npz" \
  "$HOME/文档/data/real_processed/stack_green_hanoi_cube_224/normalization.npz"; do
  test -f "$p" && ls -lh "$p" || echo "MISSING: $p"
done
```

这三个文件必须来自同一套 Hanoi 数据处理/训练流程。不要用别的任务的 normalization，也不要把 `wrist_base` 改成 `base_wrist`。

### 3.5 可选：在开始 RL 前单独验证 base DP

先只回 home；该命令完成后正常退出并关闭 torque 是预期行为：

```bash
cd ~/文档/DICE-RL-Robot
source ./prepare.sh

python scripts/eval_dp_yam.py \
  --ckpt "$HOME/training_outputs/stack_green_hanoi_cube_dp_npz_retrain/checkpoints/latest.ckpt" \
  --norm "$HOME/文档/data/real_processed/stack_green_hanoi_cube_224/normalization.npz" \
  --can_channel can_follower_l \
  --gripper_type linear_4310 \
  --base_serial 218622278369 \
  --wrist_serial 218622271309 \
  --policy_camera_order wrist_base \
  --weights ema \
  --home_joint_pos=-0.010,0.833,0.903,-0.598,-0.028,-0.029 \
  --home_only \
  --ramp_seconds 15
```

需要重新跑一条 base-policy rollout 时，去掉 `--home_only`，并设置有限的 `--num_episodes 1 --max_steps 60 --print_actions`。操作员必须在旁边随时准备中止。

## 4. 选择“全新实验”或“断点恢复”

### 4.1 全新实验

选择一个从未使用过的 run name，并在 learner 和 env runner 两个终端中设置相同的值：

```bash
export YAM_HANOI_RUN_NAME=hanoi_hire_20260803_v1
```

不要复用旧 run name，也不需要删除任何旧目录。新 run 会自动使用：

```text
~/文档/data/real_processed/yam_rl_rollouts_hanoi_hire_20260803_v1/
~/training_outputs/yam_rl_finetuning_hanoi_hire_20260803_v1/
```

全新实验前 20 个 episode 没有 residual actor，因此由 pure BC 收集。第 20 个 episode 被 learner 读入后触发第一次 2000-step 训练；之后每新增 10 个 episode 训练 1000 steps。

### 4.2 恢复已有实验

已有 Hanoi run 不要设置新的名字，或显式设置：

```bash
export YAM_HANOI_RUN_NAME=hanoi_hire_npz_epoch0500_wristbase_v2
```

learner 会：

1. 读取该 run 的全部 `episode_*.npz`。
2. 自动加载 checkpoint 目录里编号最大的 `checkpoint_*.pt`。
3. 用磁盘上的 episode 数同步 `total_episodes`。
4. 重新生成 `latest_actor.pt`，供 env runner 在下一条 rollout 前加载。

检查当前恢复点：

```bash
RUN_NAME=hanoi_hire_npz_epoch0500_wristbase_v2

find "$HOME/文档/data/real_processed/yam_rl_rollouts_${RUN_NAME}" \
  -maxdepth 1 -type f -name 'episode_*.npz' | wc -l
ls -1 "$HOME/training_outputs/yam_rl_finetuning_${RUN_NAME}"/checkpoint_*.pt | tail
```

文档编写时，该 run 已有 60 个 episode，最新 checkpoint 是 `checkpoint_006000.pt`。因此现在正常重启会从 6000 steps 恢复，而不是从 `checkpoint_003000.pt` 恢复。

> 不要直接删除较新的 checkpoint 来强制回滚。checkpoint、rollout episode 数和训练轮次是配套的；若要做回滚对照实验，应复制所需数据到一个新的 run name 后再单独规划。

## 5. 启动 finetuning

learner 和 env runner 在当前配置下运行在同一台机器，通过磁盘 episode 文件和 `latest_actor.pt` 协作。应先启动 learner，确认恢复完成，再启动 env runner。

### 5.1 Terminal 1：启动 learner

恢复默认 Hanoi run：

```bash
cd ~/文档/DICE-RL-Robot
export YAM_HANOI_RUN_NAME=hanoi_hire_npz_epoch0500_wristbase_v2
bash scripts/launch_hanoi_hire.sh learner
```

全新 run 则把 `YAM_HANOI_RUN_NAME` 换成第 4.1 节的新名字。

启动日志应确认以下内容：

```text
reward=hire
camera_order=wrist_base preprocess=center_crop
Online buffer restored: ... compact episode arrays: ... GiB
Resuming from checkpoint: .../checkpoint_XXXXXX.pt    # 新 run 没有这一行
Synced total_episodes=... from disk replay buffer
Learner running. Polling ... every 2 s for new episodes
```

如果日志显示 `reward=robometer`、`camera_order=base_wrist` 或 expert/normalization 路径不是 Hanoi，立即退出，不要启动机器人。

learner 日志同时写入：

```text
~/training_outputs/yam_rl_finetuning_<RUN_NAME>/learner.log
```

实时查看：

```bash
tail -F "$HOME/training_outputs/yam_rl_finetuning_${YAM_HANOI_RUN_NAME}/learner.log"
```

当前 YAM learner 的权威训练记录是该本地日志和 checkpoint；这条路径目前没有自动写入 W&B。

### 5.2 Terminal 2：启动 env runner

```bash
cd ~/文档/DICE-RL-Robot
export YAM_HANOI_RUN_NAME=hanoi_hire_npz_epoch0500_wristbase_v2

YAM_MAX_EPISODE_STEPS=60 \
bash scripts/launch_hanoi_hire.sh envrunner \
  --residual-scale 0.1 \
  --max-joint-step 0.08
```

参数含义：

- `YAM_MAX_EPISODE_STEPS=60`：最多执行 60 个 diffusion-query chunks，而不是 60 个 30 Hz frame。
- `--residual-scale 0.1`：只应用 10% 的 RL residual，适合当前早期 finetuning。
- `--max-joint-step 0.08`：限制相邻 30 Hz 命令的最大关节变化。

启动时先确认日志：

```text
Policy camera order: wrist_base (rgb_0=wrist rgb_1=base)
Image preprocessing: center_crop
Loading BC policy from ...stack_green_hanoi_cube_dp_npz_retrain...
both cameras streaming
```

程序随后要求输入：

```text
Type MOVE to command a 6.0s ramp to home:
```

确认路径安全后输入完全一致的大写 `MOVE`。到 home 后，程序会在 episode prompt 等待 Enter，不会像 `--home_only` 那样退出。

若恢复 checkpoint，还应看到：

```text
★ Actor weights updated (step=6000, latest_actor.pt)
[Episode ... | RL actor step=6000]
```

如果显示 `pure BC`：

- 对全新 run 的前 20 个 episode，这是正常行为。
- 对已有 checkpoint 的 run，先不要开始 episode；检查 learner 是否已生成 `latest_actor.pt`，以及两个终端的 run name 是否一致。

## 6. 每个 episode 的操作

1. 摆好 Hanoi 物体，确认与训练数据的初始分布一致。
2. 在 `[Episode N | ...]` prompt 按 Enter。
3. 始终观察手臂和夹爪；异常时按 `Ctrl-C` 中止当前 episode。
4. rollout 结束后严格标记：
   - `s`：确实完成抓取/任务成功。
   - `f`：没有完成任务。
   - `d`：本条无效，不保存，例如人为干预、相机遮挡或硬件异常。
5. 保存后机器人自动回 home，再准备下一条。

不要把“接近物体”标成成功。HiRE 的 online positive/negative buffer 依赖这些标签，错误标签会直接污染 reward shaping。

env runner 每条有效 episode 保存：

```text
~/文档/data/real_processed/yam_rl_rollouts_<RUN_NAME>/episode_NNNN.npz
```

图像以 `uint8` 保存；online replay 只记录 `(episode_id, transition_index)` 引用，sample minibatch 时才构造 observation，因此不会再为每个 transition 复制两套 image history。

## 7. 训练轮次与 checkpoint

默认 schedule：

| Episode 数 | 动作 |
| --- | --- |
| 0–19 | pure BC 收集 warmup episodes |
| 20 | 第一次训练 2000 gradient steps |
| 30 | 再训练 1000 steps，累计 3000 |
| 40 | 再训练 1000 steps，累计 4000 |
| 50 | 再训练 1000 steps，累计 5000 |
| 60 | 再训练 1000 steps，累计 6000 |

每轮先看到：

```text
Training round (episode ...): expected=... done=... → training…
Pre-encoding pool ...
Pool ready. Running ... MLP-only gradient steps…
```

完成后生成：

```text
checkpoint_XXXXXX.pt   # actor + critics + counters，用于 learner 恢复
latest_actor.pt        # env runner 自动加载的 residual actor
```

第一轮 pre-encoding/training 时建议监控：

```bash
watch -n 2 'ps -C python -o pid,rss,%mem,cmd --sort=-rss | head; free -h; nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv,noheader'
```

env runner 会在每个 episode 开始前重新检查 `latest_actor.pt`。learner 完成一轮训练后，不需要重启 env runner；新 actor 会从下一条 episode 生效。

## 8. 安全停止与再次恢复

- 在 episode 内按一次 `Ctrl-C`：请求中止当前 episode。
- 在等待 prompt 时按 `Ctrl-C`：安全关闭相机和机器人 torque。
- `Ctrl-\\` 是立即 hard kill，只在正常关闭完全失效时使用；它不会保证回 home。
- 先停止 env runner，再停止 learner。

再次启动时使用相同 `YAM_HANOI_RUN_NAME`，按第 3 节完成 preflight，然后重复第 5 节即可。不要运行 `scripts/setup_new_run.sh`，因为它会提示清空 rollout/checkpoint 目录，不适用于恢复现有 Hanoi run。

## 9. 常见故障

### `Network is down`

`can_follower_l` 存在但没有 UP。重新执行：

```bash
sudo ip link set can_follower_l down 2>/dev/null || true
sudo ip link set can_follower_l up type can bitrate 1000000
ip -details -statistics link show can_follower_l
```

### `motor id: ..., loss communication`

立即停止控制，不要继续 rollout。检查 CAN state、USB hub/供电、CANable 插头和机械臂电源。通信丢失不是“策略问题”，也不能仅凭该错误判断过热。

### `stale camera`

检查对应 serial 是否仍在、USB speed、线缆和 hub。不要放宽 `YAM_MAX_CAMERA_AGE` 来掩盖停止更新的相机。

### 回 home 后程序退出

如果运行的是 `eval_dp_yam.py --home_only`，这是正常行为。连续 rollout 应使用 `launch_hanoi_hire.sh envrunner`。

### Episode 因时间到达上限而结束

提高 chunk 上限，例如：

```bash
YAM_MAX_EPISODE_STEPS=80 bash scripts/launch_hanoi_hire.sh envrunner \
  --residual-scale 0.1 --max-joint-step 0.08
```

不要在程序已经运行后才 export；环境变量必须写在启动命令前，或先在当前终端 export。

### learner 恢复了错误的 checkpoint

检查两个终端的 run name：

```bash
echo "$YAM_HANOI_RUN_NAME"
```

learner 总是加载该 run checkpoint 目录中编号最大的 `checkpoint_*.pt`。如果想开始新实验，使用新的 run name，不要清空或覆盖旧实验。

### env runner 显示 `pure BC`

已有 checkpoint 时，确认 learner 已启动完成且下面文件存在：

```bash
ls -lh "$HOME/training_outputs/yam_rl_finetuning_${YAM_HANOI_RUN_NAME}/latest_actor.pt"
```

然后回到 env runner prompt；它会在下一条 episode 开始前再次读取权重。

## 10. 最短恢复命令清单

完成 CAN、相机和安全检查后：

Terminal 1：

```bash
cd ~/文档/DICE-RL-Robot
export YAM_HANOI_RUN_NAME=hanoi_hire_npz_epoch0500_wristbase_v2
bash scripts/launch_hanoi_hire.sh learner
```

Terminal 2：

```bash
cd ~/文档/DICE-RL-Robot
export YAM_HANOI_RUN_NAME=hanoi_hire_npz_epoch0500_wristbase_v2
YAM_MAX_EPISODE_STEPS=60 \
bash scripts/launch_hanoi_hire.sh envrunner \
  --residual-scale 0.1 \
  --max-joint-step 0.08
```
