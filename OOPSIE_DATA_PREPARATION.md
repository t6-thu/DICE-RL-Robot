# Oopsie：已有 YAM 数据离线打标准备

本流程针对当前已有的 `episode_*.npz`，先转换成 Oopsie 的
`oopsiedata_format_v1`（每个 episode 一个 HDF5，并配套 wrist/base MP4），再用
浏览器逐条人工标注。原始 NPZ 全程只读；旧 `rewards` 不会被当成人工 Oopsie 标签。

## 当前环境

- 已固定 `oopsie-data-tools==1.0.1`（2026-08-25 发布；2026-09-02 核对为 PyPI 最新版）。
- 精确版本记录在 `requirements-oopsie.txt`。不要使用会顺带升级机器人环境中
  click/numpy/OpenCV 的 `pip install --upgrade`。
- Python 3.11 和系统 FFmpeg 可用。
- 已有数据位于 `~/文档/data/real_processed/yam_rl_rollouts_*`。
- 每条 episode 是 `(T, 6, 224, 224)` 双相机图像，以及 normalized 7 维
  state/action；转换时使用 Hanoi 对应的 `normalization.npz` 还原为 6 个 joint + gripper。

不要直接把四个 `yam_rl_rollouts_*` 目录全部转换：这些实验分支包含重复 episode。
先选定一个权威数据目录；顶层 rollout 和其 `eval/` 是否都要标注也应分批决定。

## 只需人工完成的三项

1. 如果尚未注册，先从 Oopsie 注册表获得准确的 lab ID 和 HuggingFace token，然后运行：

   ```bash
   . ./prepare.sh
   oopsie-data init
   ```

   让命令写入 `~/.config/oopsie-data/contributor_config.yaml`。不要在仓库里手写或保存 token。

2. 编辑 `robot_profiles/yam_hanoi.yaml`，填写这批数据真实的 `policy_name`。六个 joint
   名称/顺序已经按 i2rt 的 YAM 4310 linear 模型和 runner 的 `[:6]` 布局填好。

3. 转换时提供真实的 `--operator-name` 与 `--language-instruction`。启动 UI 时提供真实的
   `--annotator-name`。这些身份和标签不能由脚本猜测。

## 推荐执行顺序

以下示例把 `hanoi_sparse_terminal_v1` 顶层 episode 当作一批；请先按实际情况替换路径和文本。

```bash
. ./prepare.sh

# 新环境或需要重装时使用；不要加 --upgrade
uv pip install --python .venv/bin/python -r requirements-oopsie.txt
oopsie-data --version

SOURCE="$DICE_DATASET_FOLDERS/yam_rl_rollouts_hanoi_sparse_terminal_v1"
OUTPUT="$DICE_WORKSPACE_ROOT/data/oopsie/hanoi_sparse_terminal_v1"
PROFILE="robot_profiles/yam_hanoi.yaml"
NORM="$DICE_DATASET_FOLDERS/stack_green_hanoi_cube_224/normalization.npz"

python scripts/oopsie_preflight.py \
  --source "$SOURCE" \
  --output-dir "$OUTPUT" \
  --profile "$PROFILE" \
  --normalization "$NORM"
```

Preflight 全部通过后，先做只读 dry run：

```bash
python scripts/convert_yam_npz_to_oopsie.py \
  --source "$SOURCE" \
  --output-dir "$OUTPUT" \
  --profile "$PROFILE" \
  --normalization "$NORM" \
  --operator-name "<真实操作者>" \
  --language-instruction "<这批 episode 的真实任务指令>" \
  --max-episodes 3 \
  --dry-run
```

再实际转换一个 episode：

```bash
python scripts/convert_yam_npz_to_oopsie.py \
  --source "$SOURCE" \
  --output-dir "$OUTPUT" \
  --profile "$PROFILE" \
  --normalization "$NORM" \
  --operator-name "<真实操作者>" \
  --language-instruction "<这批 episode 的真实任务指令>" \
  --max-episodes 1
```

在浏览器中检查两路视频并打标：

```bash
oopsie-data annotate \
  --samples-dir "$OUTPUT" \
  --annotator-name "<真实标注者>" \
  --port 5001
```

打开 <http://localhost:5001>。确认画面身份、方向、速度及任务说明正确后，再去掉
`--max-episodes 1` 转换整批。脚本遇到已有 HDF5 会跳过，因此可安全续跑；它不会覆盖结果。

每轮打标完成后校验：

```bash
oopsie-data validate --path "$OUTPUT" --json
```

未打标文件出现 `Annotations dict is empty` 是预期行为；必须完成标注后才应全部通过。
不要使用跳过校验的方式上传。`upload` 会发布数据，本准备流程不会代你执行。

如需包含嵌套的 `eval/checkpoint_*`，显式加 `--recursive`。转换器只接受文件名严格匹配
`episode_<数字>.npz` 的文件，因此会排除 `.reward_cache` 和
`episode_*.robometer_rewards.npz` sidecar。

## 磁盘注意事项

当前根文件系统只剩约 23 GiB，而四套 rollout 各约 7 GiB。请一次只转换一个批次，
验证并备份后再安排下一批；不要在空间不足时同时生成四份视频/HDF5。

## 以后做边录制边打标

届时应在 `YAMRLEnvRunner` 控制循环内接入 Oopsie `EpisodeRecorder`，并用
`oopsie-data annotate --with-rollouts` 或 `WebRolloutAnnotator`。这会改变实时采集路径，
因此不包含在本次“已有数据离线打标”的改动里。
