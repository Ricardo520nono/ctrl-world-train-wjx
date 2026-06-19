# S1-C 3:1:1:1 Family-Balanced 训练配置

这份文档描述当前最重要的主线训练：新版 head/left/right 三相机、5 个 S1 任务、expert + PCA + raw + random feasible 四类数据混合、3:1:1:1 family-balanced sampler。

对应一键入口：

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
bash scripts/launch_training.sh s1_c_3to1to1to1
```

EE trajectory auxiliary head ablation 入口：

```bash
bash scripts/launch_training.sh s1_c_ee_head
```

底层脚本：

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train/code/scripts/train_s1_c_expert_pca_raw_rf_family_balanced_headwrist.sh
```

## 1. 训练目标

训练一个 5-task Ctrl-World robot video world model。

模型输入：

- 历史视频 latent：`num_history=6`
- 未来动作序列：14D absolute joint action
- task text

模型输出：

- 未来视频 latent，经 VAE decode 后是 head/left/right 三相机竖排视频

这不是 policy 训练。Ctrl-World 是 action-conditioned world model：它学习“给定当前画面和未来动作，未来视频应该长什么样”。

## 2. 相机主线

当前主线只认新版 headwrist 三相机：

```text
head_camera,left_camera,right_camera
```

latent 竖向拼接顺序：

```text
head 在上
left wrist 在中
right wrist 在下
```

latent 形状：

```text
(T, 4, 90, 40)
```

其中单个相机 latent 高度是 30，三路相机拼起来高度是 90。

旧的 `front_camera,head_camera,left_camera` 不是当前主线。新增训练和推理评测都应优先使用 head/left/right。

## 3. 任务列表

S1-C 使用 5 个任务：

```text
click_alarmclock
click_bell
place_object_basket
open_laptop
stack_blocks_two
```

脚本中写成：

```bash
S1_TASKS="click_alarmclock click_bell place_object_basket open_laptop stack_blocks_two"
DATASET_NAMES="click_alarmclock+click_bell+place_object_basket+open_laptop+stack_blocks_two"
```

## 4. 数据 family

训练混合四类 family：

```text
expert
pca
raw
random_feasible
```

实际数据来源：

```text
expert:
  /mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/data_delta_ee/demo_clean_zed2i_visible

pca:
  /mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/EnhancedData/perturbed_pca_gaussian/c_8_sigma_0p05

raw:
  /mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/EnhancedData/perturbed_raw_gaussian/sigma_0p0025

random feasible:
  /mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/EnhancedData/random_feasible_300step_5task_2ep5start_formal_v1/random_feasible_random_walk/rf_5task_300step_2ep5start_formal_uniform_10seed_v1
  /mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/EnhancedData/random_feasible_300step_5task_2ep5start_formal_v1/random_feasible_random_walk/rf_5task_300step_2ep5start_formal_weighted_10seed_v1
```

`random_feasible` 在 sampler 里是一个 family，但底层合并了 uniform 和 weighted 两个 root。

## 5. latent cache

默认 latent cache 在：

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train/latents
```

脚本会使用或生成：

```text
precomputed_latents_s1A_5tasks_14d_headwrist
s1B_latents_pca_train_headwrist
s1C_latents_raw_s0025_train_headwrist
s1C_latents_rf300_uniform_train_headwrist
s1C_latents_rf300_weighted_train_headwrist
```

如果这些 latent 已经存在，预编码脚本会跳过已有任务；如果不存在，S1-C 启动脚本会自动预编码。

## 6. split

训练 split：

```text
source episodes 0-39
```

验证 split：

```text
source episodes 40-49
```

这和之前训练保持一致。

## 7. 采样方式

S1-C 使用 family-balanced sampler，不是简单把所有 windows 混进一个大池子里随机 shuffle。

两级采样逻辑：

```text
先采 family
再在 family 内采 task / trajectory / window
```

family 比例：

```bash
FAMILY_SAMPLING="expert=0.5,pca=0.166667,raw=0.166667,random_feasible=0.166667"
```

这对应：

```text
expert : pca : raw : random_feasible = 3 : 1 : 1 : 1
```

family dataset 虚拟长度：

```bash
FAMILY_DATASET_LENGTH=68429
```

这个值对应 chunk16 下的静态数量参考：

```text
expert           34214
pca              11405
raw              11405
random_feasible  11405
total            68429
```

## 8. 训练超参

核心训练参数：

```text
num_history: 6
num_frames / chunk size: 16
action_dim: 14
height: 240
train_batch_size: 1
gradient_accumulation_steps: 2
nproc_per_node: 8
effective global batch: 16
max_train_steps: 40000
learning_rate: 1e-5
mixed_precision: bf16
use_abs_joint_action: true
validation_steps: 2500
checkpointing_steps: 0
checkpointing_epochs: 1
```

checkpoint 保存逻辑：

- 不按固定 step 保存
- 每完成 1 个 epoch 保存一次
- 训练结束额外保存 `checkpoint-final-step40000.pt`

## 8.1 EE trajectory auxiliary head

`s1_c_ee_head` 复用 S1-C 主配置，并显式开启：

```bash
USE_EE_HEAD=1
EE_LOSS_WEIGHT=0.05
EE_HEAD_HIDDEN_DIM=256
```

模型从预测出的 future latent 接轻量 CNN + MLP head，逐未来帧预测左右臂 EE target：

```text
left  = xyz(3) + rotation_6d(6) + gripper(1)
right = xyz(3) + rotation_6d(6) + gripper(1)
total = 20
```

loss：

```text
position MSE + rotation 6D MSE + gripper BCE
```

目标由 RoboTwin HDF5 `endpose/*` 构造。新预编码会直接写入 `ee_target`；旧 latent cache 在 EE-head 入口下会先由 `scripts/backfill_ee_targets.py` 补齐。no-head S1-C baseline 不读取 `ee_target`。

## 9. 输出目录

默认输出目录：

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train/outputs/<RUN_NAME>
```

S1-C run name 格式：

```text
s1_C_3to1to1to1_family_balanced_headwrist_YYYYMMDD_HHMMSS
```

如果需要把 checkpoint 自动同步到推理目录，可以在训练启动后开 watcher：

```bash
SRC=/mnt/public_ckp/cscsx_projects/ctrl_world_train/outputs/<RUN_NAME> \
DST=/mnt/public_ckp/cscsx_projects/ctrl_world_infer/checkpoints/<TARGET_NAME> \
bash scripts/watch_and_copy_s1C_3to1to1to1_family_balanced_chunk16.sh
```

## 10. 推荐启动顺序

最简单方式：

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
bash scripts/install_ctrlworld_train_env.sh
bash scripts/launch_training.sh s1_c_3to1to1to1
```

如果要先做检查：

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
for f in scripts/*.sh; do bash -n "$f"; done
python3 -m py_compile scripts/train_delta_ee.py dataset/dataset_delta_ee.py dataset/dataset_delta_ee_family.py models/ctrl_world.py
python3 scripts/validate_s1c_family_pipeline.py --skip-real-inputs
```

完整真实数据检查需要相关 latent 已经存在：

```bash
python3 scripts/validate_s1c_family_pipeline.py
```

## 11. 给后续 Codex 的判断原则

如果之后要改训练需求，优先在当前 head/left/right 主线上改：

- 优先改 `scripts/train_s1_c_expert_pca_raw_rf_family_balanced_headwrist.sh`
- 保持 `CAMERAS="head_camera,left_camera,right_camera"`
- 保持 `dataset_meta_info` 指向 public 包内的 `code/dataset_meta_info`
- 大权重继续使用 public 包内 `assets/models`
- 新 latent 和新输出默认放在 public 包内 `latents/` 和 `outputs/`

除非用户明确要求复现旧版，否则不要回到旧的 front/head/left 相机配置。
