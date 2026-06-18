# 训练配置与启动入口

以下命令默认在这个目录下运行：

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
```

## 通用约定

- 训练机器：百度云 8 卡节点。
- 本地单卡机器只用于开发和推理调试，不用于 8 卡训练。
- 相机顺序：`head_camera,left_camera,right_camera`。
- headwrist latent 形状：`(T, 4, 90, 40)`。
- action dim：`14`。
- action 类型：absolute joint action。
- 训练 split：source episodes `0-39`。
- 验证 split：source episodes `40-49`。
- 默认 latent cache：`/mnt/public_ckp/cscsx_projects/ctrl_world_train/latents`。
- 默认训练输出：`/mnt/public_ckp/cscsx_projects/ctrl_world_train/outputs`。

## all50_headwrist

启动：

```bash
bash scripts/launch_training.sh all50_headwrist
```

底层脚本：

```bash
scripts/train_ctrlworld_8gpu_delta_ee_all50_nf16_60k_headwrist.sh
```

配置：

- 任务：ActionFollowingBench 全 50 个任务。
- 数据 family：expert only。
- 训练步数：`60000`。
- chunk size：`16`。
- 等效 global batch：`8 GPUs * train_batch_size 1 * grad_accum 2 = 16`。
- 保存：每个完整 epoch 保存一次，最后额外保存 final checkpoint。

## s1_a_expert

启动：

```bash
bash scripts/launch_training.sh s1_a_expert
```

底层脚本：

```bash
scripts/train_s1_a_expert_only_headwrist.sh
```

配置：

- 任务：`click_alarmclock`、`click_bell`、`place_object_basket`、`open_laptop`、`stack_blocks_two`。
- 数据 family：expert only。
- 训练步数：`40000`。
- chunk size：`16`。
- 保存：每个完整 epoch 保存一次，最后额外保存 final checkpoint。

## s1_b_expert_pca

启动：

```bash
bash scripts/launch_training.sh s1_b_expert_pca
```

底层脚本：

```bash
scripts/train_s1_b_expert_sliding_pca_single_headwrist.sh
```

配置：

- 任务：同 S1 的 5 个任务。
- 数据 family：expert + PCA perturbation。
- PCA 数据：`EnhancedData/perturbed_pca_gaussian/c_8_sigma_0p05`。
- 训练步数：`40000`。
- chunk size：`16`。

## s1_c_3to1to1to1

启动：

```bash
bash scripts/launch_training.sh s1_c_3to1to1to1
```

底层脚本：

```bash
scripts/train_s1_c_expert_pca_raw_rf_family_balanced_headwrist.sh
```

这是本次整理最重要的参考配置，对应 2026-06-10 成功跑完的 S1-C 训练。

这条入口已经按 public 训练包做过路径整理，不依赖个人目录。它会：

1. 安装/检查 Python 依赖。
2. 使用 public 包内的 SVD/CLIP 权重。
3. 从 public 数据目录读取 expert/PCA/raw/random feasible 数据。
4. 按 head/left/right 三相机预编码 latent，已有 latent 会跳过。
5. 计算或复用 `s1_C_expert_pca_raw_rf_3to1to1to1/stat.json`。
6. 用 family-balanced sampler 启动 8 卡训练。

配置：

- 任务：同 S1 的 5 个任务。
- 数据 family：
  - expert
  - PCA，`c_8_sigma_0p05`
  - raw Gaussian，`sigma_0p0025`
  - random feasible 300-step，其中 uniform 和 weighted 两个 root 合并为一个 family
- 采样方式：family-balanced sampler。
- 采样比例：`3:1:1:1`。
- 脚本里的实际值：

```bash
FAMILY_SAMPLING="expert=0.5,pca=0.166667,raw=0.166667,random_feasible=0.166667"
FAMILY_DATASET_LENGTH=68429
```

- 训练步数：`40000`。
- chunk size：`16`。
- 保存：每个完整 epoch 保存一次，最后额外保存 final checkpoint。

## s1_a_single_task

启动一个单任务训练：

```bash
bash scripts/launch_training.sh s1_a_single_task place_object_basket
```

可选任务：

```text
click_alarmclock
click_bell
place_object_basket
open_laptop
stack_blocks_two
```

配置：

- 数据 family：expert only。
- 每个 run 只训练一个 task。
- 训练步数：`40000`。
- chunk size：`16`。

## checkpoint 自动复制 watcher

watcher 脚本仍然保留。它们会把完整 checkpoint 复制到推理 checkpoint 目录。复制时先写 `.partial` 临时文件，再原子 rename，避免推理端读到半成品 checkpoint。

新训练建议显式传 `SRC` 和 `DST`：

```bash
SRC=/mnt/public_ckp/cscsx_projects/ctrl_world_train/outputs/<RUN_NAME> \
DST=/mnt/public_ckp/cscsx_projects/ctrl_world_infer/checkpoints/<TARGET_NAME> \
bash scripts/watch_and_copy_s1C_3to1to1to1_family_balanced_chunk16.sh
```
