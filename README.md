# Ctrl-World 训练包

这是整理后的 Ctrl-World headwrist 机器人视频世界模型训练交接包。

共享盘根目录：

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train
```

这个目录的目标是让后续 Codex 可以直接从共享路径接手：安装依赖、预编码 latent、计算 action stat、启动百度云 8 卡训练，并把 checkpoint 同步到推理目录。

## 目录内容

- `code/`：Ctrl-World 模型、数据集、训练、预编码、stat、watcher 和启动脚本。
- `assets/models/`：训练使用的本地 SVD 和 CLIP 权重。该目录存在于共享盘，但不会提交到 GitHub。
- `docs/`：训练交接、实验配置和操作说明。
- `.env.sample`：W&B 密钥和可选路径覆盖的模板。

## 快速开始

先创建环境变量文件：

```bash
cp /mnt/public_ckp/cscsx_projects/ctrl_world_train/.env.sample \
   /mnt/public_ckp/cscsx_projects/ctrl_world_train/.env
```

然后在下面这个文件里填写 `WANDB_API_KEY`：

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train/.env
```

安装依赖：

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
bash scripts/install_ctrlworld_train_env.sh
```

启动主实验 S1-C，也就是复现 2026-06-10 那次成功的 3:1:1:1 family-balanced 训练：

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
bash scripts/launch_training.sh s1_c_3to1to1to1
```

## 主要训练入口

```bash
bash scripts/launch_training.sh all50_headwrist
bash scripts/launch_training.sh s1_a_expert
bash scripts/launch_training.sh s1_b_expert_pca
bash scripts/launch_training.sh s1_c_3to1to1to1
bash scripts/launch_training.sh s1_a_single_task place_object_basket
```

## 默认路径

默认路径统一定义在：

```bash
code/scripts/ctrlworld_train_env.sh
```

几个关键默认值：

```bash
TRAIN_PACKAGE_ROOT=/mnt/public_ckp/cscsx_projects/ctrl_world_train
PROJECT_ROOT=${TRAIN_PACKAGE_ROOT}/code
ASSET_ROOT=${TRAIN_PACKAGE_ROOT}/assets/models
CACHE_ROOT=${TRAIN_PACKAGE_ROOT}/latents
OUTPUT_ROOT=${TRAIN_PACKAGE_ROOT}/outputs
```

原始数据仍然使用共享数据目录：

```bash
/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench
```

已有推理 checkpoint 仍然放在：

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_infer/checkpoints
```

## 参考实验

这次整理以成功跑完的 S1-C family-balanced 训练为参考。原始启动命令是：

```bash
bash /mnt/gyc/Ctrl-World/scripts/train_s1_c_expert_pca_raw_rf_family_balanced_headwrist.sh
```

整理后的等价入口是：

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
bash scripts/launch_training.sh s1_c_3to1to1to1
```

这版训练配置：

- 5 个任务：`click_alarmclock`、`click_bell`、`place_object_basket`、`open_laptop`、`stack_blocks_two`
- 4 类数据 family：expert、PCA、raw Gaussian、random feasible
- family 采样比例：`3:1:1:1`，脚本中写成 `0.5, 0.166667, 0.166667, 0.166667`
- 相机：`head_camera,left_camera,right_camera`
- chunk size：`num_frames=16`
- history：`num_history=6`
- 训练步数：`40000`
- 训练方式：百度云 8 卡分布式

更多说明见：

```bash
docs/TRAINING_RECIPES.md
docs/CODEX_HANDOFF.md
docs/s1_c_family_balanced_training_config.md
```
