# Codex 接手说明

这份文档是给后续接手 Ctrl-World 训练任务的 Codex 看的。

## 优先阅读

先读：

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train/README.md
/mnt/public_ckp/cscsx_projects/ctrl_world_train/docs/TRAINING_RECIPES.md
```

当前主线是新版 headwrist 三相机：

```text
head_camera,left_camera,right_camera
```

后续新增训练、预编码、推理评测都应该优先沿用这个相机顺序。旧的 `front/head/left` 只作为历史背景，不是默认入口。

翔哥不需要访问 `/mnt/gyc` 或 `/mnt/gyc_ckp`。当前训练包的代码、文档、stat、小权重和大模型权重都已经迁移到：

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train
```

## 代码结构

```text
ctrl_world_train/
  code/
    dataset/
    dataset_meta_info/
    models/
    scripts/
  assets/models/
  docs/
```

其中：

- GitHub 仓库保存轻量代码和文档。
- `/mnt/public_ckp/cscsx_projects/ctrl_world_train/assets/models/` 保存大模型权重，不进 GitHub。
- `/mnt/public_ckp/cscsx_projects/ctrl_world_train/latents/` 是预编码 latent cache，按需生成，不进 GitHub。
- `/mnt/public_ckp/cscsx_projects/ctrl_world_train/outputs/` 是训练输出目录，不进 GitHub。

核心文件：

```bash
code/scripts/train_delta_ee.py
code/dataset/dataset_delta_ee.py
code/dataset/dataset_delta_ee_family.py
code/scripts/precompute_latents_delta_ee.py
code/scripts/precompute_latents_s1_pca.py
code/scripts/backfill_ee_targets.py
code/scripts/compute_stat_s1.py
code/scripts/compute_stat_family_roots.py
code/models/ctrl_world.py
code/models/ee_head.py
```

## 环境准备

从模板创建环境变量文件：

```bash
cp /mnt/public_ckp/cscsx_projects/ctrl_world_train/.env.sample \
   /mnt/public_ckp/cscsx_projects/ctrl_world_train/.env
```

必须填写：

```bash
WANDB_API_KEY=...
```

安装依赖：

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
bash scripts/install_ctrlworld_train_env.sh
```

## 主训练一键入口

当前推荐的 S1-C 训练：

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
bash scripts/launch_training.sh s1_c_3to1to1to1
```

这是当前主线训练入口。它会从 public 数据目录读取原始数据和增强数据，把 latent cache 与训练输出写到 public 训练包内。

S1-C 脚本会自动准备以下 latent cache：

```text
precomputed_latents_s1A_5tasks_14d_headwrist
s1B_latents_pca_train_headwrist
s1C_latents_raw_s0025_train_headwrist
s1C_latents_rf300_uniform_train_headwrist
s1C_latents_rf300_weighted_train_headwrist
```

如果对应任务的 `meta.json` 已经存在，预编码会跳过；如果不存在，会重新从 public 数据目录编码。

## 路径覆盖

默认路径在这里定义：

```bash
code/scripts/ctrlworld_train_env.sh
```

常用覆盖项：

```bash
PYTHON_BIN=/usr/bin/python3
CACHE_ROOT=/mnt/public_ckp/cscsx_projects/ctrl_world_train/latents
OUTPUT_ROOT=/mnt/public_ckp/cscsx_projects/ctrl_world_train/outputs
```

例子：

```bash
OUTPUT_ROOT=/mnt/public_ckp/cscsx_projects/ctrl_world_train/outputs_test \
bash scripts/launch_training.sh s1_a_expert
```

## 大规模 8 卡训练前的检查

先确认必要文件都在：

```bash
test -f /mnt/public_ckp/cscsx_projects/ctrl_world_train/assets/models/stable-video-diffusion-img2vid/model_index.json
test -d /mnt/public_ckp/cscsx_projects/ctrl_world_train/assets/models/stable-video-diffusion-img2vid/vae
test -f /mnt/public_ckp/cscsx_projects/ctrl_world_train/assets/models/clip-vit-base-patch32/config.json
test -f /mnt/public_ckp/cscsx_projects/ctrl_world_train/code/dataset_meta_info/s1_C_expert_pca_raw_rf_3to1to1to1/stat.json
```

检查 shell 脚本语法：

```bash
for f in scripts/*.sh; do bash -n "$f"; done
```

检查 Python 编译：

```bash
python3 -m py_compile \
  scripts/train_delta_ee.py \
  dataset/dataset_delta_ee.py \
  dataset/dataset_delta_ee_family.py \
  models/ctrl_world.py
```

不依赖真实 latent，只检查 family sampler 接线：

```bash
python3 scripts/validate_s1c_family_pipeline.py --skip-real-inputs
```

检查 EE head / loss / target shape：

```bash
python3 scripts/validate_ee_head_smoke.py
python3 scripts/validate_ee_head_forward_mock.py
```

如果本机有 SVD / CLIP 权重，可以再跑真实单卡 forward：

```bash
python3 scripts/validate_ee_head_real_forward.py \
  --svd_model_path /mnt/dataset/public_data/cscsx_projects/ctrl_world_train/assets/models/stable-video-diffusion-img2vid \
  --clip_model_path /mnt/dataset/public_data/cscsx_projects/ctrl_world_train/assets/models/clip-vit-base-patch32
```

如果 S1-A/S1-B latent 已经存在，可以跑完整 S1-C pipeline 检查：

```bash
python3 scripts/validate_s1c_family_pipeline.py
```

## 不要提交的内容

不要提交：

- `.env`
- `assets/models/`
- `latents/`
- `outputs/`
- checkpoint
- log

这些都已经被根目录 `.gitignore` 排除。
