# Codex 接手说明

这份文档是给后续接手 Ctrl-World 训练任务的 Codex 看的。

## 优先阅读

先读：

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train/README.md
/mnt/public_ckp/cscsx_projects/ctrl_world_train/docs/TRAINING_RECIPES.md
```

历史 headwrist 交接文档也保留在：

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train/docs/HANDOFF_ctrlworld_headwrist.md
```

注意：新的 `README.md` 和本文件是当前 public 训练包的权威路径说明。历史文档里如果还出现 `/mnt/gyc/Ctrl-World` 或 `/mnt/gyc_ckp`，那是在描述当时原始训练 run 的位置。

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

核心文件：

```bash
code/scripts/train_delta_ee.py
code/dataset/dataset_delta_ee.py
code/dataset/dataset_delta_ee_family.py
code/scripts/precompute_latents_delta_ee.py
code/scripts/precompute_latents_s1_pca.py
code/scripts/compute_stat_s1.py
code/scripts/compute_stat_family_roots.py
code/models/ctrl_world.py
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

这是成功原始命令的 public 路径整理版：

```bash
bash /mnt/gyc/Ctrl-World/scripts/train_s1_c_expert_pca_raw_rf_family_balanced_headwrist.sh
```

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
