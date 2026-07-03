# Ctrl-World AF3 Training Notes

更新时间：2026-06-30

本文档是 AF3 Ctrl-World 训练仓库的入口说明。当前主线不是官方 DROID workflow，而是 ActionFollowingData / RoboTwin delta-EE 适配。

## 当前主线

默认训练目标：

```text
数据：expert clean LeRobot + enhanced_v1_split/train
测试：enhanced_v1_split/test_quick
chunk size：32
Ctrl-World history：6
Ctrl-World future frames：26
action dim：14
camera stack：head_camera,left_camera,right_camera -> latent shape (T, 4, 90, 40)
```

支持的采样协议：

```text
mix_3to1to1to1
clean : perturbed : random feasible : counterfactual replay = 3 : 1 : 1 : 1

mix_1to1to1to1
clean : perturbed : random feasible : counterfactual replay = 1 : 1 : 1 : 1
```

数据资产和比例的 source of truth 是：

```text
/mnt/dataset/csx_workspace/Ideas/AF3/docs/tasks/action_following_data_assets.md
```

## 快速入口

正式启动脚本统一放在日期目录：

```text
code/scripts_daily/20260630/run_ctrlworld_action_following_mix3_chunk32_8gpu.sh
code/scripts_daily/20260630/run_ctrlworld_action_following_mix1_chunk32_8gpu.sh
code/scripts_daily/20260630/run_ctrlworld_action_following_clean_chunk32_8gpu.sh
```

在训练环境里运行：

```bash
cd /mnt/dataset/csx_workspace/Ideas/AF3/code/ctrl-world-train-wjx/code

bash scripts_daily/20260630/run_ctrlworld_action_following_mix3_chunk32_8gpu.sh
bash scripts_daily/20260630/run_ctrlworld_action_following_mix1_chunk32_8gpu.sh
bash scripts_daily/20260630/run_ctrlworld_action_following_clean_chunk32_8gpu.sh
```

默认任务是 main experiment 的 4 个任务：

```text
place_can_basket
blocks_ranking_size
move_stapler_pad
turn_switch
```

可用环境变量覆盖：

```bash
TASKS="place_can_basket turn_switch" \
MAX_TRAIN_STEPS=1000 \
bash scripts_daily/20260630/run_ctrlworld_action_following_mix3_chunk32_8gpu.sh
```

## 关键代码

```text
code/scripts/precompute_latents_action_following.py
code/scripts/compute_stat_action_following.py
code/dataset/dataset_action_following.py
code/scripts/train_action_following_mix.sh
code/scripts/train_delta_ee.py
code/models/ctrl_world.py
```

训练仍走 Ctrl-World 原有范式：先把 RGB 预编码为 SVD latent，再训练 diffusion world model。不要让训练 dataloader 直接读原始视频。

## 数据路径

默认 enhanced split：

```text
/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData/enhanced_v1_split
```

默认 clean expert：

```text
/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingBench/data_lerobot/robotwin_delta_ee/demo_clean_zed2i_visible
```

默认 latent cache：

```text
${CACHE_ROOT}/action_following_chunk32_clean_enhanced
${CACHE_ROOT}/action_following_chunk32_clean_only
```

默认 stat：

```text
code/dataset_meta_info/action_following_chunk32_clean_enhanced/stat.json
code/dataset_meta_info/action_following_chunk32_clean_only/stat.json
```

## 验证

静态检查：

```bash
python3 -m py_compile \
  scripts/precompute_latents_action_following.py \
  scripts/compute_stat_action_following.py \
  scripts/train_delta_ee.py \
  dataset/dataset_action_following.py
```

训练环境带 `torch` 时，运行 dataset smoke：

```bash
python3 scripts/validate_action_following_pipeline.py
```

## 历史文档

旧 S1-A / S1-B / S1-C 文档、旧 random feasible 采样文档、旧 assets README 已移动到：

```text
docs/deprecated/
```

这些文档只作为 provenance，不再作为新实验入口。
