# ActionFollowingData Ctrl-World Quickstart

## 1. 环境

```bash
cd /mnt/dataset/csx_workspace/Ideas/AF3/code/ctrl-world-train-wjx/code
bash scripts/install_ctrlworld_train_env.sh
```

大模型权重默认从：

```text
${ASSET_ROOT}/stable-video-diffusion-img2vid
${ASSET_ROOT}/clip-vit-base-patch32
```

读取；`ASSET_ROOT` 默认由 `scripts/ctrlworld_train_env.sh` 定义，可通过环境变量覆盖。

## 2. 预编码

正式 launcher 会自动预编码。如果需要单独运行：

```bash
python scripts/precompute_latents_action_following.py \
  --svd_path "${ASSET_ROOT}/stable-video-diffusion-img2vid" \
  --out_root "${CACHE_ROOT}/action_following_chunk32_clean_enhanced" \
  --tasks place_can_basket blocks_ranking_size move_stapler_pad turn_switch \
  --split both \
  --include_clean \
  --include_enhanced
```

输出：

```text
samples/<family>/<subtype>/<task>/<sample_id>.pt
manifests/train.jsonl
manifests/test_quick.jsonl
```

每个 `.pt` 包含：

```text
latent:     (T, 4, 90, 40)
action_pos: (T, 14)
text:       str
```

enhanced 样本会额外包含 `ee_target: (T, 20)`；clean LeRobot 当前不会写 `ee_target`，所以默认不要打开 `--use_ee_head`。

同时会写出轻量 action sidecar：

```text
samples/.../<sample_id>.pt.action.npy
```

## 3. Stat

```bash
python scripts/compute_stat_action_following.py \
  --latent_root "${CACHE_ROOT}/action_following_chunk32_clean_enhanced" \
  --out_dir dataset_meta_info/action_following_chunk32_clean_enhanced \
  --action_dim 14
```

`compute_stat_action_following.py` 优先读取 `.action.npy`，避免为了统计 action 反序列化大 latent。`train_delta_ee.py` 会用 `state_01/state_99` 把 action clip/normalize 到 `[-1, 1]`。

## 4. 训练

推荐入口：

```bash
bash scripts_daily/20260630/run_ctrlworld_action_following_mix3_chunk32_8gpu.sh
```

四类均分：

```bash
bash scripts_daily/20260630/run_ctrlworld_action_following_mix1_chunk32_8gpu.sh
```

clean baseline：

```bash
bash scripts_daily/20260630/run_ctrlworld_action_following_clean_chunk32_8gpu.sh
```

关键默认值：

```text
chunk size = 32
num_history = 6
num_frames = 26
action_dim = 14
validation = test_quick
```

## 5. Sampler 规则

`dataset/dataset_action_following.py` 实现以下约定：

- 训练按 family 先采样，再采 subtype / task / sample / window。
- `perturbed` 不做 sliding，每条 canonical sample 只取前缀 chunk。
- `random_feasible` 和 `counterfactual_replay` 做 stride-1 sliding。
- `test_quick` 使用 manifest 里的固定 `chunk_start`，不重新随机采样。
- `mix_3to1to1to1` 和 `mix_1to1to1to1` 是大类 chunk-sample 目标比例。

## 6. 最小 checkpoint 推理

入口：

```bash
cd /mnt/dataset/csx_workspace/Ideas/AF3/code/ctrl-world-train-wjx/code
python scripts/infer_action_following_ckpt.py \
  --preset clean \
  --sample_index 0 \
  --steps 8
```

也可以用 wrapper：

```bash
PRESET=clean SAMPLE_INDEX=0 NUM_INFERENCE_STEPS=8 \
  bash scripts_daily/20260703/run_ctrlworld_action_following_infer_ckpt2500_smoke.sh
```

`--preset clean` 默认加载：

```text
/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/outputs/ACWM_ctrlworld_action_following_clean_test4v2_chunk32_8gpu_ckpt2500_20260702/checkpoint-step2500-epoch0.21.pt
```

`--preset mix3` 默认加载：

```text
/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/outputs/ACWM_ctrlworld_action_following_mix3_test4v2_chunk32_8gpu_ckpt2500_20260702/checkpoint-step2500-epoch0.21.pt
```

脚本会加载 Ctrl-World checkpoint，从训练 manifest 取一个 ActionFollowingData 样本，运行 world-model latent prediction，并写出：

```text
gt_left_pred_right.mp4
prediction_latents.pt
metadata.json
```

默认输出目录：

```text
/tmp/ctrlworld_action_following_infer/${PRESET}_sample${SAMPLE_INDEX}_steps${NUM_INFERENCE_STEPS}
```

`gt_left_pred_right.mp4` 中左侧是 GT latent decode，右侧是预测 latent decode；三行对应 head / left / right 三个 camera latent band。

## 7. 注意事项

- 不要把旧 S1 `num_frames=16` 或 `num_frames=32` 直接搬到 ActionFollowingData 主线。
- 对 chunk size 32，Ctrl-World 应使用 `num_history=6, num_frames=26`，总长度正好是 32。
- clean expert 是 LeRobot 原格式；enhanced 是 canonical sample 目录；训练前统一转成 latent `.pt`。
- 当前 clean expert 没有 quaternion endpose，预处理脚本不会给 clean `.pt` 写 `ee_target`。默认 `use_ee_head=False`，不影响主训练。
