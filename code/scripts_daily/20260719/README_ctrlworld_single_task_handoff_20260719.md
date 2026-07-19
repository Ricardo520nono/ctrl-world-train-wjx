# Ctrl-World 单任务模型交接

本目录提供 `place_burger_fries` 和 `dump_bin_bigbin` 两个 Ctrl-World 单任务模型的最小评测命令。当前固定使用训练过程中已完整保存的 step12500 checkpoint；两个 40k 训练仍在继续，因此这不是最终模型。

## 模型约定

- 数据：单任务 Mix4，chunk-sample level 为 clean 50%，perturbed / random feasible / counterfactual replay / exploration 各 12.5%。
- 输入输出：当前 1 帧三视角图像 + 32 步 Rot6D20 action + full task description，生成未来 32 帧三视角视频。
- Action：20D，双臂各为 XYZ 3D + Rot6D 6D + gripper 1D。
- 推理：BF16，Ctrl-World diffusion 20 steps，chunk size 32。

Checkpoint：

```text
/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/outputs/ACWM_ctrlworld_mix4_place_burger_fries_c32_cur1f32_rot6d20_cfstatemajor_8gpu_20260718/checkpoint-step12500-epoch3.70.pt

/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/outputs/ACWM_ctrlworld_mix4_dump_bin_bigbin_c32_cur1f32_rot6d20_cfstatemajor_8gpu_20260718/checkpoint-step12500-epoch3.82.pt
```

Policy closed-loop 固定使用：

```text
/mnt/dataset/csx_workspace/Ideas/data_AF3/starVLA_dev/checkpoints/qwenoft_rot6d20_clean50_randomized500_decord_lru8_nofuture_locality_b4_official_48g_bs4/ACWM_T2A_qwenoft_rot6d20_clean50_randomized500_decord_lru8_nofuture_locality_b4_official_48g_bs4_20260709/checkpoints/steps_90000_pytorch_model.pt
```

该 policy 的冻结配置保留旧路径 `/mnt/public_ckp/Qwen3-VL-4B-Instruct`。评测脚本在容器内把它映射到实际只读模型目录 `/mnt/dataset/public_data/Qwen3-VL-4B-Instruct`，不会修改 policy checkpoint 或其 `config.full.yaml`。

## 两类评测

每个入口都在该任务 clean trajectory index 0 上生成两组结果：

1. `open_loop_expert_actions`：动作来自记录的专家轨迹，但每段生成的最后一帧会递归作为下一段输入。
2. `closed_loop_qwenoft_step90000`：QwenOFT 每轮根据当前 WM 图像预测 32 步动作，Ctrl-World 生成 32 帧，末帧再反馈给 policy。

第二项是 learned policy/world-model compatibility rollout，不是 RoboTwin 物理仿真成功率，不能作为任务成功率报告。

## 运行

每个任务需要 3 张 GPU：policy server、open-loop WM、closed-loop WM 各一张。

```bash
POLICY_GPU=0 OPEN_GPU=1 CLOSED_GPU=2 POLICY_PORT=5694 \
bash scripts_daily/20260719/run_ctrlworld_place_burger_fries_handoff_step12500_20260719.sh
```

```bash
POLICY_GPU=0 OPEN_GPU=1 CLOSED_GPU=2 POLICY_PORT=5694 \
bash scripts_daily/20260719/run_ctrlworld_dump_bin_bigbin_handoff_step12500_20260719.sh
```

8GPU 节点上并行跑两个任务：

```bash
bash scripts_daily/20260719/run_ctrlworld_single2_handoff_step12500_open_qwenoftclosed_8gpu_20260719.sh
```

只做路径和参数检查：

```bash
DRY_RUN=1 bash scripts_daily/20260719/run_ctrlworld_place_burger_fries_handoff_step12500_20260719.sh
DRY_RUN=1 bash scripts_daily/20260719/run_ctrlworld_dump_bin_bigbin_handoff_step12500_20260719.sh
```

## 输出

结果保存在各模型训练目录下的：

```text
handoff_eval/<run_tag>/
```

关键文件：

- Open-loop：`open_loop_expert_actions/autoregressive_pred.mp4`、`gt_left_autoreg_right.mp4`、`metadata.json`。
- Closed-loop：`closed_loop_qwenoft_step90000/policy_server_autoreg_pred.mp4`、`gt_left_policy_autoreg_right.mp4`、`metadata.json`。
- 根目录：`eval_contract.txt`、`policy_server.log`、`open_loop.log`、`closed_loop.log`、完成标记 `.complete`。

脚本拒绝写入非空输出目录。评测更晚 checkpoint 时，请设置新的 `WM_CKPT` 和 `RUN_TAG`，不要覆盖现有结果。

## 已完成的 step12500 参考结果

AIHC `train22` job `job-ums20lzohe3z` 已成功完成两个任务的四条 rollout。原始 `job-2cj11v9gofb8` 因 policy 冻结配置中的旧 base-VLM 路径缺失而在加载阶段失败，未启动 WM rollout；`retry1` 使用上文兼容映射修复，原失败证据保留且未覆盖。

结果目录：

```text
/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/outputs/ACWM_ctrlworld_mix4_place_burger_fries_c32_cur1f32_rot6d20_cfstatemajor_8gpu_20260718/handoff_eval/step12500_sample0_steps20_aihc_train22_retry1_20260719

/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/outputs/ACWM_ctrlworld_mix4_dump_bin_bigbin_c32_cur1f32_rot6d20_cfstatemajor_8gpu_20260718/handoff_eval/step12500_sample0_steps20_aihc_train22_retry1_20260719
```

| Task | Frames / chunks | Open latent MSE | Closed latent MSE |
|---|---:|---:|---:|
| `place_burger_fries` | 241 / 8 | 0.164567 | 0.716690 |
| `dump_bin_bigbin` | 340 / 11 | 0.145155 | 0.426697 |

最终审计：8 个主要 MP4 均可解码；place 为241帧，dump 为340帧；生成视频 `320x720`，GT/生成对比视频 `640x720`；policy 与 WM action arrays 分别为 `(8,32,20)` 和 `(11,32,20)`，全部 finite；checkpoint/stat/policy/sample provenance 均写入 metadata。四个结果目录各包含 `review_contact_sheet.png`，肉眼检查 RGB 通道正确，没有 R/B 反转。

这些结果显示 step12500 模型可完成长时递归生成，但随着 horizon 增长存在明显视觉漂移，closed-loop 更强。它们适合作为接口和定性 sanity check，不应作为最终40k模型质量或 simulator task success 的结论。
