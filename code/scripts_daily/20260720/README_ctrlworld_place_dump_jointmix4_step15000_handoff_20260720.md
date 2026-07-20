# Ctrl-World Place+Dump 联合模型交接

本目录提供当前 Place+Dump 联合 Mix4 Ctrl-World 的最小开环与 policy 闭环评测。固定 checkpoint 为训练中的 step15000；80k 训练仍在继续，因此该 checkpoint 是阶段性模型，不是最终模型。

## 模型

```text
/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/outputs/ACWM_ctrlworld_mix4_place_dump_c32_cur1f32_rot6d20_cfstatemajor_rgbfix1_8gpu_80k_20260719_debug/checkpoint-step15000-epoch2.26.pt
```

- 任务：`place_burger_fries`、`dump_bin_bigbin`，训练中按 task 各50%采样。
- 每任务 Mix4：clean50%，perturbed / random feasible / counterfactual replay / exploration各12.5%。
- 输入：当前1帧三视角图像、32步Rot6D20 action、RoboTwin full description。
- 输出：未来32帧三视角视频。
- Action：20D，双臂各为 XYZ3 + Rot6D6 + gripper1。
- 推理：BF16，Ctrl-World20步去噪，chunk size32。

联合训练 stat：

```text
/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/dataset_meta_info/action_following_cur1f32_mix4_place_dump_rot6d20_cfstatemajor_rgbfix1_20260719_debug/stat.json
```

闭环 policy：

```text
/mnt/dataset/csx_workspace/Ideas/data_AF3/starVLA_dev/checkpoints/qwenoft_rot6d20_clean50_randomized500_decord_lru8_nofuture_locality_b4_official_48g_bs4/ACWM_T2A_qwenoft_rot6d20_clean50_randomized500_decord_lru8_nofuture_locality_b4_official_48g_bs4_20260709/checkpoints/steps_90000_pytorch_model.pt
```

## 评测语义

每个任务使用 `clean_train.jsonl` 中该任务的 trajectory index0：

1. `open_loop_expert_actions`：使用记录的 expert actions；每个32步 chunk 的最终生成帧递归作为下一段 current frame。
2. `closed_loop_qwenoft_step90000`：QwenOFT 每轮从当前三视角 WM 图像预测32步动作，Ctrl-World 生成32帧，末帧再反馈给 policy。

闭环严格执行：`policy normalized -> physical Rot6D20 -> 当前联合 Ctrl-World stat p01/p99 normalized`。禁止将 policy normalized action 直接传给 Ctrl-World。

闭环结果是 learned policy/world-model compatibility rollout，不是 RoboTwin 物理仿真成功率。

## 运行

8GPU 节点并行完成两个任务的四条 rollout：

```bash
bash scripts_daily/20260720/run_ctrlworld_place_dump_jointmix4_step15000_open_qwenoftclosed_8gpu_20260720.sh
```

GPU 布局：Place 使用 policy/open/closed GPU `0/2/4`；Dump 使用 `1/3/5`。GPU6、7保留。

只检查路径和参数：

```bash
DRY_RUN=1 bash scripts_daily/20260720/run_ctrlworld_place_dump_jointmix4_step15000_open_qwenoftclosed_8gpu_20260720.sh
```

单任务入口：

```bash
bash scripts_daily/20260720/run_ctrlworld_place_burger_fries_jointmix4_step15000_handoff_20260720.sh
bash scripts_daily/20260720/run_ctrlworld_dump_bin_bigbin_jointmix4_step15000_handoff_20260720.sh
```

## 输出

```text
/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/outputs/ACWM_ctrlworld_mix4_place_dump_c32_cur1f32_rot6d20_cfstatemajor_rgbfix1_8gpu_80k_20260719_debug/handoff_eval/step15000_sample0_steps20_statbridgefix_aihc_20260720/
```

每个任务目录包含：

- Open-loop：`open_loop_expert_actions/autoregressive_pred.mp4`、`gt_left_autoreg_right.mp4`、`metadata.json`。
- Closed-loop：`closed_loop_qwenoft_step90000/policy_server_autoreg_pred.mp4`、`gt_left_policy_autoreg_right.mp4`、`metadata.json`。
- 根目录：`eval_contract.txt`、policy/open/closed日志和任务级 `.complete`。

批量 launcher 最后运行 `scripts/audit_ctrlworld_handoff_eval.py`，检查四个主要MP4、checkpoint/stat/policy provenance、20D action bridge、三套闭环 action arrays 和全部 finite 条件。全部通过后生成顶层 `audit_summary.json` 与 `.complete`。

脚本拒绝写入非空结果目录。评测其它 checkpoint 时必须使用新的 `RUN_TAG`，不要覆盖既有结果。

## 当前状态

脚本、交接约定和AIHC审核卡已准备；正式 step15000 四条 rollout 尚未提交。结果完成后在本节补充帧数、chunk数、open/closed latent MSE和视频审计结论。
