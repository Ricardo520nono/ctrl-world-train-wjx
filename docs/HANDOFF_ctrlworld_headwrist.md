# Ctrl-World Headwrist 训练/推理 交接文档

> 最后更新：2026-06-05。接手人请先通读本文件，再看引用到的代码/脚本。

## 0. TL;DR（当前状态）
- 我们在 Ctrl-World（基于 SVD 的机器人视频世界模型）上，把相机视角从旧的 `front+head+left` 换成新的 **`head+left+right`（headwrist：头部 + 左腕 + 右腕）**，重训了 **3 个版本**。
- **3 版训练已全部跑完**（all50=60k、s1_A=40k、s1_B=40k），ckpt 都已自动复制到共享盘 `/mnt/public_ckp/.../ctrl_world_infer/checkpoints/`。
- **下一步重心 = 推理/评测**。⚠️ **最大坑：所有 eval/推理脚本里相机仍写死旧的 `front/head/left`，做 headwrist ckpt 推理前必须改成 `head/left/right`，否则画面布局与训练不一致、结果全错。**

## 1. 仓库与环境
- 主仓库：`/mnt/gyc/Ctrl-World/`（**本项目开发都在这里**）。
- 训练在**百度云 8 卡**跑（脚本内含 `torch.distributed.run --nproc_per_node=8`）。本地 **wjx1 只有 1 张 A800**，仅用于推理/开发/跑 watcher，**不能在本地起 8 卡训练**。
- ckpt 一律存 `/mnt/gyc_ckp`（47T），不要存 `/mnt/gyc`（1T，会撑满）。
- `/mnt/public_ckp` 是**共享盘**（多人用），不是个人目录。
- runtime env（wandb key 等）：`source /mnt/gyc/cosmos-predict2.5-self-evolving-wjx/.env`。
- 终端命令里 `python -c "..."` 一律写成**单行**，不要带换行。

## 2. 模型与数据约定（必懂）
- **模型**：`models/ctrl_world.py` 的 `CrtlWorld`，SVD img2vid 为底座；condition = action(14D) + text(CLIP)；UNet 是 SpatioTemporal。
- **相机拼接**：每路相机单独过 SVD-VAE 编码成 `(T,4,30,40)`，**3 路沿高度方向竖排**成 `(T,4,90,40)`（90=3×30）一整张高图喂进模型。顺序 = 哪路在上/中/下，纯布局，但**预编码与推理必须用同一顺序**。
  - 旧版顺序：`front(0-30) / head(30-60) / left(60-90)`
  - **headwrist 新版顺序：`head(0-30) / left(60) / right(90)`**
- **Action**：14D abs-joint = 左臂(6D delta_pose + 1D gripper) + 右臂(6D + 1D)；训练加 `--use_abs_joint_action`。
- **stat（归一化 p01/p99）只由 action 算，与相机无关** → 换相机时 stat.json 可直接复用，不必重算。
- 关键超参：`height=240`（实际 3×=720）、`num_history=6`、`num_frames=16`(nf16)、bf16、lr=1e-5、bs=1×grad_accum=2×8卡。

## 3. 本轮 3 个 headwrist 训练（已完成）

启动脚本都在 `scripts/`，提交到百度云的 bash 指令就是直接 `bash <脚本>`：

| 版本 | 启动命令（百度云 8 卡示例） | 数据组成 | steps |
|---|---|---|---|
| **all50** | `bash scripts/train_ctrlworld_8gpu_delta_ee_all50_nf16_60k_headwrist.sh` | 全 50 任务，仅原始 demo_clean | 60k，每 5000 步存 |
| **s1_A** | `bash scripts/train_s1_a_expert_only_headwrist.sh` | 5 任务，仅 expert | 40k，**每 1 epoch 存** |
| **s1_B** | `bash scripts/train_s1_b_expert_sliding_pca_single_headwrist.sh` | 5 任务，expert(sliding)+PCA扰动(single) | 40k，**每 1 epoch 存** |

- **s1 的 5 个任务**：`click_alarmclock, click_bell, place_object_basket, open_laptop, stack_blocks_two`；train=episode 0-39 / val=40-49。
- 脚本流程：装依赖 → 预编码 latent（新相机）→ 复用/算 stat → 8 卡训练。**为避免与并发任务争写同一 latent 目录，s1_A/s1_B 各用独立 latent 根**（见脚本内 `EXPERT_LATENT_ROOT`/`PCA_LATENT_ROOT` 注释）。
- 旧脚本（`train_ctrlworld_8gpu_delta_ee_all50_nf16_60k.sh`、`train_s1_a_expert_only.sh`、`train_s1_b_expert_sliding_pca_single.sh`）保留未删，是 headwrist 版的来源，可当对照。

## 4. 关键代码文件
- `scripts/train_delta_ee.py` — 训练主程序（所有版本共用）。本轮新增：
  - `--checkpointing_epochs N`：每完成 N 个整数 epoch 在 epoch 边界存 `checkpoint-epoch{e}-step{g}.pt`。
  - `--checkpointing_steps 0` 可关闭按步存（向后兼容，不传则照常）。
  - 训练末尾**必存** `checkpoint-final-step{g}.pt`。
- `dataset/dataset_delta_ee.py` — 数据集；`dataset_root_path` 支持多根 `+` 拼接，`dataset_root_sampling`(`sliding`/`single`)，`episode_split`。latent 固定 `(T,4,90,40)`。
- `scripts/precompute_latents_delta_ee.py` — 原始数据预编码；本轮加了 `--cameras`（默认旧三路，传 `head_camera,left_camera,right_camera` 用新相机）。
- `scripts/precompute_latents_s1_pca.py` — PCA 扰动数据预编码；同样加了 `--cameras`。
- `scripts/compute_stat_s1.py` / `compute_stat_enhanced.py` — 算 action stat。
- `config.py` — 默认超参（`wm_args`），命令行参数会覆盖它。

## 5. ckpt 路径与自动复制 watcher
训练输出在 `/mnt/gyc_ckp/wjx/ctrlworld/<RUN_NAME>/`；3 个 watcher（本地 wjx1 后台 nohup）把新 ckpt 实时同步到共享盘：

| run | 源 dir | 目标 dir (在 `/mnt/public_ckp/cscsx_projects/ctrl_world_infer/checkpoints/`) | watcher 脚本 |
|---|---|---|---|
| all50 | `.../ctrlworld_delta_ee_all50_8gpu_nf16_60k_headwrist_20260604_222617` | `50tasks_headwrist/` | `scripts/watch_and_copy_50tasks_headwrist.sh` |
| s1_A | `.../s1_A_expert_only_headwrist_20260605_004728` | `5tasks_experts_only_epoch/` | `scripts/watch_and_copy_s1A_5tasks_experts_only_epoch.sh` |
| s1_B | `.../s1_B_expert_sliding_pca_single_headwrist_20260605_004833` | `5tasks_experts_plus_pca_epoch/` | `scripts/watch_and_copy_s1B_5tasks_experts_plus_pca_epoch.sh` |

- watcher 每 60s 轮询，只拷稳定满 120s 的 ckpt（跳过半成品），先拷 `.partial` 再原子 rename。log 在 `/mnt/gyc_ckp/wjx/ctrlworld/watch_*.log`。
- 三版训练已完成，final ckpt 都已就位（all50=13、s1_A=20、s1_B=13 个 ckpt，含各自 `checkpoint-final`）。watcher 可继续留着或 `kill`（训练已结束，没新 ckpt 了）。查 PID：`pgrep -af watch_and_copy`。机器重启需重新 nohup 启动。

## 6. 推理 / 评测（下一步）⚠️
- 入口脚本（在 `scripts/`）：`eval_delta_ee_3cam.py`、`eval_absjoint_3cam.py`、`eval_absjoint_60k_3cam.py`、`infer_single_episode.py`、`eval_action_following.py`、`eval_batch_pcp.py` / `eval_batch_pob.py`，以及对应的 `run_batch_*.sh`、`rollout_*` 系列。
- **必须先改相机**（否则 headwrist ckpt 推理结果全错）：
  - `scripts/eval_delta_ee_3cam.py:34` → `CAMERA_KEYS = ["front_camera","head_camera","left_camera"]`
  - `scripts/eval_delta_ee_3cam.py:180` → `load_gt_single_cam(task, ep_idx, "head_camera")`（GT 取的相机也要对应）
  - `scripts/eval_absjoint_3cam.py:33` → `CAMERA_KEYS = [...]`，且 `:165` 处 `["front","head","left"]` 标签也要同步改
  - 其余 eval 脚本若也写死相机，一并 grep 改：`grep -rn "front_camera\|CAMERA_KEYS" scripts/`
- 改成 `["head_camera","left_camera","right_camera"]`，顺序与训练一致（head 在最上）。
- WorldArena 16 指标评测框架在 `/mnt/gyc/worldarena/WorldArena/`（之前评测过旧版 ckpt）。
- 注意：百度云/部分机器没装 ffmpeg，训练 log 里 `write_video failed (ffmpeg missing)` 是无害 warning（只是没生成验证视频），不影响 ckpt。

## 7. 相关记忆/文档
- `project_docs/progress.md`、`note.md`、`task.md`、`plan.md` — 早期进度。
- 本轮详情见用户 memory：`project_ctrlworld_headwrist_training`、`project_ctrlworld_ckpt_watchers`、`project_worldarena_benchmark`。
