# Assets

本目录只放轻量说明。大模型权重不要提交到 Git。

Ctrl-World 训练需要：

```text
stable-video-diffusion-img2vid/
clip-vit-base-patch32/
```

默认读取位置由 `code/scripts/ctrlworld_train_env.sh` 的 `ASSET_ROOT` 控制。正式训练时优先使用公共大盘路径，并通过环境变量覆盖：

```bash
ASSET_ROOT=/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/assets/models \
bash scripts/launch_training.sh action_following_mix3
```
