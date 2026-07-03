# 资产目录

这个目录用于放训练需要的大模型权重。

大模型权重保存在共享盘上，但不会提交到 GitHub：

- `/mnt/public_ckp/cscsx_projects/ctrl_world_train/assets/models/stable-video-diffusion-img2vid`
- `/mnt/public_ckp/cscsx_projects/ctrl_world_train/assets/models/clip-vit-base-patch32`

训练脚本通过 `ASSET_ROOT` 读取这些路径。默认值定义在：

- `/mnt/public_ckp/cscsx_projects/ctrl_world_train/code/scripts/ctrlworld_train_env.sh`
