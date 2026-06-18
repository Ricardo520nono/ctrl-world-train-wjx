# Assets

This directory is intentionally present in the public training package.

Large model weights live on the shared filesystem and are excluded from GitHub:

- `/mnt/public_ckp/cscsx_projects/ctrl_world_train/assets/models/stable-video-diffusion-img2vid`
- `/mnt/public_ckp/cscsx_projects/ctrl_world_train/assets/models/clip-vit-base-patch32`

Training scripts read these paths through `ASSET_ROOT`, defined in:

- `/mnt/public_ckp/cscsx_projects/ctrl_world_train/code/scripts/ctrlworld_train_env.sh`
