# Ctrl-World Training Package

This package is the cleaned training handoff for Ctrl-World headwrist robot video world models.

Public root:

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train
```

The code is set up so a future Codex agent can install dependencies, precompute latents, compute stats, and launch 8-GPU training from this shared path.

## What Is Included

- `code/`: Ctrl-World model, dataset, training, precompute, stat, watcher, and launch scripts.
- `assets/models/`: local SVD and CLIP weights used by training. This directory exists on the shared filesystem but is excluded from GitHub.
- `docs/`: handoff and experiment notes.
- `.env.sample`: template for W&B secrets and optional path overrides.

## Quick Start

Create the env file:

```bash
cp /mnt/public_ckp/cscsx_projects/ctrl_world_train/.env.sample \
   /mnt/public_ckp/cscsx_projects/ctrl_world_train/.env
```

Fill `WANDB_API_KEY` in:

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train/.env
```

Install dependencies:

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
bash scripts/install_ctrlworld_train_env.sh
```

Launch the main S1-C experiment, matching the successful 2026-06-10 run:

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
bash scripts/launch_training.sh s1_c_3to1to1to1
```

## Main Recipes

```bash
bash scripts/launch_training.sh all50_headwrist
bash scripts/launch_training.sh s1_a_expert
bash scripts/launch_training.sh s1_b_expert_pca
bash scripts/launch_training.sh s1_c_3to1to1to1
bash scripts/launch_training.sh s1_a_single_task place_object_basket
```

## Default Paths

Defaults are defined in:

```bash
code/scripts/ctrlworld_train_env.sh
```

Important defaults:

```bash
TRAIN_PACKAGE_ROOT=/mnt/public_ckp/cscsx_projects/ctrl_world_train
PROJECT_ROOT=${TRAIN_PACKAGE_ROOT}/code
ASSET_ROOT=${TRAIN_PACKAGE_ROOT}/assets/models
CACHE_ROOT=${TRAIN_PACKAGE_ROOT}/latents
OUTPUT_ROOT=${TRAIN_PACKAGE_ROOT}/outputs
```

Raw data remains under:

```bash
/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench
```

Existing inference checkpoints remain under:

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_infer/checkpoints
```

## Reference Experiment

The key reference is the successful S1-C family-balanced run:

```bash
bash /mnt/gyc/Ctrl-World/scripts/train_s1_c_expert_pca_raw_rf_family_balanced_headwrist.sh
```

The cleaned equivalent is:

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
bash scripts/launch_training.sh s1_c_3to1to1to1
```

It trains:

- 5 tasks: `click_alarmclock`, `click_bell`, `place_object_basket`, `open_laptop`, `stack_blocks_two`
- 4 data families: expert, PCA, raw Gaussian, random feasible
- Family sampling: `3:1:1:1`, implemented as `0.5, 0.166667, 0.166667, 0.166667`
- Cameras: `head_camera,left_camera,right_camera`
- Chunk size: `num_frames=16`
- History: `num_history=6`
- Steps: `40000`
- 8-GPU distributed training

More details:

```bash
docs/TRAINING_RECIPES.md
docs/CODEX_HANDOFF.md
docs/s1_c_family_balanced_training_config.md
```
