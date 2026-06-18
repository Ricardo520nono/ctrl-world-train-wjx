# Training Recipes

All commands assume:

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
```

## Shared Conventions

- Training machine: Baidu 8-GPU node.
- Local single-GPU machine is for development and inference debugging only.
- Camera order: `head_camera,left_camera,right_camera`.
- Latent shape for headwrist videos: `(T, 4, 90, 40)`.
- Action dim: `14`.
- Action type: absolute joint action.
- Train split: source episodes `0-39`.
- Val split: source episodes `40-49`.
- Default latent cache: `/mnt/public_ckp/cscsx_projects/ctrl_world_train/latents`.
- Default training output: `/mnt/public_ckp/cscsx_projects/ctrl_world_train/outputs`.

## all50_headwrist

Launch:

```bash
bash scripts/launch_training.sh all50_headwrist
```

Underlying script:

```bash
scripts/train_ctrlworld_8gpu_delta_ee_all50_nf16_60k_headwrist.sh
```

Config:

- Tasks: all 50 ActionFollowingBench tasks.
- Families: expert only.
- Steps: `60000`.
- Chunk size: `16`.
- Effective global batch: `8 GPUs * train_batch_size 1 * grad_accum 2 = 16`.
- Saves: every completed epoch plus final checkpoint.

## s1_a_expert

Launch:

```bash
bash scripts/launch_training.sh s1_a_expert
```

Underlying script:

```bash
scripts/train_s1_a_expert_only_headwrist.sh
```

Config:

- Tasks: `click_alarmclock`, `click_bell`, `place_object_basket`, `open_laptop`, `stack_blocks_two`.
- Families: expert only.
- Steps: `40000`.
- Chunk size: `16`.
- Saves: every completed epoch plus final checkpoint.

## s1_b_expert_pca

Launch:

```bash
bash scripts/launch_training.sh s1_b_expert_pca
```

Underlying script:

```bash
scripts/train_s1_b_expert_sliding_pca_single_headwrist.sh
```

Config:

- Tasks: same 5 S1 tasks.
- Families: expert + PCA perturbation.
- PCA data: `EnhancedData/perturbed_pca_gaussian/c_8_sigma_0p05`.
- Steps: `40000`.
- Chunk size: `16`.

## s1_c_3to1to1to1

Launch:

```bash
bash scripts/launch_training.sh s1_c_3to1to1to1
```

Underlying script:

```bash
scripts/train_s1_c_expert_pca_raw_rf_family_balanced_headwrist.sh
```

This is the main reference for the cleaned package.

Config:

- Tasks: same 5 S1 tasks.
- Families:
  - expert
  - PCA, `c_8_sigma_0p05`
  - raw Gaussian, `sigma_0p0025`
  - random feasible 300-step, uniform + weighted roots combined as one family
- Sampling: family-balanced sampler.
- Ratio: `3:1:1:1`.
- Script values:

```bash
FAMILY_SAMPLING="expert=0.5,pca=0.166667,raw=0.166667,random_feasible=0.166667"
FAMILY_DATASET_LENGTH=68429
```

- Steps: `40000`.
- Chunk size: `16`.
- Saves: every completed epoch plus final checkpoint.

## s1_a_single_task

Launch one task:

```bash
bash scripts/launch_training.sh s1_a_single_task place_object_basket
```

Valid tasks:

```text
click_alarmclock
click_bell
place_object_basket
open_laptop
stack_blocks_two
```

Config:

- Families: expert only.
- One task per run.
- Steps: `40000`.
- Chunk size: `16`.

## Checkpoint Copy Watchers

Watcher scripts are still available. They copy complete checkpoint files to the inference checkpoint directory using a `.partial` temp file and an atomic rename.

For new runs, pass `SRC` and `DST` explicitly:

```bash
SRC=/mnt/public_ckp/cscsx_projects/ctrl_world_train/outputs/<RUN_NAME> \
DST=/mnt/public_ckp/cscsx_projects/ctrl_world_infer/checkpoints/<TARGET_NAME> \
bash scripts/watch_and_copy_s1C_3to1to1to1_family_balanced_chunk16.sh
```
