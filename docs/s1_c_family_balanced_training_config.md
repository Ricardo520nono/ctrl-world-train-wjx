# S1-C Family-Balanced Training Config

## Experiment

- Name prefix: `s1_C_3to1to1to1_family_balanced_headwrist`
- Purpose: S1 5-task training with mixed `expert + PCA + raw + random feasible` data.
- Target machine: Baidu 8-GPU distributed training.
- Launch script:
  - `/mnt/gyc/Ctrl-World/scripts/train_s1_c_expert_pca_raw_rf_family_balanced_headwrist.sh`

## Tasks And Splits

- Task count: 5
- Tasks:
  - `click_alarmclock`
  - `click_bell`
  - `place_object_basket`
  - `open_laptop`
  - `stack_blocks_two`
- Train episodes: `0-39`
- Validation episodes: `40-49`

## Cameras And Action

- Cameras: `head_camera,left_camera,right_camera`
- Camera stacking order in latent height: `head -> left -> right`
- Action dimension: `14`
- Action type: absolute joint action
- Training flag: `--use_abs_joint_action`

## Window / Chunk Config

- `num_history = 6`
- `num_frames = 16`
- Effective window length: `6 + 16 = 22`
- Chunk size / prediction frames: `16`
- Input height: `240`
- Expected latent shape per sampled item:
  - video latent: `(22, 4, 90, 40)`
  - action: `(22, 14)`

## Sampling

This run uses the new family-balanced sampler, not a pre-generated static manifest.

Sampling hierarchy:

```text
family -> task -> trajectory/sample -> window
```

Family ratio: `3:1:1:1`

```text
expert:          0.500000
pca:             0.166667
raw:             0.166667
random_feasible: 0.166667
```

Inside `random_feasible`, the two variants are sampled uniformly:

```text
rf_uniform : rf_weighted = 1 : 1
```

Training arguments:

```text
--use_family_balanced_sampler
--family_sampling "expert=0.5,pca=0.166667,raw=0.166667,random_feasible=0.166667"
--family_sampling_seed 20260610
```

## Training Scale

- GPUs: 8
- Per-GPU train batch size: `1`
- Gradient accumulation steps: `2`
- Effective global batch size:

```text
1 * 2 * 8 = 16 windows / optimizer step
```

- Max train steps: `40000`
- Family dataset length: `68429`

The dataset length matches the chunk-size-16 static-manifest reference size from the sampling doc:

```text
expert: 34214
pca:    11405
raw:    11405
rf:     11405
total:  68429
```

Although this run uses family-balanced online sampling, keeping `family_dataset_length=68429` makes epoch/checkpoint semantics comparable to the documented chunk-size-16 manifest estimate.

## Optimization

- Learning rate: `1e-5`
- Mixed precision: `bf16`
- Checkpointing by step: disabled
- Checkpointing by epoch: every `1` epoch
- Validation interval: every `2500` steps

Training arguments:

```text
--learning_rate 1e-5
--mixed_precision bf16
--checkpointing_steps 0
--checkpointing_epochs 1
--validation_steps 2500
```

## Data And Output Paths

Output root:

```text
/mnt/gyc_ckp/wjx/ctrlworld/${RUN_NAME}
```

Dataset meta/stat:

```text
/mnt/gyc/Ctrl-World/dataset_meta_info/s1_C_expert_pca_raw_rf_3to1to1to1/stat.json
```

Latent roots:

```text
expert:
/mnt/gyc_ckp/wjx/ctrlworld/precomputed_latents_s1A_5tasks_14d_headwrist

pca:
/mnt/gyc_ckp/wjx/ctrlworld/s1B_latents_pca_train_headwrist

raw:
/mnt/gyc_ckp/wjx/ctrlworld/s1C_latents_raw_s0025_train_headwrist

random feasible uniform:
/mnt/gyc_ckp/wjx/ctrlworld/s1C_latents_rf300_uniform_train_headwrist

random feasible weighted:
/mnt/gyc_ckp/wjx/ctrlworld/s1C_latents_rf300_weighted_train_headwrist
```

Source enhanced data roots:

```text
raw:
/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/EnhancedData/perturbed_raw_gaussian/sigma_0p0025

random feasible base:
/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/EnhancedData/random_feasible_300step_5task_2ep5start_formal_v1/random_feasible_random_walk
```

## One-Line Summary

S1-C trains the 5-task headwrist model with `num_frames/chunk_size=16`, `history=6`, 14D absolute joint actions, 8-GPU DDP, effective global batch size 16, 40k steps, bf16, LR `1e-5`, and family-balanced online sampling with `expert:pca:raw:random_feasible = 3:1:1:1`.
