# Codex Handoff

This document is written for a future Codex agent taking over Ctrl-World training.

## First Read

Start here:

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train/README.md
/mnt/public_ckp/cscsx_projects/ctrl_world_train/docs/TRAINING_RECIPES.md
```

The historical headwrist handoff is also included:

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train/docs/HANDOFF_ctrlworld_headwrist.md
```

Treat the new README and this file as authoritative for paths. Some historical docs still mention `/mnt/gyc/Ctrl-World` or `/mnt/gyc_ckp`; those describe the original run locations.

## Code Layout

```text
ctrl_world_train/
  code/
    dataset/
    dataset_meta_info/
    models/
    scripts/
  assets/models/
  docs/
```

Core files:

```bash
code/scripts/train_delta_ee.py
code/dataset/dataset_delta_ee.py
code/dataset/dataset_delta_ee_family.py
code/scripts/precompute_latents_delta_ee.py
code/scripts/precompute_latents_s1_pca.py
code/scripts/compute_stat_s1.py
code/scripts/compute_stat_family_roots.py
code/models/ctrl_world.py
```

## Environment

Create:

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train/.env
```

from:

```bash
/mnt/public_ckp/cscsx_projects/ctrl_world_train/.env.sample
```

Required:

```bash
WANDB_API_KEY=...
```

Install:

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
bash scripts/install_ctrlworld_train_env.sh
```

## Main One-Command Training

For the current recommended S1-C recipe:

```bash
cd /mnt/public_ckp/cscsx_projects/ctrl_world_train/code
bash scripts/launch_training.sh s1_c_3to1to1to1
```

This is the cleaned equivalent of the successful original command:

```bash
bash /mnt/gyc/Ctrl-World/scripts/train_s1_c_expert_pca_raw_rf_family_balanced_headwrist.sh
```

## Path Overrides

Defaults are in:

```bash
code/scripts/ctrlworld_train_env.sh
```

Common overrides:

```bash
PYTHON_BIN=/usr/bin/python3
CACHE_ROOT=/mnt/public_ckp/cscsx_projects/ctrl_world_train/latents
OUTPUT_ROOT=/mnt/public_ckp/cscsx_projects/ctrl_world_train/outputs
```

Example:

```bash
OUTPUT_ROOT=/mnt/public_ckp/cscsx_projects/ctrl_world_train/outputs_test \
bash scripts/launch_training.sh s1_a_expert
```

## Validation Before Expensive 8-GPU Jobs

Static shell syntax:

```bash
for f in scripts/*.sh; do bash -n "$f"; done
```

Python compile:

```bash
python3 -m py_compile \
  scripts/train_delta_ee.py \
  dataset/dataset_delta_ee.py \
  dataset/dataset_delta_ee_family.py \
  models/ctrl_world.py
```

Family sampler wiring without requiring precomputed real latents:

```bash
python3 scripts/validate_s1c_family_pipeline.py --skip-real-inputs
```

Full S1-C validation after S1-A/S1-B latents exist:

```bash
python3 scripts/validate_s1c_family_pipeline.py
```

## What Not To Commit

Do not commit:

- `.env`
- `assets/models/`
- `latents/`
- `outputs/`
- checkpoints
- logs

These are excluded by the root `.gitignore`.
