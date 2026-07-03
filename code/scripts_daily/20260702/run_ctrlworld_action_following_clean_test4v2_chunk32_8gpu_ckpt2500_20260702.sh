#!/usr/bin/env bash
set -euo pipefail

# Ctrl-World ActionFollowingData clean-only baseline for the 2026-07-01 test4 task set.
# Training uses expert clean LeRobot only; validation uses enhanced_v1_split/test_quick.
# Saves model checkpoints every 2500 optimizer steps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}"

export TASKS="${TASKS:-place_burger_fries lift_pot dump_bin_bigbin rotate_qrcode}"
export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_action_following_clean_test4v2_chunk32_8gpu_ckpt2500_20260702}"
export LATENT_ROOT="${LATENT_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/latents/action_following_chunk32_clean_enhanced_test4v2_20260701}"
export META_ROOT="${META_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_chunk32_clean_only_test4v2_20260701}"
export STAT_MANIFEST="${STAT_MANIFEST:-${LATENT_ROOT}/manifests/clean_train.jsonl}"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python}"
export WANDB_MODE="${WANDB_MODE:-online}"
export SWANLAB_MODE="${SWANLAB_MODE:-local}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CHUNK_SIZE="${CHUNK_SIZE:-32}"
export NUM_HISTORY="${NUM_HISTORY:-6}"
export NUM_FRAMES="${NUM_FRAMES:-26}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-40000}"
export VALIDATION_STEPS="${VALIDATION_STEPS:-2500}"
export CHECKPOINTING_EPOCHS="${CHECKPOINTING_EPOCHS:-0}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-2500}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export MASTER_PORT="${MASTER_PORT:-29652}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

exec bash scripts/launch_training.sh action_following_clean
