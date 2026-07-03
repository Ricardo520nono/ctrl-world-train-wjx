#!/usr/bin/env bash
set -euo pipefail

# Formal Ctrl-World ActionFollowingData main mix.
# Protocol: clean : perturbed : random_feasible : counterfactual_replay = 3 : 1 : 1 : 1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}"

export TASKS="${TASKS:-place_can_basket blocks_ranking_size move_stapler_pad turn_switch}"
export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_action_following_mix3_chunk32_8gpu_20260630}"
export LATENT_ROOT="${LATENT_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/latents/action_following_chunk32_clean_enhanced}"
export META_ROOT="${META_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_chunk32_clean_enhanced}"
export STAT_MANIFEST="${STAT_MANIFEST:-${LATENT_ROOT}/manifests/train.jsonl}"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python}"
export WANDB_MODE="${WANDB_MODE:-offline}"
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
export CHECKPOINTING_EPOCHS="${CHECKPOINTING_EPOCHS:-1}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-0}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export MASTER_PORT="${MASTER_PORT:-29630}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

exec bash scripts/launch_training.sh action_following_mix3
