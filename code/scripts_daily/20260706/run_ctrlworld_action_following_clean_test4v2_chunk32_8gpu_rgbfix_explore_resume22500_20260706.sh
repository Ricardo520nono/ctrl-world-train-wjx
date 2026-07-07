#!/usr/bin/env bash
set -euo pipefail

# Warm-start continuation for the interrupted 20260705 clean 4-task run.
# train_delta_ee.py only restores model weights from --ckpt_path; optimizer,
# scheduler, and global_step are restarted. MAX_TRAIN_STEPS is set to the
# remaining 17,500 optimizer steps to reach the original 40k total budget.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}"

export TASKS="${TASKS:-place_burger_fries lift_pot dump_bin_bigbin rotate_qrcode}"
export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_action_following_clean_test4v2_chunk32_8gpu_rgbfix_explore_resume22500_20260706}"
export AF3_CTRLWORLD_ROOT="${AF3_CTRLWORLD_ROOT:-/mnt/dataset/csx_workspace/Ideas/data_AF3/ctrl-world}"
export LATENT_ROOT="${LATENT_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/latents/action_following_chunk32_clean_only_test4v2_rgbfix_explore_20260705}"
export META_ROOT="${META_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_chunk32_clean_only_test4v2_rgbfix_explore_20260705}"
export STAT_MANIFEST="${STAT_MANIFEST:-${LATENT_ROOT}/manifests/clean_train.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${AF3_CTRLWORLD_ROOT}/outputs/${RUN_NAME}}"
export RESUME_CKPT="${RESUME_CKPT:-${AF3_CTRLWORLD_ROOT}/outputs/ACWM_ctrlworld_action_following_clean_test4v2_chunk32_8gpu_rgbfix_explore_20260705/checkpoint-step22500-epoch12.04.pt}"
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
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-17500}"
export VALIDATION_STEPS="${VALIDATION_STEPS:-2500}"
export CHECKPOINTING_EPOCHS="${CHECKPOINTING_EPOCHS:-0}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-2500}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export MASTER_PORT="${MASTER_PORT:-29682}"
export SAMPLER_AUDIT_SAMPLES="${SAMPLER_AUDIT_SAMPLES:-10000}"
export SAMPLER_AUDIT_SEED="${SAMPLER_AUDIT_SEED:-20260705}"
export SAMPLER_AUDIT_TOLERANCE="${SAMPLER_AUDIT_TOLERANCE:-0.02}"
export CKPT_PATH="${RESUME_CKPT}"

if [[ ! -s "${RESUME_CKPT}" ]]; then
  echo "[ERROR] resume checkpoint missing or empty: ${RESUME_CKPT}" >&2
  exit 2
fi

exec bash scripts/launch_training.sh action_following_clean
