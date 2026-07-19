#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_mix4_place_burger_fries_c32_cur1f32_rot6d20_cfstatemajor_8gpu_20260718}"
export TASKS="place_burger_fries"
export PUBLIC_ROOT="${PUBLIC_ROOT:-/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world}"
export LATENT_ROOT="${LATENT_ROOT:-${PUBLIC_ROOT}/latents/action_following_cur1f32_mix4_place_burger_fries_rot6d20_cfstatemajor_incremental_20260718}"
export META_ROOT="${META_ROOT:-${PUBLIC_ROOT}/dataset_meta_info/action_following_cur1f32_mix4_place_burger_fries_rot6d20_cfstatemajor_incremental_20260718}"
export STAT_MANIFEST="${STAT_MANIFEST:-${LATENT_ROOT}/manifests/train.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PUBLIC_ROOT}/outputs/${RUN_NAME}}"
export ACTION_DIM="${ACTION_DIM:-20}"
export CHUNK_SIZE="${CHUNK_SIZE:-32}"
export NUM_HISTORY="${NUM_HISTORY:-1}"
export NUM_FRAMES="${NUM_FRAMES:-32}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-40000}"
export VALIDATION_STEPS="${VALIDATION_STEPS:-2500}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-2500}"
export CHECKPOINTING_EPOCHS="${CHECKPOINTING_EPOCHS:-0}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export WANDB_MODE="${WANDB_MODE:-online}"
export REQUIRE_WANDB_ONLINE="${REQUIRE_WANDB_ONLINE:-1}"
export SWANLAB_MODE="${SWANLAB_MODE:-local}"
export SAMPLER_AUDIT_SAMPLES="${SAMPLER_AUDIT_SAMPLES:-10000}"
export SAMPLER_AUDIT_SEED="${SAMPLER_AUDIT_SEED:-20260718}"
export SAMPLER_AUDIT_TOLERANCE="${SAMPLER_AUDIT_TOLERANCE:-0.02}"
export MASTER_PORT="${MASTER_PORT:-29742}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] run=${RUN_NAME}"
  echo "[DRY_RUN] tasks=${TASKS}"
  echo "[DRY_RUN] latent_root=${LATENT_ROOT}"
  echo "[DRY_RUN] meta_root=${META_ROOT}"
  echo "[DRY_RUN] output_dir=${OUTPUT_DIR}"
  echo "[DRY_RUN] protocol=mix_4to1to1to1to1 action_dim=${ACTION_DIM} chunk_size=${CHUNK_SIZE} num_history=${NUM_HISTORY} num_frames=${NUM_FRAMES}"
  echo "[DRY_RUN] nproc=${NPROC_PER_NODE} train_batch_size=${TRAIN_BATCH_SIZE} grad_accum=${GRAD_ACCUM} max_steps=${MAX_TRAIN_STEPS} checkpoint_steps=${CHECKPOINTING_STEPS}"
  echo "[DRY_RUN] wandb_mode=${WANDB_MODE} wandb_project=ctrlworld_action_following wandb_run_name=${RUN_NAME} require_online=${REQUIRE_WANDB_ONLINE}"
  exit 0
fi

for required in \
  "${LATENT_ROOT}/manifests/train.jsonl" \
  "${LATENT_ROOT}/manifests/test_quick.jsonl" \
  "${META_ROOT}/stat.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "[ERROR] prerequisite missing; run the place_burger_fries incremental precompute first: ${required}" >&2
    exit 2
  fi
done

if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[ERROR] refusing to reuse non-empty output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi

exec bash "${CODE_ROOT}/scripts/launch_training.sh" action_following_mix4
