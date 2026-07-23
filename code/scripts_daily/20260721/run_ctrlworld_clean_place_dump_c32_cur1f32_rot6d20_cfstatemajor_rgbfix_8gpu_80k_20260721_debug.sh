#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python}"
export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_clean_place_dump_c32_cur1f32_rot6d20_cfstatemajor_rgbfix1_8gpu_80k_20260721_debug}"
export TASKS="place_burger_fries dump_bin_bigbin"
export PUBLIC_ROOT="${PUBLIC_ROOT:-/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world}"
export LATENT_ROOT="${LATENT_ROOT:-${PUBLIC_ROOT}/latents/action_following_cur1f32_mix4_place_dump_rot6d20_cfstatemajor_rgbfix1_20260719_debug}"
export META_ROOT="${META_ROOT:-${PUBLIC_ROOT}/dataset_meta_info/action_following_cur1f32_clean_place_dump_rot6d20_cfstatemajor_rgbfix1_20260721_debug}"
export STAT_MANIFEST="${STAT_MANIFEST:-${LATENT_ROOT}/manifests/clean_train.jsonl}"
export TRAIN_MANIFEST_NAME=clean_train.jsonl
export VAL_MANIFEST_NAME=test_quick.jsonl
export OUTPUT_DIR="${OUTPUT_DIR:-${PUBLIC_ROOT}/outputs/${RUN_NAME}}"
export ACTION_DIM=20
export CHUNK_SIZE=32
export NUM_HISTORY=1
export NUM_FRAMES=32
export TRAIN_BATCH_SIZE=1
export GRAD_ACCUM=2
export NPROC_PER_NODE=8
export MAX_TRAIN_STEPS=80000
export VALIDATION_STEPS=2500
export CHECKPOINTING_STEPS=2500
export CHECKPOINTING_EPOCHS=0
export LEARNING_RATE=1e-5
export MIXED_PRECISION=bf16
export WANDB_MODE=online
export REQUIRE_WANDB_ONLINE=1
export SWANLAB_MODE=local
export SAMPLER_AUDIT_SAMPLES=10000
export SAMPLER_AUDIT_SEED=20260721
export SAMPLER_AUDIT_TOLERANCE=0.02
export ACTION_FOLLOWING_TASK_BALANCED=1
export CKPT_PATH=none
export MASTER_PORT="${MASTER_PORT:-29761}"

if [[ "${RUN_NAME}" != *_debug ]]; then
  echo "[ERROR] debug run name must end with _debug: ${RUN_NAME}" >&2
  exit 2
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] run=${RUN_NAME} tasks=${TASKS}"
  echo "[DRY_RUN] latent_root=${LATENT_ROOT} meta_root=${META_ROOT} output=${OUTPUT_DIR}"
  echo "[DRY_RUN] protocol=clean_only train_manifest=${STAT_MANIFEST}"
  echo "[DRY_RUN] steps=80000 batch_per_gpu=1 grad_accum=2 global_bs=16 lr=1e-5 bf16 ckpt=none"
  exit 0
fi

for required in \
  "${LATENT_ROOT}/.merged_complete" \
  "${LATENT_ROOT}/merge_summary.json" \
  "${LATENT_ROOT}/merged_cache_audit.json" \
  "${LATENT_ROOT}/manifests/clean_train.jsonl" \
  "${LATENT_ROOT}/manifests/test_quick.jsonl"; do
  if [[ ! -f "${required}" ]]; then
    echo "[ERROR] RGB-fix clean-only prerequisite missing: ${required}" >&2
    exit 2
  fi
done
"${PYTHON_BIN}" "${CODE_ROOT}/scripts/merge_action_following_task_caches.py" \
  --audit_only \
  --out_root "${LATENT_ROOT}" \
  --stat_path "${PUBLIC_ROOT}/dataset_meta_info/action_following_cur1f32_mix4_place_dump_rot6d20_cfstatemajor_rgbfix1_20260719_debug/stat.json" \
  --audit_out "${LATENT_ROOT}/merged_cache_audit.clean_reuse.json"
if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[ERROR] refusing to reuse non-empty output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi

exec bash "${CODE_ROOT}/scripts/launch_training.sh" action_following_clean
