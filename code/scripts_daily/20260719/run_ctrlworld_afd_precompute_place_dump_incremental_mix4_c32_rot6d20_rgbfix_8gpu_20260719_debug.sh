#!/usr/bin/env bash
set -euo pipefail

DAILY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${DAILY_DIR}/../.." && pwd)"
cd "${CODE_ROOT}"

source "${CODE_ROOT}/scripts/ctrlworld_train_env.sh"
if [[ "${PYTHON_BIN}" == "/usr/bin/python3" && -x "/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python" ]]; then
  PYTHON_BIN="/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python"
fi

export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_afd_precompute_place_dump_c32_rot6d20_cfstatemajor_rgbfix1_8gpu_20260719_debug}"
export PUBLIC_ROOT="${PUBLIC_ROOT:-/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world}"
export PLACE_CACHE_ROOT="${PLACE_CACHE_ROOT:-${PUBLIC_ROOT}/latents/action_following_cur1f32_mix4_place_burger_fries_rot6d20_cfstatemajor_rgbfix1_20260719_debug}"
export DUMP_CACHE_ROOT="${DUMP_CACHE_ROOT:-${PUBLIC_ROOT}/latents/action_following_cur1f32_mix4_dump_bin_bigbin_rot6d20_cfstatemajor_rgbfix1_20260719_debug}"
export MERGED_LATENT_ROOT="${MERGED_LATENT_ROOT:-${PUBLIC_ROOT}/latents/action_following_cur1f32_mix4_place_dump_rot6d20_cfstatemajor_rgbfix1_20260719_debug}"
export MERGED_META_ROOT="${MERGED_META_ROOT:-${PUBLIC_ROOT}/dataset_meta_info/action_following_cur1f32_mix4_place_dump_rot6d20_cfstatemajor_rgbfix1_20260719_debug}"
export LOG_ROOT="${LOG_ROOT:-${PUBLIC_ROOT}/logs/${RUN_NAME}}"
STAGE_DRIVER="${DAILY_DIR}/run_ctrlworld_afd_precompute_single_task_incremental_mix4_c32_rot6d20_rgbfix_8gpu_20260719.sh"

if [[ ! -x "${STAGE_DRIVER}" ]]; then
  echo "[ERROR] internal stage driver is not executable: ${STAGE_DRIVER}" >&2
  exit 2
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] run=${RUN_NAME}"
  echo "[DRY_RUN] sequential_task_precompute=place_burger_fries,dump_bin_bigbin on one 8GPU node"
  echo "[DRY_RUN] place_cache=${PLACE_CACHE_ROOT} dump_cache=${DUMP_CACHE_ROOT}"
  echo "[DRY_RUN] merged_latent=${MERGED_LATENT_ROOT} merged_meta=${MERGED_META_ROOT}"
  echo "[DRY_RUN] final_audit=RGB source/cache + action20 stat + 10000 task-balanced Mix4 draws"
  exit 0
fi

for path in "${PLACE_CACHE_ROOT}" "${DUMP_CACHE_ROOT}" "${MERGED_LATENT_ROOT}"; do
  if [[ -e "${path}" ]]; then
    echo "[ERROR] refusing to reuse existing RGB-fix debug path: ${path}" >&2
    exit 2
  fi
done
for path in \
  "${PUBLIC_ROOT}/dataset_meta_info/action_following_cur1f32_mix4_place_burger_fries_rot6d20_cfstatemajor_rgbfix1_20260719_debug" \
  "${PUBLIC_ROOT}/dataset_meta_info/action_following_cur1f32_mix4_dump_bin_bigbin_rot6d20_cfstatemajor_rgbfix1_20260719_debug" \
  "${MERGED_META_ROOT}" \
  "${LOG_ROOT}/place_stage" \
  "${LOG_ROOT}/dump_stage"; do
  if [[ -e "${path}" ]]; then
    echo "[ERROR] refusing to reuse existing RGB-fix debug path: ${path}" >&2
    exit 2
  fi
done
mkdir -p "${LOG_ROOT}"

TASK=place_burger_fries \
RUN_NAME="${RUN_NAME}_place_stage" \
LATENT_ROOT="${PLACE_CACHE_ROOT}" \
META_ROOT="${PUBLIC_ROOT}/dataset_meta_info/action_following_cur1f32_mix4_place_burger_fries_rot6d20_cfstatemajor_rgbfix1_20260719_debug" \
LOG_ROOT="${LOG_ROOT}/place_stage" \
bash "${STAGE_DRIVER}"

TASK=dump_bin_bigbin \
RUN_NAME="${RUN_NAME}_dump_stage" \
LATENT_ROOT="${DUMP_CACHE_ROOT}" \
META_ROOT="${PUBLIC_ROOT}/dataset_meta_info/action_following_cur1f32_mix4_dump_bin_bigbin_rot6d20_cfstatemajor_rgbfix1_20260719_debug" \
LOG_ROOT="${LOG_ROOT}/dump_stage" \
bash "${STAGE_DRIVER}"

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/merge_action_following_task_caches.py" \
  --task_cache "place_burger_fries=${PLACE_CACHE_ROOT}" \
  --task_cache "dump_bin_bigbin=${DUMP_CACHE_ROOT}" \
  --out_root "${MERGED_LATENT_ROOT}" | tee "${LOG_ROOT}/merge.log"

mkdir -p "${MERGED_META_ROOT}"
"${PYTHON_BIN}" "${CODE_ROOT}/scripts/compute_stat_action_following.py" \
  --latent_root "${MERGED_LATENT_ROOT}" \
  --manifest "${MERGED_LATENT_ROOT}/manifests/train.jsonl" \
  --out_dir "${MERGED_META_ROOT}" \
  --action_dim 20 | tee "${LOG_ROOT}/compute_stat.log"

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/audit_action_following_sampler.py" \
  --latent_root "${MERGED_LATENT_ROOT}" \
  --train_manifest "${MERGED_LATENT_ROOT}/manifests/train.jsonl" \
  --val_manifest "${MERGED_LATENT_ROOT}/manifests/test_quick.jsonl" \
  --stat_path "${MERGED_META_ROOT}/stat.json" \
  --protocol mix_4to1to1to1to1 \
  --chunk_size 32 \
  --num_history 1 \
  --num_frames 32 \
  --action_dim 20 \
  --num_samples 10000 \
  --seed 20260718 \
  --tolerance 0.02 \
  --task_balanced | tee "${LOG_ROOT}/sampler_audit_task_balanced.log"

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/merge_action_following_task_caches.py" \
  --audit_only \
  --out_root "${MERGED_LATENT_ROOT}" \
  --stat_path "${MERGED_META_ROOT}/stat.json" \
  --audit_out "${MERGED_LATENT_ROOT}/merged_cache_audit.json" | tee "${LOG_ROOT}/merged_cache_audit.log"

date -u +"%Y-%m-%dT%H:%M:%SZ" >"${MERGED_LATENT_ROOT}/.merged_complete"
echo "[INFO] merged RGB-fix debug cache complete: ${MERGED_LATENT_ROOT}"
