#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}"

source "${CODE_ROOT}/scripts/ctrlworld_train_env.sh"
if [[ "${PYTHON_BIN}" == "/usr/bin/python3" && -x "/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python" ]]; then
  PYTHON_BIN="/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

: "${TASK:?TASK must be set by a task-specific wrapper}"
: "${RUN_NAME:?RUN_NAME must be set by a task-specific wrapper}"

export SVD_PATH="${SVD_PATH:-${ASSET_ROOT}/stable-video-diffusion-img2vid}"
export ENHANCED_SPLIT_ROOT="${ENHANCED_SPLIT_ROOT:-/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData_Rot6D/enhanced_v1_split}"
export OLD_CACHE_ROOT="${OLD_CACHE_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/latents/action_following_current1_future32_clean_enhanced_explore_50task_fulldesc_rot6d20_frameidxfix_20260711}"
export TASK_INSTRUCTION_ROOT="${TASK_INSTRUCTION_ROOT:-/mnt/dataset/csx_workspace/Ideas/AF3/code/RoboTwin/description/task_instruction}"
export REPAIR_ROOT="${REPAIR_ROOT:-/mnt/dataset/public_data/cscsx_projects/AF3/counterfactual_replay_state_major/rgb_fix_20260719}"
export PUBLIC_ROOT="${PUBLIC_ROOT:-/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world}"
export LATENT_ROOT="${LATENT_ROOT:-${PUBLIC_ROOT}/latents/action_following_cur1f32_mix4_${TASK}_rot6d20_cfstatemajor_rgbfix1_20260719}"
export META_ROOT="${META_ROOT:-${PUBLIC_ROOT}/dataset_meta_info/action_following_cur1f32_mix4_${TASK}_rot6d20_cfstatemajor_rgbfix1_20260719}"
export LOG_ROOT="${LOG_ROOT:-${PUBLIC_ROOT}/logs/${RUN_NAME}}"
export DELTA_ORIGIN="${DELTA_ORIGIN:-state_major_rgbfix_20260719}"
export OTHER_DELTA_ORIGIN="${OTHER_DELTA_ORIGIN:-task_local_incremental_rgbfix_20260719}"
export NUM_SHARDS="${NUM_SHARDS:-8}"
export LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-8}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export ACTION_DIM="${ACTION_DIM:-20}"
export CHUNK_SIZE="${CHUNK_SIZE:-32}"
export NUM_HISTORY="${NUM_HISTORY:-1}"
export NUM_FRAMES="${NUM_FRAMES:-32}"
export SAMPLER_AUDIT_SAMPLES="${SAMPLER_AUDIT_SAMPLES:-10000}"
export SAMPLER_AUDIT_SEED="${SAMPLER_AUDIT_SEED:-20260718}"
export SAMPLER_AUDIT_TOLERANCE="${SAMPLER_AUDIT_TOLERANCE:-0.02}"

MANIFEST_ROOT="${LATENT_ROOT}/manifests"
SELECTION_ROOT="${LATENT_ROOT}/selection_manifests"

for path in "${SVD_PATH}" "${ENHANCED_SPLIT_ROOT}" "${OLD_CACHE_ROOT}" "${TASK_INSTRUCTION_ROOT}" "${REPAIR_ROOT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] required input does not exist: ${path}" >&2
    exit 2
  fi
done
if [[ "${NUM_SHARDS}" -ne 8 || "${LOCAL_GPU_COUNT}" -ne 8 ]]; then
  echo "[ERROR] this launcher requires NUM_SHARDS=LOCAL_GPU_COUNT=8" >&2
  exit 2
fi
if [[ "${ACTION_DIM}" -ne 20 || "${CHUNK_SIZE}" -ne 32 || "${NUM_HISTORY}" -ne 1 || "${NUM_FRAMES}" -ne 32 ]]; then
  echo "[ERROR] expected action20/chunk32/current1/future32" >&2
  exit 2
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] run=${RUN_NAME} task=${TASK}"
  echo "[DRY_RUN] old_cache_root=${OLD_CACHE_ROOT}"
  echo "[DRY_RUN] enhanced_split_root=${ENHANCED_SPLIT_ROOT}"
  echo "[DRY_RUN] repair_root=${REPAIR_ROOT} delta_origin=${DELTA_ORIGIN} other_delta_origin=${OTHER_DELTA_ORIGIN}"
  echo "[DRY_RUN] latent_root=${LATENT_ROOT} meta_root=${META_ROOT}"
  echo "[DRY_RUN] action_dim=${ACTION_DIM} chunk_size=${CHUNK_SIZE} num_history=${NUM_HISTORY} num_frames=${NUM_FRAMES}"
  exit 0
fi

if [[ -f "${LATENT_ROOT}/.incremental_complete" ]]; then
  echo "[ERROR] RGB-fix cache is already complete; refusing to overwrite: ${LATENT_ROOT}" >&2
  exit 2
fi

mkdir -p "${MANIFEST_ROOT}" "${META_ROOT}" "${LOG_ROOT}"
"${PYTHON_BIN}" "${CODE_ROOT}/scripts/prepare_action_following_incremental_cache.py" plan \
  --task "${TASK}" \
  --old_cache_root "${OLD_CACHE_ROOT}" \
  --enhanced_split_root "${ENHANCED_SPLIT_ROOT}" \
  --out_root "${LATENT_ROOT}" \
  --delta_origin "${DELTA_ORIGIN}" \
  --other_delta_origin "${OTHER_DELTA_ORIGIN}" | tee "${LOG_ROOT}/incremental_plan.log"

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/preflight_action_following_counterfactual_rgb_fix.py" \
  --mode source \
  --task "${TASK}" \
  --repair_root "${REPAIR_ROOT}" \
  --enhanced_split_root "${ENHANCED_SPLIT_ROOT}" \
  --train_selection_manifest "${SELECTION_ROOT}/train_delta_source.jsonl" \
  --quick_selection_manifest "${SELECTION_ROOT}/test_quick_delta_source.jsonl" \
  --out "${LATENT_ROOT}/rgb_preflight_source.json" | tee "${LOG_ROOT}/rgb_preflight_source.log"

run_shard() {
  local shard="$1"
  local suffix
  suffix="$(printf 'delta_shard%02d' "${shard}")"
  local log_file="${LOG_ROOT}/precompute_${suffix}.log"
  echo "[INFO] launch ${suffix} on cuda:${shard}"
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    "${PYTHON_BIN}" "${CODE_ROOT}/scripts/precompute_latents_action_following.py" \
      --svd_path "${SVD_PATH}" \
      --out_root "${LATENT_ROOT}" \
      --enhanced_split_root "${ENHANCED_SPLIT_ROOT}" \
      --enhanced_train_manifest_path "${SELECTION_ROOT}/train_delta_source.jsonl" \
      --enhanced_test_quick_manifest_path "${SELECTION_ROOT}/test_quick_delta_source.jsonl" \
      --task_instruction_root "${TASK_INSTRUCTION_ROOT}" \
      --tasks "${TASK}" \
      --split both \
      --include_enhanced \
      --batch_size "${BATCH_SIZE}" \
      --num_shards "${NUM_SHARDS}" \
      --shard_index "${shard}" \
      --manifest_suffix "${suffix}" \
      --skip_train_merge
  ) >"${log_file}" 2>&1
}

nvidia-smi || true
start_ts="$(date +%s)"
pids=()
for shard in $(seq 0 7); do
  run_shard "${shard}" &
  pids+=("$!")
done

failed=0
for shard in $(seq 0 7); do
  if ! wait "${pids[$shard]}"; then
    echo "[ERROR] delta shard ${shard} failed" >&2
    tail -n 120 "${LOG_ROOT}/precompute_$(printf 'delta_shard%02d' "${shard}").log" >&2 || true
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/prepare_action_following_incremental_cache.py" finalize \
  --task "${TASK}" \
  --old_cache_root "${OLD_CACHE_ROOT}" \
  --enhanced_split_root "${ENHANCED_SPLIT_ROOT}" \
  --out_root "${LATENT_ROOT}" \
  --delta_origin "${DELTA_ORIGIN}" \
  --other_delta_origin "${OTHER_DELTA_ORIGIN}" \
  --num_shards "${NUM_SHARDS}" | tee "${LOG_ROOT}/incremental_finalize.log"

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/compute_stat_action_following.py" \
  --latent_root "${LATENT_ROOT}" \
  --manifest "${MANIFEST_ROOT}/train.jsonl" \
  --out_dir "${META_ROOT}" \
  --action_dim "${ACTION_DIM}" | tee "${LOG_ROOT}/compute_stat.log"

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/audit_action_following_sampler.py" \
  --latent_root "${LATENT_ROOT}" \
  --train_manifest "${MANIFEST_ROOT}/train.jsonl" \
  --val_manifest "${MANIFEST_ROOT}/test_quick.jsonl" \
  --stat_path "${META_ROOT}/stat.json" \
  --protocol mix_4to1to1to1to1 \
  --chunk_size "${CHUNK_SIZE}" \
  --num_history "${NUM_HISTORY}" \
  --num_frames "${NUM_FRAMES}" \
  --action_dim "${ACTION_DIM}" \
  --num_samples "${SAMPLER_AUDIT_SAMPLES}" \
  --seed "${SAMPLER_AUDIT_SEED}" \
  --tolerance "${SAMPLER_AUDIT_TOLERANCE}" | tee "${LOG_ROOT}/sampler_audit.log"

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/preflight_action_following_counterfactual_rgb_fix.py" \
  --mode cache \
  --task "${TASK}" \
  --repair_root "${REPAIR_ROOT}" \
  --latent_root "${LATENT_ROOT}" \
  --expected_origin "${DELTA_ORIGIN}" \
  --out "${LATENT_ROOT}/rgb_preflight_cache.json" | tee "${LOG_ROOT}/rgb_preflight_cache.log"

elapsed="$(( $(date +%s) - start_ts ))"
cat >"${LATENT_ROOT}/incremental_run_summary.json" <<EOF
{
  "run_name": "${RUN_NAME}",
  "task": "${TASK}",
  "old_cache_root": "${OLD_CACHE_ROOT}",
  "latent_root": "${LATENT_ROOT}",
  "meta_root": "${META_ROOT}",
  "repair_root": "${REPAIR_ROOT}",
  "delta_origin": "${DELTA_ORIGIN}",
  "other_delta_origin": "${OTHER_DELTA_ORIGIN}",
  "num_shards": ${NUM_SHARDS},
  "action_dim": ${ACTION_DIM},
  "chunk_size": ${CHUNK_SIZE},
  "num_history": ${NUM_HISTORY},
  "num_frames": ${NUM_FRAMES},
  "elapsed_seconds": ${elapsed},
  "status": "complete"
}
EOF
date -u +"%Y-%m-%dT%H:%M:%SZ" >"${LATENT_ROOT}/.incremental_complete"

echo "[INFO] RGB-fix incremental cache complete: ${LATENT_ROOT}"
