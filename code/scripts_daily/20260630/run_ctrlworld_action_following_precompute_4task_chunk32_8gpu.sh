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

RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_afd_precompute_4task_chunk32_8gpu_20260630}"
SVD_PATH="${SVD_PATH:-${ASSET_ROOT}/stable-video-diffusion-img2vid}"
TASKS=(${TASKS:-place_can_basket blocks_ranking_size move_stapler_pad turn_switch})
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LATENT_ROOT="${LATENT_ROOT:-${CACHE_ROOT}/action_following_chunk32_clean_enhanced}"
META_ROOT="${META_ROOT:-${TRAIN_PACKAGE_ROOT}/dataset_meta_info/action_following_chunk32_clean_enhanced}"
LOG_ROOT="${LOG_ROOT:-${TRAIN_PACKAGE_ROOT}/logs/${RUN_NAME}}"
MANIFEST_DIR="${LATENT_ROOT}/manifests"

mkdir -p "${MANIFEST_DIR}" "${META_ROOT}" "${LOG_ROOT}"

if [[ ! -d "${SVD_PATH}" ]]; then
  echo "[ERROR] SVD_PATH does not exist: ${SVD_PATH}" >&2
  exit 2
fi

echo "[INFO] run=${RUN_NAME}"
echo "[INFO] code_root=${CODE_ROOT}"
echo "[INFO] python=${PYTHON_BIN}"
echo "[INFO] svd_path=${SVD_PATH}"
echo "[INFO] latent_root=${LATENT_ROOT}"
echo "[INFO] meta_root=${META_ROOT}"
echo "[INFO] tasks=${TASKS[*]}"
echo "[INFO] num_shards=${NUM_SHARDS}, batch_size=${BATCH_SIZE}"
nvidia-smi || true

rm -f \
  "${MANIFEST_DIR}"/clean_train_shard*.jsonl \
  "${MANIFEST_DIR}"/enhanced_train_shard*.jsonl \
  "${MANIFEST_DIR}"/test_quick_shard*.jsonl \
  "${MANIFEST_DIR}"/clean_train.jsonl \
  "${MANIFEST_DIR}"/enhanced_train.jsonl \
  "${MANIFEST_DIR}"/train.jsonl \
  "${MANIFEST_DIR}"/test_quick.jsonl

start_ts="$(date +%s)"

run_shard() {
  local shard="$1"
  local gpu="$2"
  local suffix
  suffix="$(printf 'shard%02d' "${shard}")"
  local log_file="${LOG_ROOT}/precompute_${suffix}.log"
  echo "[INFO] launch ${suffix} on cuda:${gpu}, log=${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_BIN}" "${CODE_ROOT}/scripts/precompute_latents_action_following.py" \
      --svd_path "${SVD_PATH}" \
      --out_root "${LATENT_ROOT}" \
      --tasks "${TASKS[@]}" \
      --split both \
      --include_clean \
      --include_enhanced \
      --batch_size "${BATCH_SIZE}" \
      --num_shards "${NUM_SHARDS}" \
      --shard_index "${shard}" \
      --manifest_suffix "${suffix}" \
      --skip_train_merge
  ) >"${log_file}" 2>&1
}

pids=()
for shard in $(seq 0 "$((NUM_SHARDS - 1))"); do
  run_shard "${shard}" "${shard}" &
  pids+=("$!")
done

failed=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "[ERROR] shard ${idx} failed; tail follows:" >&2
    tail -n 80 "${LOG_ROOT}/precompute_$(printf 'shard%02d' "${idx}").log" >&2 || true
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

: >"${MANIFEST_DIR}/clean_train.jsonl"
for path in "${MANIFEST_DIR}"/clean_train_shard*.jsonl; do
  [[ -f "${path}" ]] && cat "${path}" >>"${MANIFEST_DIR}/clean_train.jsonl"
done

: >"${MANIFEST_DIR}/enhanced_train.jsonl"
for path in "${MANIFEST_DIR}"/enhanced_train_shard*.jsonl; do
  [[ -f "${path}" ]] && cat "${path}" >>"${MANIFEST_DIR}/enhanced_train.jsonl"
done

cat "${MANIFEST_DIR}/clean_train.jsonl" "${MANIFEST_DIR}/enhanced_train.jsonl" >"${MANIFEST_DIR}/train.jsonl"

: >"${MANIFEST_DIR}/test_quick.jsonl"
for path in "${MANIFEST_DIR}"/test_quick_shard*.jsonl; do
  [[ -f "${path}" ]] && cat "${path}" >>"${MANIFEST_DIR}/test_quick.jsonl"
done

"${PYTHON_BIN}" "${CODE_ROOT}/scripts/compute_stat_action_following.py" \
  --latent_root "${LATENT_ROOT}" \
  --manifest "${MANIFEST_DIR}/train.jsonl" \
  --out_dir "${META_ROOT}" \
  --action_dim 14 | tee "${LOG_ROOT}/compute_stat.log"

train_count="$(wc -l <"${MANIFEST_DIR}/train.jsonl")"
clean_count="$(wc -l <"${MANIFEST_DIR}/clean_train.jsonl")"
enhanced_count="$(wc -l <"${MANIFEST_DIR}/enhanced_train.jsonl")"
quick_count="$(wc -l <"${MANIFEST_DIR}/test_quick.jsonl")"
elapsed="$(( $(date +%s) - start_ts ))"

cat >"${LOG_ROOT}/summary.txt" <<EOF
run_name=${RUN_NAME}
latent_root=${LATENT_ROOT}
meta_root=${META_ROOT}
train_records=${train_count}
clean_train_records=${clean_count}
enhanced_train_records=${enhanced_count}
test_quick_records=${quick_count}
elapsed_seconds=${elapsed}
EOF

cat "${LOG_ROOT}/summary.txt"
echo "[INFO] done"
