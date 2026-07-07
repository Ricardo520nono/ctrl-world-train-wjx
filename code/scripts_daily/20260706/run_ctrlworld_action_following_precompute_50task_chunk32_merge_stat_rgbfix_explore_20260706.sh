#!/usr/bin/env bash
set -euo pipefail

# Merge shard manifests and compute stats after both 8GPU precompute parts finish.

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

export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_afd_precompute_50task_chunk32_merge_stat_rgbfix_explore_20260706}"
export LATENT_ROOT="${LATENT_ROOT:-${CACHE_ROOT}/action_following_chunk32_clean_enhanced_explore_50task_20260706}"
export MIX4_META_ROOT="${MIX4_META_ROOT:-${TRAIN_PACKAGE_ROOT}/dataset_meta_info/action_following_chunk32_clean_enhanced_explore_50task_20260706}"
export CLEAN_META_ROOT="${CLEAN_META_ROOT:-${TRAIN_PACKAGE_ROOT}/dataset_meta_info/action_following_chunk32_clean_only_50task_rgbfix_explore_20260706}"
export LOG_ROOT="${LOG_ROOT:-${TRAIN_PACKAGE_ROOT}/logs/${RUN_NAME}}"
export NUM_SHARDS="${NUM_SHARDS:-16}"
export ACTION_DIM="${ACTION_DIM:-14}"

MANIFEST_DIR="${LATENT_ROOT}/manifests"
mkdir -p "${MANIFEST_DIR}" "${MIX4_META_ROOT}" "${CLEAN_META_ROOT}" "${LOG_ROOT}"

echo "[INFO] run=${RUN_NAME}"
echo "[INFO] latent_root=${LATENT_ROOT}"
echo "[INFO] mix4_meta_root=${MIX4_META_ROOT}"
echo "[INFO] clean_meta_root=${CLEAN_META_ROOT}"
echo "[INFO] num_shards=${NUM_SHARDS}"

for shard in $(seq 0 "$((NUM_SHARDS - 1))"); do
  suffix="$(printf 'shard%02d' "${shard}")"
  for split in clean_train enhanced_train test_quick; do
    path="${MANIFEST_DIR}/${split}_${suffix}.jsonl"
    if [[ ! -f "${path}" ]]; then
      echo "[ERROR] missing shard manifest: ${path}" >&2
      exit 2
    fi
  done
done

: >"${MANIFEST_DIR}/clean_train.jsonl"
for path in "${MANIFEST_DIR}"/clean_train_shard*.jsonl; do
  cat "${path}" >>"${MANIFEST_DIR}/clean_train.jsonl"
done

: >"${MANIFEST_DIR}/enhanced_train.jsonl"
for path in "${MANIFEST_DIR}"/enhanced_train_shard*.jsonl; do
  cat "${path}" >>"${MANIFEST_DIR}/enhanced_train.jsonl"
done

cat "${MANIFEST_DIR}/clean_train.jsonl" "${MANIFEST_DIR}/enhanced_train.jsonl" >"${MANIFEST_DIR}/train.jsonl"

: >"${MANIFEST_DIR}/test_quick.jsonl"
for path in "${MANIFEST_DIR}"/test_quick_shard*.jsonl; do
  cat "${path}" >>"${MANIFEST_DIR}/test_quick.jsonl"
done

echo "[INFO] computing mix4 train stat"
"${PYTHON_BIN}" "${CODE_ROOT}/scripts/compute_stat_action_following.py" \
  --latent_root "${LATENT_ROOT}" \
  --manifest "${MANIFEST_DIR}/train.jsonl" \
  --out_dir "${MIX4_META_ROOT}" \
  --action_dim "${ACTION_DIM}" | tee "${LOG_ROOT}/compute_stat_mix4.log"

echo "[INFO] computing clean-only stat"
"${PYTHON_BIN}" "${CODE_ROOT}/scripts/compute_stat_action_following.py" \
  --latent_root "${LATENT_ROOT}" \
  --manifest "${MANIFEST_DIR}/clean_train.jsonl" \
  --out_dir "${CLEAN_META_ROOT}" \
  --action_dim "${ACTION_DIM}" | tee "${LOG_ROOT}/compute_stat_clean.log"

train_count="$(wc -l <"${MANIFEST_DIR}/train.jsonl")"
clean_count="$(wc -l <"${MANIFEST_DIR}/clean_train.jsonl")"
enhanced_count="$(wc -l <"${MANIFEST_DIR}/enhanced_train.jsonl")"
quick_count="$(wc -l <"${MANIFEST_DIR}/test_quick.jsonl")"

cat >"${LATENT_ROOT}/precompute_summary.json" <<EOF
{
  "run_name": "${RUN_NAME}",
  "latent_root": "${LATENT_ROOT}",
  "mix4_meta_root": "${MIX4_META_ROOT}",
  "clean_meta_root": "${CLEAN_META_ROOT}",
  "num_shards": ${NUM_SHARDS},
  "train_records": ${train_count},
  "clean_train_records": ${clean_count},
  "enhanced_train_records": ${enhanced_count},
  "test_quick_records": ${quick_count}
}
EOF

cat >"${LOG_ROOT}/summary.txt" <<EOF
run_name=${RUN_NAME}
latent_root=${LATENT_ROOT}
mix4_meta_root=${MIX4_META_ROOT}
clean_meta_root=${CLEAN_META_ROOT}
num_shards=${NUM_SHARDS}
train_records=${train_count}
clean_train_records=${clean_count}
enhanced_train_records=${enhanced_count}
test_quick_records=${quick_count}
EOF

cat "${LOG_ROOT}/summary.txt"
echo "[INFO] done"
