#!/usr/bin/env bash
set -euo pipefail

# Ctrl-World ActionFollowingData 50-task precompute.
# Canonical enhanced HDF5/video are already semantic RGB; do not R/B swap.
# Default behavior reuses the 20260705 4-task safe cache by copying samples
# into the 50-task target root, then fills missing records with no-overwrite.

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

export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_afd_precompute_50task_chunk32_sharded_rgbfix_explore_20260706}"
export SVD_PATH="${SVD_PATH:-${ASSET_ROOT}/stable-video-diffusion-img2vid}"
export ENHANCED_SPLIT_ROOT="${ENHANCED_SPLIT_ROOT:-/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData/enhanced_v1_split}"
export CLEAN_LEROBOT_ROOT="${CLEAN_LEROBOT_ROOT:-/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingBench/data_lerobot/robotwin_delta_ee/demo_clean_zed2i_visible}"
export LATENT_ROOT="${LATENT_ROOT:-${CACHE_ROOT}/action_following_chunk32_clean_enhanced_explore_50task_20260706}"
export MIX4_META_ROOT="${MIX4_META_ROOT:-${TRAIN_PACKAGE_ROOT}/dataset_meta_info/action_following_chunk32_clean_enhanced_explore_50task_20260706}"
export CLEAN_META_ROOT="${CLEAN_META_ROOT:-${TRAIN_PACKAGE_ROOT}/dataset_meta_info/action_following_chunk32_clean_only_50task_rgbfix_explore_20260706}"
export LOG_ROOT="${LOG_ROOT:-${TRAIN_PACKAGE_ROOT}/logs/${RUN_NAME}}"
export SAFE_REUSE_ROOT="${SAFE_REUSE_ROOT:-${CACHE_ROOT}/action_following_chunk32_clean_enhanced_explore_test4v2_20260705}"
export NUM_SHARDS="${NUM_SHARDS:-16}"
export LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-${NUM_SHARDS}}"
export SHARD_START="${SHARD_START:-0}"
export SHARD_COUNT="${SHARD_COUNT:-${NUM_SHARDS}}"
export RUN_MERGE_AND_STAT="${RUN_MERGE_AND_STAT:-1}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export ACTION_DIM="${ACTION_DIM:-14}"
export CLEAR_TARGET="${CLEAR_TARGET:-0}"
export REUSE_SAFE_CACHE="${REUSE_SAFE_CACHE:-1}"
export WAIT_FOR_SAFE_REUSE_MARKER="${WAIT_FOR_SAFE_REUSE_MARKER:-0}"
export SAFE_REUSE_WAIT_TIMEOUT_SEC="${SAFE_REUSE_WAIT_TIMEOUT_SEC:-1800}"

MANIFEST_DIR="${LATENT_ROOT}/manifests"
SAFE_REUSE_MARKER="${LATENT_ROOT}/.safe_reuse_done"
mkdir -p "${MANIFEST_DIR}" "${MIX4_META_ROOT}" "${CLEAN_META_ROOT}" "${LOG_ROOT}"

if [[ ! -d "${SVD_PATH}" ]]; then
  echo "[ERROR] SVD_PATH does not exist: ${SVD_PATH}" >&2
  exit 2
fi
if [[ ! -f "${ENHANCED_SPLIT_ROOT}/manifests/train_samples.jsonl" ]]; then
  echo "[ERROR] missing enhanced train manifest under ${ENHANCED_SPLIT_ROOT}" >&2
  exit 2
fi
if (( SHARD_START < 0 || SHARD_START >= NUM_SHARDS )); then
  echo "[ERROR] SHARD_START must be in [0, NUM_SHARDS): ${SHARD_START}" >&2
  exit 2
fi
if (( SHARD_COUNT < 1 || SHARD_START + SHARD_COUNT > NUM_SHARDS )); then
  echo "[ERROR] invalid SHARD_COUNT=${SHARD_COUNT} for SHARD_START=${SHARD_START}, NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
fi

mapfile -t TASKS < <(
  "${PYTHON_BIN}" - <<'PY' "${ENHANCED_SPLIT_ROOT}/manifests/train_samples.jsonl"
import json
import sys

tasks = set()
with open(sys.argv[1]) as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        task = rec.get("task") or rec.get("task_name")
        if task:
            tasks.add(task)
for task in sorted(tasks):
    print(task)
PY
)
if [[ "${#TASKS[@]}" -ne 50 ]]; then
  echo "[ERROR] expected 50 tasks from enhanced manifest, got ${#TASKS[@]}" >&2
  printf 'tasks=%s\n' "${TASKS[*]}" >&2
  exit 2
fi

echo "[INFO] run=${RUN_NAME}"
echo "[INFO] code_root=${CODE_ROOT}"
echo "[INFO] python=${PYTHON_BIN}"
echo "[INFO] svd_path=${SVD_PATH}"
echo "[INFO] enhanced_split_root=${ENHANCED_SPLIT_ROOT}"
echo "[INFO] clean_lerobot_root=${CLEAN_LEROBOT_ROOT}"
echo "[INFO] latent_root=${LATENT_ROOT}"
echo "[INFO] mix4_meta_root=${MIX4_META_ROOT}"
echo "[INFO] clean_meta_root=${CLEAN_META_ROOT}"
echo "[INFO] safe_reuse_root=${SAFE_REUSE_ROOT}"
echo "[INFO] task_count=${#TASKS[@]}"
echo "[INFO] num_shards=${NUM_SHARDS}, shard_start=${SHARD_START}, shard_count=${SHARD_COUNT}, local_gpu_count=${LOCAL_GPU_COUNT}, batch_size=${BATCH_SIZE}"
printf '[INFO] tasks=%s\n' "${TASKS[*]}"
nvidia-smi || true

if [[ "${CLEAR_TARGET}" == "1" && "${SHARD_START}" == "0" ]]; then
  echo "[WARN] CLEAR_TARGET=1: removing target latent/meta/log roots"
  rm -rf "${LATENT_ROOT}" "${MIX4_META_ROOT}" "${CLEAN_META_ROOT}" "${LOG_ROOT}"
  mkdir -p "${MANIFEST_DIR}" "${MIX4_META_ROOT}" "${CLEAN_META_ROOT}" "${LOG_ROOT}"
elif [[ "${CLEAR_TARGET}" == "1" ]]; then
  echo "[WARN] CLEAR_TARGET ignored because SHARD_START=${SHARD_START}; only shard 0 may clear shared target roots"
fi

if [[ "${REUSE_SAFE_CACHE}" == "1" && "${SHARD_START}" == "0" && -d "${SAFE_REUSE_ROOT}/samples" ]]; then
  echo "[INFO] reusing safe 4-task cache samples from ${SAFE_REUSE_ROOT}"
  mkdir -p "${LATENT_ROOT}/samples"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --ignore-existing "${SAFE_REUSE_ROOT}/samples/" "${LATENT_ROOT}/samples/"
  elif cp --help 2>/dev/null | grep -q -- '--no-clobber'; then
    cp -a -n "${SAFE_REUSE_ROOT}/samples/." "${LATENT_ROOT}/samples/"
  else
    "${PYTHON_BIN}" - <<'PY' "${SAFE_REUSE_ROOT}/samples" "${LATENT_ROOT}/samples"
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
for path in src.rglob("*"):
    rel = path.relative_to(src)
    out = dst / rel
    if path.is_dir():
        out.mkdir(parents=True, exist_ok=True)
    elif not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
PY
  fi
  date -u +"%Y-%m-%dT%H:%M:%SZ" >"${SAFE_REUSE_MARKER}"
elif [[ "${REUSE_SAFE_CACHE}" == "1" && "${SHARD_START}" != "0" ]]; then
  echo "[INFO] safe cache reuse skipped on shard_start=${SHARD_START}; shard 0 handles shared copy"
else
  echo "[WARN] safe reuse disabled or missing: ${SAFE_REUSE_ROOT}/samples"
fi

if [[ "${WAIT_FOR_SAFE_REUSE_MARKER}" == "1" ]]; then
  echo "[INFO] waiting for safe reuse marker: ${SAFE_REUSE_MARKER}"
  wait_start="$(date +%s)"
  while [[ ! -f "${SAFE_REUSE_MARKER}" ]]; do
    elapsed_wait="$(( $(date +%s) - wait_start ))"
    if (( elapsed_wait > SAFE_REUSE_WAIT_TIMEOUT_SEC )); then
      echo "[ERROR] timed out waiting for safe reuse marker after ${elapsed_wait}s: ${SAFE_REUSE_MARKER}" >&2
      exit 2
    fi
    sleep 15
  done
  echo "[INFO] safe reuse marker found"
fi

if [[ "${SHARD_START}" == "0" ]]; then
  rm -f \
    "${MANIFEST_DIR}"/clean_train.jsonl \
    "${MANIFEST_DIR}"/enhanced_train.jsonl \
    "${MANIFEST_DIR}"/train.jsonl \
    "${MANIFEST_DIR}"/test_quick.jsonl \
    "${LATENT_ROOT}"/precompute_summary.json \
    "${LOG_ROOT}"/summary.txt
fi
for shard in $(seq "${SHARD_START}" "$((SHARD_START + SHARD_COUNT - 1))"); do
  suffix="$(printf 'shard%02d' "${shard}")"
  rm -f \
    "${MANIFEST_DIR}/clean_train_${suffix}.jsonl" \
    "${MANIFEST_DIR}/enhanced_train_${suffix}.jsonl" \
    "${MANIFEST_DIR}/test_quick_${suffix}.jsonl"
done

start_ts="$(date +%s)"

run_shard() {
  local shard="$1"
  local gpu="$(( shard % LOCAL_GPU_COUNT ))"
  local suffix
  suffix="$(printf 'shard%02d' "${shard}")"
  local log_file="${LOG_ROOT}/precompute_${suffix}.log"
  echo "[INFO] launch ${suffix} on cuda:${gpu}, log=${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_BIN}" "${CODE_ROOT}/scripts/precompute_latents_action_following.py" \
      --svd_path "${SVD_PATH}" \
      --out_root "${LATENT_ROOT}" \
      --enhanced_split_root "${ENHANCED_SPLIT_ROOT}" \
      --clean_lerobot_root "${CLEAN_LEROBOT_ROOT}" \
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
shard_ids=()
for shard in $(seq "${SHARD_START}" "$((SHARD_START + SHARD_COUNT - 1))"); do
  run_shard "${shard}" &
  pids+=("$!")
  shard_ids+=("${shard}")
done

failed=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    shard="${shard_ids[$idx]}"
    echo "[ERROR] shard ${shard} failed; tail follows:" >&2
    tail -n 120 "${LOG_ROOT}/precompute_$(printf 'shard%02d' "${shard}").log" >&2 || true
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

if [[ "${RUN_MERGE_AND_STAT}" != "1" ]]; then
  echo "[INFO] RUN_MERGE_AND_STAT=${RUN_MERGE_AND_STAT}; leaving shard manifests only"
  exit 0
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
elapsed="$(( $(date +%s) - start_ts ))"

cat >"${LATENT_ROOT}/precompute_summary.json" <<EOF
{
  "run_name": "${RUN_NAME}",
  "latent_root": "${LATENT_ROOT}",
  "mix4_meta_root": "${MIX4_META_ROOT}",
  "clean_meta_root": "${CLEAN_META_ROOT}",
  "task_count": ${#TASKS[@]},
  "num_shards": ${NUM_SHARDS},
  "batch_size": ${BATCH_SIZE},
  "train_records": ${train_count},
  "clean_train_records": ${clean_count},
  "enhanced_train_records": ${enhanced_count},
  "test_quick_records": ${quick_count},
  "elapsed_seconds": ${elapsed}
}
EOF

cat >"${LOG_ROOT}/summary.txt" <<EOF
run_name=${RUN_NAME}
latent_root=${LATENT_ROOT}
mix4_meta_root=${MIX4_META_ROOT}
clean_meta_root=${CLEAN_META_ROOT}
task_count=${#TASKS[@]}
num_shards=${NUM_SHARDS}
batch_size=${BATCH_SIZE}
train_records=${train_count}
clean_train_records=${clean_count}
enhanced_train_records=${enhanced_count}
test_quick_records=${quick_count}
elapsed_seconds=${elapsed}
EOF

cat "${LOG_ROOT}/summary.txt"
echo "[INFO] done"
