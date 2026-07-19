#!/usr/bin/env bash
set -euo pipefail

# Ctrl-World ActionFollowingData 50-task Rot6D20 precompute after clean LeRobot frame-index fix.
# Shared cache for clean-only and mix4 training:
#   - clean_train: 50-task clean LeRobot Rot6D actions + legacy clean videos read with global parquet index
#   - enhanced_train: Rot6D enhanced families, including exploration
#   - test_quick: Rot6D enhanced test_quick for validation
# The image latents are VAE outputs, but cached samples also carry action_pos;
# therefore old 14D caches are not reused for 20D Rot6D training.

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

export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_afd_precompute_50task_chunk32_rot6d20_frameidxfix_sharded_20260711}"
export SVD_PATH="${SVD_PATH:-${ASSET_ROOT}/stable-video-diffusion-img2vid}"
export ENHANCED_SPLIT_ROOT="${ENHANCED_SPLIT_ROOT:-/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData_Rot6D/enhanced_v1_split}"
export CLEAN_LEROBOT_ROOT="${CLEAN_LEROBOT_ROOT:-/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData_LeRobot_Rot6D/train/demo_clean_zed2i_visible}"
export CLEAN_LEROBOT_VIDEO_ROOT="${CLEAN_LEROBOT_VIDEO_ROOT:-/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingBench/data_lerobot/robotwin_delta_ee/demo_clean_zed2i_visible}"
export TASK_INSTRUCTION_ROOT="${TASK_INSTRUCTION_ROOT:-/mnt/dataset/csx_workspace/Ideas/AF3/code/RoboTwin/description/task_instruction}"
export LATENT_ROOT="${LATENT_ROOT:-${CACHE_ROOT}/action_following_current1_future32_clean_enhanced_explore_50task_fulldesc_rot6d20_frameidxfix_20260711}"
export MIX4_META_ROOT="${MIX4_META_ROOT:-${TRAIN_PACKAGE_ROOT}/dataset_meta_info/action_following_current1_future32_clean_enhanced_explore_50task_fulldesc_rot6d20_frameidxfix_20260711}"
export CLEAN_META_ROOT="${CLEAN_META_ROOT:-${TRAIN_PACKAGE_ROOT}/dataset_meta_info/action_following_current1_future32_clean_only_50task_fulldesc_rot6d20_frameidxfix_20260711}"
export LOG_ROOT="${LOG_ROOT:-${TRAIN_PACKAGE_ROOT}/logs/${RUN_NAME}}"
export NUM_SHARDS="${NUM_SHARDS:-16}"
export LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-${NUM_SHARDS}}"
export SHARD_START="${SHARD_START:-0}"
export SHARD_COUNT="${SHARD_COUNT:-${NUM_SHARDS}}"
export RUN_MERGE_AND_STAT="${RUN_MERGE_AND_STAT:-1}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export ACTION_DIM="${ACTION_DIM:-20}"
export CHUNK_SIZE="${CHUNK_SIZE:-32}"
export NUM_HISTORY="${NUM_HISTORY:-1}"
export NUM_FRAMES="${NUM_FRAMES:-32}"
export CLEAR_TARGET="${CLEAR_TARGET:-0}"
export WAIT_FOR_TARGET_READY="${WAIT_FOR_TARGET_READY:-0}"
export TARGET_READY_WAIT_TIMEOUT_SEC="${TARGET_READY_WAIT_TIMEOUT_SEC:-1800}"
export EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT:-50}"
export WAIT_FOR_ALL_SHARDS_BEFORE_MERGE="${WAIT_FOR_ALL_SHARDS_BEFORE_MERGE:-0}"
export ALL_SHARDS_WAIT_TIMEOUT_SEC="${ALL_SHARDS_WAIT_TIMEOUT_SEC:-86400}"
export DRY_RUN="${DRY_RUN:-0}"

MANIFEST_DIR="${LATENT_ROOT}/manifests"
TARGET_READY_MARKER="${LATENT_ROOT}/.target_ready"

if [[ ! -d "${SVD_PATH}" ]]; then
  echo "[ERROR] SVD_PATH does not exist: ${SVD_PATH}" >&2
  exit 2
fi
if [[ ! -f "${ENHANCED_SPLIT_ROOT}/manifests/train_samples.jsonl" ]]; then
  echo "[ERROR] missing enhanced train manifest under ${ENHANCED_SPLIT_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${CLEAN_LEROBOT_ROOT}" ]]; then
  echo "[ERROR] CLEAN_LEROBOT_ROOT does not exist: ${CLEAN_LEROBOT_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${CLEAN_LEROBOT_VIDEO_ROOT}" ]]; then
  echo "[ERROR] CLEAN_LEROBOT_VIDEO_ROOT does not exist: ${CLEAN_LEROBOT_VIDEO_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${TASK_INSTRUCTION_ROOT}" ]]; then
  echo "[ERROR] TASK_INSTRUCTION_ROOT does not exist: ${TASK_INSTRUCTION_ROOT}" >&2
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
if [[ "${ACTION_DIM}" -ne 20 ]]; then
  echo "[ERROR] this Rot6D20 precompute expects ACTION_DIM=20, got ${ACTION_DIM}" >&2
  exit 2
fi

if [[ -n "${TASKS:-}" ]]; then
  read -r -a TASKS <<<"${TASKS}"
else
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
fi
if [[ "${#TASKS[@]}" -ne "${EXPECTED_TASK_COUNT}" ]]; then
  echo "[ERROR] expected ${EXPECTED_TASK_COUNT} tasks, got ${#TASKS[@]}" >&2
  printf 'tasks=%s\n' "${TASKS[*]}" >&2
  exit 2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] run=${RUN_NAME}"
  echo "[DRY_RUN] code_root=${CODE_ROOT}"
  echo "[DRY_RUN] python=${PYTHON_BIN}"
  echo "[DRY_RUN] svd_path=${SVD_PATH}"
  echo "[DRY_RUN] enhanced_split_root=${ENHANCED_SPLIT_ROOT}"
  echo "[DRY_RUN] clean_lerobot_root=${CLEAN_LEROBOT_ROOT}"
  echo "[DRY_RUN] clean_lerobot_video_root=${CLEAN_LEROBOT_VIDEO_ROOT}"
  echo "[DRY_RUN] task_instruction_root=${TASK_INSTRUCTION_ROOT}"
  echo "[DRY_RUN] latent_root=${LATENT_ROOT}"
  echo "[DRY_RUN] mix4_meta_root=${MIX4_META_ROOT}"
  echo "[DRY_RUN] clean_meta_root=${CLEAN_META_ROOT}"
  echo "[DRY_RUN] task_count=${#TASKS[@]}"
  echo "[DRY_RUN] num_shards=${NUM_SHARDS}, shard_start=${SHARD_START}, shard_count=${SHARD_COUNT}, local_gpu_count=${LOCAL_GPU_COUNT}, batch_size=${BATCH_SIZE}"
  echo "[DRY_RUN] action_dim=${ACTION_DIM}, chunk_size=${CHUNK_SIZE}, num_history=${NUM_HISTORY}, num_frames=${NUM_FRAMES}"
  printf '[DRY_RUN] tasks=%s\n' "${TASKS[*]}"
  exit 0
fi

if [[ "${WAIT_FOR_TARGET_READY}" == "1" && "${SHARD_START}" != "0" ]]; then
  echo "[INFO] waiting for target-ready marker: ${TARGET_READY_MARKER}"
  wait_start="$(date +%s)"
  while [[ ! -f "${TARGET_READY_MARKER}" ]]; do
    elapsed_wait="$(( $(date +%s) - wait_start ))"
    if (( elapsed_wait > TARGET_READY_WAIT_TIMEOUT_SEC )); then
      echo "[ERROR] timed out waiting for target-ready marker after ${elapsed_wait}s: ${TARGET_READY_MARKER}" >&2
      exit 2
    fi
    sleep 15
  done
  echo "[INFO] target-ready marker found"
fi

mkdir -p "${MANIFEST_DIR}" "${MIX4_META_ROOT}" "${CLEAN_META_ROOT}" "${LOG_ROOT}"

if [[ "${CLEAR_TARGET}" == "1" && "${SHARD_START}" == "0" ]]; then
  echo "[WARN] CLEAR_TARGET=1: removing target latent/meta/log roots"
  rm -rf "${LATENT_ROOT}" "${MIX4_META_ROOT}" "${CLEAN_META_ROOT}" "${LOG_ROOT}"
  mkdir -p "${MANIFEST_DIR}" "${MIX4_META_ROOT}" "${CLEAN_META_ROOT}" "${LOG_ROOT}"
elif [[ "${CLEAR_TARGET}" == "1" ]]; then
  echo "[WARN] CLEAR_TARGET ignored because SHARD_START=${SHARD_START}; only shard 0 may clear shared target roots"
fi
if [[ "${SHARD_START}" == "0" ]]; then
  date -u +"%Y-%m-%dT%H:%M:%SZ" >"${TARGET_READY_MARKER}"
fi

echo "[INFO] run=${RUN_NAME}"
echo "[INFO] code_root=${CODE_ROOT}"
echo "[INFO] python=${PYTHON_BIN}"
echo "[INFO] svd_path=${SVD_PATH}"
echo "[INFO] enhanced_split_root=${ENHANCED_SPLIT_ROOT}"
echo "[INFO] clean_lerobot_root=${CLEAN_LEROBOT_ROOT}"
echo "[INFO] clean_lerobot_video_root=${CLEAN_LEROBOT_VIDEO_ROOT}"
echo "[INFO] task_instruction_root=${TASK_INSTRUCTION_ROOT}"
echo "[INFO] latent_root=${LATENT_ROOT}"
echo "[INFO] mix4_meta_root=${MIX4_META_ROOT}"
echo "[INFO] clean_meta_root=${CLEAN_META_ROOT}"
echo "[INFO] task_count=${#TASKS[@]}"
echo "[INFO] num_shards=${NUM_SHARDS}, shard_start=${SHARD_START}, shard_count=${SHARD_COUNT}, local_gpu_count=${LOCAL_GPU_COUNT}, batch_size=${BATCH_SIZE}"
echo "[INFO] action_dim=${ACTION_DIM}, chunk_size=${CHUNK_SIZE}, num_history=${NUM_HISTORY}, num_frames=${NUM_FRAMES}"
printf '[INFO] tasks=%s\n' "${TASKS[*]}"
nvidia-smi || true

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
      --clean_lerobot_video_root "${CLEAN_LEROBOT_VIDEO_ROOT}" \
      --task_instruction_root "${TASK_INSTRUCTION_ROOT}" \
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

if [[ "${WAIT_FOR_ALL_SHARDS_BEFORE_MERGE}" == "1" ]]; then
  echo "[INFO] waiting for all shard manifests before merge"
  wait_start="$(date +%s)"
  while true; do
    missing=0
    for shard in $(seq 0 "$((NUM_SHARDS - 1))"); do
      suffix="$(printf 'shard%02d' "${shard}")"
      for split in clean_train enhanced_train test_quick; do
        [[ -f "${MANIFEST_DIR}/${split}_${suffix}.jsonl" ]] || missing=1
      done
    done
    if [[ "${missing}" -eq 0 ]]; then
      break
    fi
    elapsed_wait="$(( $(date +%s) - wait_start ))"
    if (( elapsed_wait > ALL_SHARDS_WAIT_TIMEOUT_SEC )); then
      echo "[ERROR] timed out waiting for all shard manifests after ${elapsed_wait}s" >&2
      exit 2
    fi
    sleep 60
  done
fi

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
  "enhanced_split_root": "${ENHANCED_SPLIT_ROOT}",
  "clean_lerobot_root": "${CLEAN_LEROBOT_ROOT}",
  "clean_lerobot_video_root": "${CLEAN_LEROBOT_VIDEO_ROOT}",
  "task_instruction_root": "${TASK_INSTRUCTION_ROOT}",
  "task_count": ${#TASKS[@]},
  "num_shards": ${NUM_SHARDS},
  "batch_size": ${BATCH_SIZE},
  "action_dim": ${ACTION_DIM},
  "chunk_size": ${CHUNK_SIZE},
  "num_history": ${NUM_HISTORY},
  "num_frames": ${NUM_FRAMES},
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
enhanced_split_root=${ENHANCED_SPLIT_ROOT}
clean_lerobot_root=${CLEAN_LEROBOT_ROOT}
clean_lerobot_video_root=${CLEAN_LEROBOT_VIDEO_ROOT}
task_instruction_root=${TASK_INSTRUCTION_ROOT}
task_count=${#TASKS[@]}
num_shards=${NUM_SHARDS}
batch_size=${BATCH_SIZE}
action_dim=${ACTION_DIM}
chunk_size=${CHUNK_SIZE}
num_history=${NUM_HISTORY}
num_frames=${NUM_FRAMES}
train_records=${train_count}
clean_train_records=${clean_count}
enhanced_train_records=${enhanced_count}
test_quick_records=${quick_count}
elapsed_seconds=${elapsed}
EOF

cat "${LOG_ROOT}/summary.txt"
echo "[INFO] done"
