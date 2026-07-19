#!/usr/bin/env bash
set -euo pipefail

# Deprecated: superseded by the per-task incremental cache workflow.
# Shared precompute for two single-task mix4 runs using the active state-major
# counterfactual replay data. Task-local manifest views and stats are generated
# after the shared VAE cache is complete.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${CODE_ROOT}"

source "${CODE_ROOT}/scripts/ctrlworld_train_env.sh"

export PYTHON_BIN="${PYTHON_BIN:-/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python}"
if [[ "${PYTHON_BIN}" == "/usr/bin/python3" && -x "/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python" ]]; then
  export PYTHON_BIN="/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_afd_precompute_single2_mix4_c32_rot6d20_cfstatemajor_8gpu_20260718}"
export TASKS="${TASKS:-place_burger_fries dump_bin_bigbin}"
export EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT:-2}"
export NUM_SHARDS="${NUM_SHARDS:-8}"
export LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-8}"
export SHARD_START="${SHARD_START:-0}"
export SHARD_COUNT="${SHARD_COUNT:-8}"
export RUN_MERGE_AND_STAT="${RUN_MERGE_AND_STAT:-1}"
export ACTION_DIM="${ACTION_DIM:-20}"
export CHUNK_SIZE="${CHUNK_SIZE:-32}"
export NUM_HISTORY="${NUM_HISTORY:-1}"
export NUM_FRAMES="${NUM_FRAMES:-32}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export CLEAR_TARGET="${CLEAR_TARGET:-0}"
export ENHANCED_SPLIT_ROOT="${ENHANCED_SPLIT_ROOT:-/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData_Rot6D/enhanced_v1_split}"
export CLEAN_LEROBOT_ROOT="${CLEAN_LEROBOT_ROOT:-/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData_LeRobot_Rot6D/train/demo_clean_zed2i_visible}"
export CLEAN_LEROBOT_VIDEO_ROOT="${CLEAN_LEROBOT_VIDEO_ROOT:-/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingBench/data_lerobot/robotwin_delta_ee/demo_clean_zed2i_visible}"
export TASK_INSTRUCTION_ROOT="${TASK_INSTRUCTION_ROOT:-/mnt/dataset/csx_workspace/Ideas/AF3/code/RoboTwin/description/task_instruction}"

PUBLIC_ROOT="${PUBLIC_ROOT:-/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world}"
export LATENT_ROOT="${LATENT_ROOT:-${PUBLIC_ROOT}/latents/action_following_cur1f32_mix4_single2_rot6d20_cfstatemajor_20260718}"
export MIX4_META_ROOT="${MIX4_META_ROOT:-${PUBLIC_ROOT}/dataset_meta_info/action_following_cur1f32_mix4_single2_rot6d20_cfstatemajor_20260718}"
export CLEAN_META_ROOT="${CLEAN_META_ROOT:-${PUBLIC_ROOT}/dataset_meta_info/action_following_cur1f32_clean_single2_rot6d20_cfstatemajor_20260718}"
export LOG_ROOT="${LOG_ROOT:-${PUBLIC_ROOT}/logs/${RUN_NAME}}"
export TASK_VIEW_ROOT="${TASK_VIEW_ROOT:-${LATENT_ROOT}/task_views}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] run=${RUN_NAME}"
  echo "[DRY_RUN] tasks=${TASKS}"
  echo "[DRY_RUN] latent_root=${LATENT_ROOT}"
  echo "[DRY_RUN] task_view_root=${TASK_VIEW_ROOT}"
  echo "[DRY_RUN] public_root=${PUBLIC_ROOT}"
fi

bash "${CODE_ROOT}/scripts_daily/20260711/run_ctrlworld_action_following_precompute_50task_clean_mix4_chunk32_sharded_current1_future32_fulldesc_rot6d20_frameidxfix_20260711.sh"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

read -r -a TASK_ARRAY <<<"${TASKS}"
"${PYTHON_BIN}" "${CODE_ROOT}/scripts/deprecated/action_following/prepare_action_following_single_task_views.py" \
  --source_root "${LATENT_ROOT}" \
  --view_root "${TASK_VIEW_ROOT}" \
  --tasks "${TASK_ARRAY[@]}"

for task in "${TASK_ARRAY[@]}"; do
  task_root="${TASK_VIEW_ROOT}/${task}"
  task_meta_root="${PUBLIC_ROOT}/dataset_meta_info/action_following_cur1f32_mix4_${task}_rot6d20_cfstatemajor_20260718"
  mkdir -p "${task_meta_root}"
  "${PYTHON_BIN}" "${CODE_ROOT}/scripts/compute_stat_action_following.py" \
    --latent_root "${task_root}" \
    --manifest "${task_root}/manifests/train.jsonl" \
    --out_dir "${task_meta_root}" \
    --action_dim 20 | tee "${LOG_ROOT}/compute_stat_${task}.log"
done

echo "[INFO] single-task views and stats ready"
