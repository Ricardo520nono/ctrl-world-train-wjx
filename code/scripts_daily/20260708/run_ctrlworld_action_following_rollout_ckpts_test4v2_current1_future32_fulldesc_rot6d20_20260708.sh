#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python}"
RUN_KIND="${RUN_KIND:-clean}"
TASKS=(${TASKS:-place_burger_fries lift_pot dump_bin_bigbin rotate_qrcode})
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-8}"
MAX_ACTIONS="${MAX_ACTIONS:-0}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
FORCE="${FORCE:-0}"

case "${RUN_KIND}" in
  clean)
    RUN_DIR="${RUN_DIR:-/mnt/dataset/csx_workspace/Ideas/data_AF3/ctrl-world/outputs/ACWM_ctrlworld_action_following_clean_test4v2_chunk32_8gpu_current1_future32_fulldesc_rot6d20_retry1_20260707}"
    LATENT_ROOT="${LATENT_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/latents/action_following_current1_future32_clean_only_test4v2_fulldesc_rot6d20_retry1_20260707}"
    STAT_PATH="${STAT_PATH:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_current1_future32_clean_only_test4v2_fulldesc_rot6d20_retry1_20260707/stat.json}"
    ;;
  mix4)
    RUN_DIR="${RUN_DIR:-/mnt/dataset/csx_workspace/Ideas/data_AF3/ctrl-world/outputs/ACWM_ctrlworld_action_following_mix4_test4v2_chunk32_8gpu_current1_future32_fulldesc_rot6d20_retry1_20260707}"
    LATENT_ROOT="${LATENT_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/latents/action_following_current1_future32_clean_enhanced_explore_test4v2_fulldesc_rot6d20_retry1_20260707}"
    STAT_PATH="${STAT_PATH:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_current1_future32_clean_enhanced_explore_test4v2_fulldesc_rot6d20_retry1_20260707/stat.json}"
    ;;
  *)
    echo "[ERROR] RUN_KIND must be clean or mix4, got: ${RUN_KIND}" >&2
    exit 2
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[ERROR] RUN_DIR does not exist: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -d "${LATENT_ROOT}" ]]; then
  echo "[ERROR] LATENT_ROOT does not exist: ${LATENT_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${STAT_PATH}" ]]; then
  echo "[ERROR] STAT_PATH does not exist: ${STAT_PATH}" >&2
  exit 2
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
fi
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[ERROR] no visible GPUs" >&2
  exit 2
fi
MAX_PARALLEL="${MAX_PARALLEL:-${#GPUS[@]}}"

ROLLOUT_ROOT="${RUN_DIR}/rollout"
LOG_ROOT="${ROLLOUT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

mapfile -t CKPTS < <(find "${RUN_DIR}" -maxdepth 1 -name 'checkpoint-step*.pt' | sort -V)
if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "[ERROR] no checkpoint-step*.pt files under ${RUN_DIR}" >&2
  exit 2
fi

echo "[rollout] run_kind=${RUN_KIND}"
echo "[rollout] run_dir=${RUN_DIR}"
echo "[rollout] latent_root=${LATENT_ROOT}"
echo "[rollout] stat_path=${STAT_PATH}"
echo "[rollout] ckpts=${#CKPTS[@]} tasks=${#TASKS[@]} gpus=${GPUS[*]} max_parallel=${MAX_PARALLEL}"
echo "[rollout] steps=${NUM_INFERENCE_STEPS} max_actions=${MAX_ACTIONS} sample_index=${SAMPLE_INDEX}"

declare -a PIDS=()
FAIL=0
JOB_INDEX=0

wait_for_first() {
  local pid="${PIDS[0]}"
  if ! wait "${pid}"; then
    FAIL=1
  fi
  PIDS=("${PIDS[@]:1}")
}

for ckpt in "${CKPTS[@]}"; do
  ckpt_tag="$(basename "${ckpt}" .pt)"
  for task in "${TASKS[@]}"; do
    out_dir="${ROLLOUT_ROOT}/${ckpt_tag}/${task}"
    log_file="${LOG_ROOT}/${ckpt_tag}__${task}.log"
    if [[ "${FORCE}" != "1" && -f "${out_dir}/metadata.json" && -f "${out_dir}/autoregressive_pred.mp4" && -f "${out_dir}/gt_left_autoreg_right.mp4" ]]; then
      echo "[rollout] skip existing ${ckpt_tag}/${task}"
      continue
    fi
    gpu="${GPUS[$((JOB_INDEX % ${#GPUS[@]}))]}"
    JOB_INDEX=$((JOB_INDEX + 1))
    mkdir -p "${out_dir}"
    echo "[rollout] launch gpu=${gpu} ckpt=${ckpt_tag} task=${task} out=${out_dir}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      "${PYTHON_BIN}" scripts/replay_clean_expert_autoreg.py \
        --ckpt "${ckpt}" \
        --latent_root "${LATENT_ROOT}" \
        --stat "${STAT_PATH}" \
        --task "${task}" \
        --sample_index "${SAMPLE_INDEX}" \
        --steps "${NUM_INFERENCE_STEPS}" \
        --max_actions "${MAX_ACTIONS}" \
        --out "${out_dir}"
    ) >"${log_file}" 2>&1 &
    PIDS+=("$!")
    while [[ "${#PIDS[@]}" -ge "${MAX_PARALLEL}" ]]; do
      wait_for_first
    done
  done
done

while [[ "${#PIDS[@]}" -gt 0 ]]; do
  wait_for_first
done

find "${ROLLOUT_ROOT}" -mindepth 3 -maxdepth 3 -name metadata.json | sort > "${ROLLOUT_ROOT}/metadata_files.txt"
find "${ROLLOUT_ROOT}" -mindepth 3 -maxdepth 3 -name gt_left_autoreg_right.mp4 | sort > "${ROLLOUT_ROOT}/compare_videos.txt"
find "${ROLLOUT_ROOT}" -mindepth 3 -maxdepth 3 -name autoregressive_pred.mp4 | sort > "${ROLLOUT_ROOT}/pred_videos.txt"

if [[ "${FAIL}" != "0" ]]; then
  echo "[ERROR] one or more rollout jobs failed; see ${LOG_ROOT}" >&2
  exit 1
fi

echo "[rollout] done"
echo "[rollout] outputs=${ROLLOUT_ROOT}"
wc -l "${ROLLOUT_ROOT}/metadata_files.txt" "${ROLLOUT_ROOT}/compare_videos.txt" "${ROLLOUT_ROOT}/pred_videos.txt"
