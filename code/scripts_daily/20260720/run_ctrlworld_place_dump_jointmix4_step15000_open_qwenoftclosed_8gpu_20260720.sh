#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_DIR="/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/outputs/ACWM_ctrlworld_mix4_place_dump_c32_cur1f32_rot6d20_cfstatemajor_rgbfix1_8gpu_80k_20260719_debug"
RUN_TAG="${RUN_TAG:-step15000_sample0_steps20_statbridgefix_aihc_20260720}"
EVAL_ROOT="${RUN_DIR}/handoff_eval/${RUN_TAG}"
WM_CKPT="${WM_CKPT:-${RUN_DIR}/checkpoint-step15000-epoch2.26.pt}"
STAT_PATH="${STAT_PATH:-/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/dataset_meta_info/action_following_cur1f32_mix4_place_dump_rot6d20_cfstatemajor_rgbfix1_20260719_debug/stat.json}"
POLICY_CKPT="${POLICY_CKPT:-/mnt/dataset/csx_workspace/Ideas/data_AF3/starVLA_dev/checkpoints/qwenoft_rot6d20_clean50_randomized500_decord_lru8_nofuture_locality_b4_official_48g_bs4/ACWM_T2A_qwenoft_rot6d20_clean50_randomized500_decord_lru8_nofuture_locality_b4_official_48g_bs4_20260709/checkpoints/steps_90000_pytorch_model.pt}"
POLICY_BASE_VLM="${POLICY_BASE_VLM:-/mnt/dataset/public_data/Qwen3-VL-4B-Instruct}"
POLICY_LEGACY_BASE_VLM="${POLICY_LEGACY_BASE_VLM:-/mnt/public_ckp/Qwen3-VL-4B-Instruct}"
CTRLWORLD_PYTHON="${CTRLWORLD_PYTHON:-/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python}"
export WM_CKPT STAT_PATH POLICY_CKPT POLICY_BASE_VLM POLICY_LEGACY_BASE_VLM CTRLWORLD_PYTHON

for path in "${WM_CKPT}" "${STAT_PATH}" "${POLICY_CKPT}" "${POLICY_BASE_VLM}" "${CTRLWORLD_PYTHON}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 2
  fi
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RUN_TAG="${RUN_TAG}" POLICY_GPU=0 OPEN_GPU=2 CLOSED_GPU=4 POLICY_PORT=5694 \
    DRY_RUN=1 bash "${SCRIPT_DIR}/run_ctrlworld_place_burger_fries_jointmix4_step15000_handoff_20260720.sh"
  RUN_TAG="${RUN_TAG}" POLICY_GPU=1 OPEN_GPU=3 CLOSED_GPU=5 POLICY_PORT=5695 \
    DRY_RUN=1 bash "${SCRIPT_DIR}/run_ctrlworld_dump_bin_bigbin_jointmix4_step15000_handoff_20260720.sh"
  echo "[DRY_RUN] audit_root=${EVAL_ROOT}"
  exit 0
fi

if [[ -d "${EVAL_ROOT}" && -n "$(find "${EVAL_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[ERROR] Refusing non-empty evaluation root: ${EVAL_ROOT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${POLICY_LEGACY_BASE_VLM}")"
if [[ -e "${POLICY_LEGACY_BASE_VLM}" || -L "${POLICY_LEGACY_BASE_VLM}" ]]; then
  if [[ "$(realpath "${POLICY_LEGACY_BASE_VLM}")" != "$(realpath "${POLICY_BASE_VLM}")" ]]; then
    echo "[ERROR] Existing legacy policy base path points elsewhere: ${POLICY_LEGACY_BASE_VLM}" >&2
    exit 2
  fi
else
  ln -s "${POLICY_BASE_VLM}" "${POLICY_LEGACY_BASE_VLM}"
fi

place_pid=""
dump_pid=""
cleanup() {
  for pid in "${place_pid}" "${dump_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

RUN_TAG="${RUN_TAG}" POLICY_GPU=0 OPEN_GPU=2 CLOSED_GPU=4 POLICY_PORT=5694 \
  POLICY_BASE_VLM="${POLICY_BASE_VLM}" POLICY_LEGACY_BASE_VLM="${POLICY_LEGACY_BASE_VLM}" \
  bash "${SCRIPT_DIR}/run_ctrlworld_place_burger_fries_jointmix4_step15000_handoff_20260720.sh" &
place_pid=$!

RUN_TAG="${RUN_TAG}" POLICY_GPU=1 OPEN_GPU=3 CLOSED_GPU=5 POLICY_PORT=5695 \
  POLICY_BASE_VLM="${POLICY_BASE_VLM}" POLICY_LEGACY_BASE_VLM="${POLICY_LEGACY_BASE_VLM}" \
  bash "${SCRIPT_DIR}/run_ctrlworld_dump_bin_bigbin_jointmix4_step15000_handoff_20260720.sh" &
dump_pid=$!

set +e
wait "${place_pid}"
place_status=$?
place_pid=""
wait "${dump_pid}"
dump_status=$?
dump_pid=""
set -e

if [[ "${place_status}" -ne 0 || "${dump_status}" -ne 0 ]]; then
  echo "[ERROR] Handoff evaluation failed: place=${place_status} dump=${dump_status}" >&2
  exit 2
fi

"${CTRLWORLD_PYTHON}" "${CODE_ROOT}/scripts/audit_ctrlworld_handoff_eval.py" \
  --root "${EVAL_ROOT}" \
  --tasks place_burger_fries dump_bin_bigbin \
  --ckpt "${WM_CKPT}" \
  --stat "${STAT_PATH}" \
  --policy_ckpt "${POLICY_CKPT}" \
  --wm_steps 20 \
  --policy_ddim_steps 10 \
  --action_dim 20

echo "[INFO] Both tasks completed and final audit passed: ${EVAL_ROOT}"
