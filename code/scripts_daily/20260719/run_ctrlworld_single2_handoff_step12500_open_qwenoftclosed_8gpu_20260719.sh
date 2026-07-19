#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG="${RUN_TAG:-step12500_sample0_steps20_aihc_train22_retry1_20260719}"

POLICY_BASE_VLM="${POLICY_BASE_VLM:-/mnt/dataset/public_data/Qwen3-VL-4B-Instruct}"
POLICY_LEGACY_BASE_VLM="${POLICY_LEGACY_BASE_VLM:-/mnt/public_ckp/Qwen3-VL-4B-Instruct}"
if [[ ! -d "${POLICY_BASE_VLM}" ]]; then
  echo "[ERROR] Missing policy base VLM: ${POLICY_BASE_VLM}" >&2
  exit 2
fi
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  mkdir -p "$(dirname "${POLICY_LEGACY_BASE_VLM}")"
  if [[ -e "${POLICY_LEGACY_BASE_VLM}" || -L "${POLICY_LEGACY_BASE_VLM}" ]]; then
    if [[ "$(realpath "${POLICY_LEGACY_BASE_VLM}")" != "$(realpath "${POLICY_BASE_VLM}")" ]]; then
      echo "[ERROR] Existing legacy policy base path points elsewhere: ${POLICY_LEGACY_BASE_VLM}" >&2
      exit 2
    fi
  else
    ln -s "${POLICY_BASE_VLM}" "${POLICY_LEGACY_BASE_VLM}"
  fi
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

RUN_TAG="${RUN_TAG}" \
POLICY_BASE_VLM="${POLICY_BASE_VLM}" POLICY_LEGACY_BASE_VLM="${POLICY_LEGACY_BASE_VLM}" \
POLICY_GPU=0 OPEN_GPU=2 CLOSED_GPU=4 POLICY_PORT=5694 \
bash "${SCRIPT_DIR}/run_ctrlworld_place_burger_fries_handoff_step12500_20260719.sh" &
place_pid=$!

RUN_TAG="${RUN_TAG}" \
POLICY_BASE_VLM="${POLICY_BASE_VLM}" POLICY_LEGACY_BASE_VLM="${POLICY_LEGACY_BASE_VLM}" \
POLICY_GPU=1 OPEN_GPU=3 CLOSED_GPU=5 POLICY_PORT=5695 \
bash "${SCRIPT_DIR}/run_ctrlworld_dump_bin_bigbin_handoff_step12500_20260719.sh" &
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

echo "[INFO] Both Ctrl-World handoff evaluations completed."
