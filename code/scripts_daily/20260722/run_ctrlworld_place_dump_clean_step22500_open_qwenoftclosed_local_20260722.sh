#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}"

RUN_DIR="/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/outputs/ACWM_ctrlworld_clean_place_dump_c32_cur1f32_rot6d20_cfstatemajor_rgbfix1_8gpu_80k_20260721_debug"
RUN_TAG="${RUN_TAG:-step22500_sample0_steps20_statbridgefix_local_20260722}"
WM_CKPT="${WM_CKPT:-${RUN_DIR}/checkpoint-step22500-epoch18.35.pt}"
LATENT_ROOT="${LATENT_ROOT:-/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/latents/action_following_cur1f32_mix4_place_dump_rot6d20_cfstatemajor_rgbfix1_20260719_debug}"
STAT_PATH="${STAT_PATH:-/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/dataset_meta_info/action_following_cur1f32_clean_place_dump_rot6d20_cfstatemajor_rgbfix1_20260721_debug/stat.json}"
MANIFEST_PATH="${MANIFEST_PATH:-${LATENT_ROOT}/manifests/clean_train.jsonl}"
POLICY_CKPT="${POLICY_CKPT:-/mnt/dataset/csx_workspace/Ideas/data_AF3/starVLA_dev/checkpoints/qwenoft_rot6d20_clean50_randomized500_decord_lru8_nofuture_locality_b4_official_48g_bs4/ACWM_T2A_qwenoft_rot6d20_clean50_randomized500_decord_lru8_nofuture_locality_b4_official_48g_bs4_20260709/checkpoints/steps_90000_pytorch_model.pt}"
EVAL_ROOT="${RUN_DIR}/handoff_eval/${RUN_TAG}"
CTRLWORLD_PYTHON="${CTRLWORLD_PYTHON:-/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python}"
GPU="${GPU:-0}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for task in place_burger_fries dump_bin_bigbin; do
    TASK="${task}" \
    WM_CKPT="${WM_CKPT}" \
    LATENT_ROOT="${LATENT_ROOT}" \
    MANIFEST_PATH="${MANIFEST_PATH}" \
    STAT_PATH="${STAT_PATH}" \
    POLICY_CKPT="${POLICY_CKPT}" \
    OUT_ROOT="${EVAL_ROOT}/${task}" \
    GPU="${GPU}" \
    POLICY_PORT="$([[ "${task}" == "place_burger_fries" ]] && echo 5694 || echo 5695)" \
    DRY_RUN=1 bash "${SCRIPT_DIR}/run_ctrlworld_single_task_open_policy_closed_loop_sequential_20260722.sh"
  done
  exit 0
fi

if [[ -d "${EVAL_ROOT}" && -n "$(find "${EVAL_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[ERROR] Refusing non-empty eval root: ${EVAL_ROOT}" >&2
  exit 2
fi
mkdir -p "${EVAL_ROOT}"

for task in place_burger_fries dump_bin_bigbin; do
  TASK="${task}" \
  WM_CKPT="${WM_CKPT}" \
  LATENT_ROOT="${LATENT_ROOT}" \
  MANIFEST_PATH="${MANIFEST_PATH}" \
  STAT_PATH="${STAT_PATH}" \
  POLICY_CKPT="${POLICY_CKPT}" \
  OUT_ROOT="${EVAL_ROOT}/${task}" \
  GPU="${GPU}" \
  POLICY_PORT="$([[ "${task}" == "place_burger_fries" ]] && echo 5694 || echo 5695)" \
  bash "${SCRIPT_DIR}/run_ctrlworld_single_task_open_policy_closed_loop_sequential_20260722.sh"
done

"${CTRLWORLD_PYTHON}" scripts/audit_ctrlworld_handoff_eval.py \
  --root "${EVAL_ROOT}" \
  --tasks place_burger_fries dump_bin_bigbin \
  --ckpt "${WM_CKPT}" \
  --stat "${STAT_PATH}" \
  --policy_ckpt "${POLICY_CKPT}" \
  --wm_steps 20 \
  --policy_ddim_steps 10 \
  --action_dim 20

echo "[INFO] Clean step22500 handoff evaluation complete: ${EVAL_ROOT}"
