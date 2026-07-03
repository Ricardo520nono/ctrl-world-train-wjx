#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python}"
PRESET="${PRESET:-clean}"

if [[ "${PRESET}" == "clean" ]]; then
  DEFAULT_CKPT="/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/outputs/ACWM_ctrlworld_action_following_clean_test4v2_chunk32_8gpu_ckpt2500_20260702/checkpoint-step2500-epoch0.21.pt"
else
  DEFAULT_CKPT="/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/outputs/ACWM_ctrlworld_action_following_mix3_test4v2_chunk32_8gpu_ckpt2500_20260702/checkpoint-step2500-epoch0.21.pt"
fi

CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ctrlworld_action_following_infer/${PRESET}_sample${SAMPLE_INDEX}_steps${NUM_INFERENCE_STEPS}}"

exec "${PYTHON_BIN}" scripts/infer_action_following_ckpt.py \
  --preset "${PRESET}" \
  --ckpt_path "${CKPT_PATH}" \
  --mode train \
  --sample_index "${SAMPLE_INDEX}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --output_dir "${OUTPUT_DIR}"
