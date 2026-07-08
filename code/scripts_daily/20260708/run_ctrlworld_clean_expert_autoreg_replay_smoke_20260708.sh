#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python}"

CKPT_PATH="${CKPT_PATH:-/mnt/dataset/csx_workspace/Ideas/data_AF3/ctrl-world/outputs/ACWM_ctrlworld_action_following_clean_test4v2_chunk32_8gpu_current1_future32_fulldesc_rot6d20_retry1_20260707/checkpoint-step10000-epoch5.39.pt}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-8}"
MAX_ACTIONS="${MAX_ACTIONS:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ctrlworld_clean_expert_autoreg/sample${SAMPLE_INDEX}_actions${MAX_ACTIONS}_steps${NUM_INFERENCE_STEPS}}"

exec "${PYTHON_BIN}" scripts/replay_clean_expert_autoreg.py \
  --ckpt "${CKPT_PATH}" \
  --sample_index "${SAMPLE_INDEX}" \
  --steps "${NUM_INFERENCE_STEPS}" \
  --max_actions "${MAX_ACTIONS}" \
  --out "${OUTPUT_DIR}"
