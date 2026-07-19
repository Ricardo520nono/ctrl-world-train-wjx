#!/usr/bin/env bash
set -euo pipefail

# Deprecated: retained only to reproduce the legacy 14D/multi-history rollout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${CODE_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python}"

RUN_DIR="/mnt/dataset/public_data/cscsx_projects/AF3/starVLA_dev/legacy_flat_results_20260706/ctrl-world/outputs/ACWM_ctrlworld_action_following_mix4_test4v2_chunk32_8gpu_rgbfix_explore_resume15000_20260706"
CKPT_PATH="${RUN_DIR}/checkpoint-step25000-epoch2.06.pt"
LATENT_ROOT="/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/latents/action_following_chunk32_clean_enhanced_explore_test4v2_20260705"
MANIFEST_PATH="${LATENT_ROOT}/manifests/clean_train.jsonl"
STAT_PATH="/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_chunk32_clean_enhanced_explore_test4v2_20260705/stat.json"

OUT_DIR="${RUN_DIR}/rollout_latest_rotate_qrcode/checkpoint-step25000-epoch2.06_rotate_qrcode_sample0_steps20"

exec "${PYTHON_BIN}" scripts/replay_clean_expert_autoreg.py \
  --ckpt "${CKPT_PATH}" \
  --latent_root "${LATENT_ROOT}" \
  --manifest "${MANIFEST_PATH}" \
  --stat "${STAT_PATH}" \
  --task rotate_qrcode \
  --sample_index 0 \
  --steps 20 \
  --max_actions 0 \
  --chunk_size 32 \
  --num_history 6 \
  --num_frames 26 \
  --action_chunk_size 32 \
  --action_dim 14 \
  --decode_chunk 7 \
  --out "${OUT_DIR}"
