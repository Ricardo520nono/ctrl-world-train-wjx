#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python}"

LATENT_ROOT="${LATENT_ROOT:-/mnt/dataset/csx_workspace/Ideas/data_AF3/starVLA_dev/legacy_flat_results_20260706/ctrl_world_train/latents/action_following_current1_future32_clean_enhanced_explore_test4v2_fulldesc_rot6d20_retry1_20260707}"
OUT_ROOT="${OUT_ROOT:-$(dirname "${LATENT_ROOT}")/$(basename "${LATENT_ROOT}")_decoded_videos}"
SVD_PATH="${SVD_PATH:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/assets/models/stable-video-diffusion-img2vid}"

LIMIT="${LIMIT:-0}"
OFFSET="${OFFSET:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
FPS="${FPS:-2}"
DECODE_CHUNK="${DECODE_CHUNK:-32}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
INPUT_SUBDIR="${INPUT_SUBDIR:-}"

ARGS=(
  --latent_root "${LATENT_ROOT}"
  --out_root "${OUT_ROOT}"
  --svd_path "${SVD_PATH}"
  --offset "${OFFSET}"
  --limit "${LIMIT}"
  --shard_index "${SHARD_INDEX}"
  --shard_count "${SHARD_COUNT}"
  --fps "${FPS}"
  --decode_chunk "${DECODE_CHUNK}"
)

if [[ -n "${INPUT_SUBDIR}" ]]; then
  ARGS+=(--input_subdir "${INPUT_SUBDIR}")
fi
if [[ "${FORCE}" == "1" ]]; then
  ARGS+=(--force)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=(--dry_run)
fi

exec "${PYTHON_BIN}" scripts/decode_action_following_latents_to_videos.py "${ARGS[@]}"
