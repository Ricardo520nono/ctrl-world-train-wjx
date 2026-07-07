#!/usr/bin/env bash

# Shared paths for Ctrl-World training.
# By default, code runs from the current repo, assets/latents stay on the public disk,
# and model checkpoints/videos are written under the AF3 data fileset.
# Override these environment variables when running from a packaged copy.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_PACKAGE_ROOT="${TRAIN_PACKAGE_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train}"
AF3_CTRLWORLD_ROOT="${AF3_CTRLWORLD_ROOT:-/mnt/dataset/csx_workspace/Ideas/data_AF3/ctrl-world}"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_CODE_ROOT}}"
ASSET_ROOT="${ASSET_ROOT:-${TRAIN_PACKAGE_ROOT}/assets/models}"
CACHE_ROOT="${CACHE_ROOT:-${TRAIN_PACKAGE_ROOT}/latents}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${AF3_CTRLWORLD_ROOT}/outputs}"
CTRLWORLD_ENV_FILE="${CTRLWORLD_ENV_FILE:-${TRAIN_PACKAGE_ROOT}/.env}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

if [[ -f "${CTRLWORLD_ENV_FILE}" ]]; then
  set +u
  set -a
  source "${CTRLWORLD_ENV_FILE}"
  set +a
  set -u
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
