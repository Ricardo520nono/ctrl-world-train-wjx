#!/usr/bin/env bash

# Shared paths for Ctrl-World training.
# By default, code runs from the current repo while cache/assets/outputs live on the public disk.
# Override these environment variables when running from a packaged copy.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_PACKAGE_ROOT="${TRAIN_PACKAGE_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train}"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_CODE_ROOT}}"
ASSET_ROOT="${ASSET_ROOT:-${TRAIN_PACKAGE_ROOT}/assets/models}"
CACHE_ROOT="${CACHE_ROOT:-${TRAIN_PACKAGE_ROOT}/latents}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_PACKAGE_ROOT}/outputs}"
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
