#!/usr/bin/env bash

# Shared paths for the public Ctrl-World training package.
# Override these environment variables when running on a different mount.

TRAIN_PACKAGE_ROOT="${TRAIN_PACKAGE_ROOT:-/mnt/public_ckp/cscsx_projects/ctrl_world_train}"
PROJECT_ROOT="${PROJECT_ROOT:-${TRAIN_PACKAGE_ROOT}/code}"
ASSET_ROOT="${ASSET_ROOT:-${TRAIN_PACKAGE_ROOT}/assets/models}"
CACHE_ROOT="${CACHE_ROOT:-${TRAIN_PACKAGE_ROOT}/latents}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_PACKAGE_ROOT}/outputs}"
CTRLWORLD_ENV_FILE="${CTRLWORLD_ENV_FILE:-${TRAIN_PACKAGE_ROOT}/.env}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
