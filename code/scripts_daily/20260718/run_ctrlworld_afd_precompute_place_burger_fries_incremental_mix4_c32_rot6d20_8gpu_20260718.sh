#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK="place_burger_fries"
export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_afd_precompute_place_burger_fries_c32_rot6d20_cfstatemajor_incremental_8gpu_20260718}"
exec bash "${SCRIPT_DIR}/run_ctrlworld_afd_precompute_single_task_incremental_mix4_c32_rot6d20_8gpu_20260718.sh"

