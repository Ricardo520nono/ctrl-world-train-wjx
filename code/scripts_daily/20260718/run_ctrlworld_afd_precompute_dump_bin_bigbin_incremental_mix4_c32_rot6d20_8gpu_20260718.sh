#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK="dump_bin_bigbin"
export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_afd_precompute_dump_bin_bigbin_c32_rot6d20_cfstatemajor_incremental_8gpu_20260718}"
exec bash "${SCRIPT_DIR}/run_ctrlworld_afd_precompute_single_task_incremental_mix4_c32_rot6d20_8gpu_20260718.sh"

