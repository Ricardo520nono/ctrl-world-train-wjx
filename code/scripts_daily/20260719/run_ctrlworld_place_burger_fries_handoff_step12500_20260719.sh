#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/outputs/ACWM_ctrlworld_mix4_place_burger_fries_c32_cur1f32_rot6d20_cfstatemajor_8gpu_20260718"
LATENT_ROOT="/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/latents/action_following_cur1f32_mix4_place_burger_fries_rot6d20_cfstatemajor_incremental_20260718"

export TASK="place_burger_fries"
export WM_CKPT="${WM_CKPT:-${RUN_DIR}/checkpoint-step12500-epoch3.70.pt}"
export LATENT_ROOT="${LATENT_ROOT}"
export STAT_PATH="${STAT_PATH:-/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/dataset_meta_info/action_following_cur1f32_mix4_place_burger_fries_rot6d20_cfstatemajor_incremental_20260718/stat.json}"
export RUN_TAG="${RUN_TAG:-step12500_sample0_steps20_20260719}"
export OUT_ROOT="${OUT_ROOT:-${RUN_DIR}/handoff_eval/${RUN_TAG}}"

exec bash "${SCRIPT_DIR}/run_ctrlworld_single_task_open_policy_closed_loop_20260719.sh"
