#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERIC_SCRIPT="${SCRIPT_DIR}/../20260719/run_ctrlworld_single_task_open_policy_closed_loop_20260719.sh"
RUN_DIR="/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/outputs/ACWM_ctrlworld_mix4_place_dump_c32_cur1f32_rot6d20_cfstatemajor_rgbfix1_8gpu_80k_20260719_debug"

export TASK="dump_bin_bigbin"
export WM_CKPT="${WM_CKPT:-${RUN_DIR}/checkpoint-step15000-epoch2.26.pt}"
export LATENT_ROOT="${LATENT_ROOT:-/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/latents/action_following_cur1f32_mix4_place_dump_rot6d20_cfstatemajor_rgbfix1_20260719_debug}"
export MANIFEST_PATH="${MANIFEST_PATH:-${LATENT_ROOT}/manifests/clean_train.jsonl}"
export STAT_PATH="${STAT_PATH:-/mnt/dataset/public_data/cscsx_projects/AF3/ctrl-world/dataset_meta_info/action_following_cur1f32_mix4_place_dump_rot6d20_cfstatemajor_rgbfix1_20260719_debug/stat.json}"
export RUN_TAG="${RUN_TAG:-step15000_sample0_steps20_statbridgefix_aihc_20260720}"
export OUT_ROOT="${OUT_ROOT:-${RUN_DIR}/handoff_eval/${RUN_TAG}/${TASK}}"

exec bash "${GENERIC_SCRIPT}"
