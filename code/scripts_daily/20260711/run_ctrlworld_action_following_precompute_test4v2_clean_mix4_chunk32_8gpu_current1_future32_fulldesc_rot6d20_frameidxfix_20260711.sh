#!/usr/bin/env bash
set -euo pipefail

# 4-task Rot6D20 ActionFollowingData precompute after clean LeRobot frame-index fix.
# Single 8GPU job: 8 shards for place_burger_fries/lift_pot/dump_bin_bigbin/rotate_qrcode.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_afd_precompute_test4v2_chunk32_rot6d20_frameidxfix_20260711}"
export TASKS="${TASKS:-place_burger_fries lift_pot dump_bin_bigbin rotate_qrcode}"
export EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT:-4}"
export NUM_SHARDS="${NUM_SHARDS:-8}"
export LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-8}"
export SHARD_START="${SHARD_START:-0}"
export SHARD_COUNT="${SHARD_COUNT:-8}"
export RUN_MERGE_AND_STAT="${RUN_MERGE_AND_STAT:-1}"
export CLEAR_TARGET="${CLEAR_TARGET:-1}"
export WAIT_FOR_TARGET_READY="${WAIT_FOR_TARGET_READY:-0}"
export LATENT_ROOT="${LATENT_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/latents/action_following_current1_future32_clean_enhanced_explore_test4v2_fulldesc_rot6d20_frameidxfix_20260711}"
export MIX4_META_ROOT="${MIX4_META_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_current1_future32_clean_enhanced_explore_test4v2_fulldesc_rot6d20_frameidxfix_20260711}"
export CLEAN_META_ROOT="${CLEAN_META_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_current1_future32_clean_only_test4v2_fulldesc_rot6d20_frameidxfix_20260711}"

exec "${SCRIPT_DIR}/run_ctrlworld_action_following_precompute_50task_clean_mix4_chunk32_sharded_current1_future32_fulldesc_rot6d20_frameidxfix_20260711.sh"
