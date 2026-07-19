#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_afd_precompute_50task_chunk32_rot6d20_frameidxfix_multichunk_retry1_part0_20260712}"
export NUM_SHARDS="${NUM_SHARDS:-16}"
export LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-8}"
export SHARD_START="${SHARD_START:-0}"
export SHARD_COUNT="${SHARD_COUNT:-8}"
export RUN_MERGE_AND_STAT="${RUN_MERGE_AND_STAT:-0}"
export CLEAR_TARGET="${CLEAR_TARGET:-0}"
export WAIT_FOR_TARGET_READY="${WAIT_FOR_TARGET_READY:-0}"
exec "${SCRIPT_DIR}/run_ctrlworld_action_following_precompute_50task_clean_mix4_chunk32_sharded_current1_future32_fulldesc_rot6d20_frameidxfix_20260711.sh"
