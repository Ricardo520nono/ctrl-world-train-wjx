#!/usr/bin/env bash
set -euo pipefail

# Part 0/2 for Ctrl-World 50-task ActionFollowingData precompute.
# Runs shards 00-07 on one 8GPU AIHC job. Merge/stat is done separately
# after both part0 and part1 finish.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-ACWM_ctrlworld_afd_precompute_50task_chunk32_8gpu_part0_rgbfix_explore_20260706}"
export NUM_SHARDS="${NUM_SHARDS:-16}"
export LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-8}"
export SHARD_START="${SHARD_START:-0}"
export SHARD_COUNT="${SHARD_COUNT:-8}"
export RUN_MERGE_AND_STAT="${RUN_MERGE_AND_STAT:-0}"
export REUSE_SAFE_CACHE="${REUSE_SAFE_CACHE:-1}"
export CLEAR_TARGET="${CLEAR_TARGET:-0}"

exec "${SCRIPT_DIR}/run_ctrlworld_action_following_precompute_50task_chunk32_sharded_rgbfix_explore_20260706.sh"
