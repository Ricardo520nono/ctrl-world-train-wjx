#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_KIND="clean"
exec "${SCRIPT_DIR}/run_ctrlworld_action_following_rollout_ckpts_test4v2_current1_future32_fulldesc_rot6d20_20260708.sh"
