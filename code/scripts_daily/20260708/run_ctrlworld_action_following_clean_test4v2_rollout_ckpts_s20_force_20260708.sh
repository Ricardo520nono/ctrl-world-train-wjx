#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NUM_INFERENCE_STEPS=20
export FORCE=1
exec "${SCRIPT_DIR}/run_ctrlworld_action_following_clean_test4v2_rollout_ckpts_current1_future32_fulldesc_rot6d20_20260708.sh"
