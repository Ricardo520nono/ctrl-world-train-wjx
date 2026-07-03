#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ctrlworld_train_env.sh"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/launch_training.sh <recipe> [task]

Recipes:
  action_following_mix3  formal ActionFollowingData, clean:enhanced = 1:1, chunk32
  action_following_mix1  formal ActionFollowingData, clean:3 enhanced families = 1:1:1:1, chunk32
  action_following_clean formal ActionFollowingData, clean-only baseline, chunk32
  all50_headwrist       deprecated historical recipe: 50 tasks, expert, chunk16, 60k steps
  s1_a_expert           deprecated historical recipe: 5 tasks, expert-only, chunk16, 40k steps
  s1_b_expert_pca       deprecated historical recipe: 5 tasks, expert + PCA, chunk16, 40k steps
  s1_c_3to1to1to1       deprecated historical recipe: 5 tasks, expert + PCA + raw + random-feasible, 3:1:1:1, chunk16
  s1_c_ee_head          deprecated historical recipe: S1-C with EE trajectory auxiliary head enabled
  s1_a_single_task      deprecated historical recipe: one S1 task expert-only, chunk16; requires [task]

Single-task names:
  click_alarmclock
  click_bell
  place_object_basket
  open_laptop
  stack_blocks_two

Examples:
  bash scripts/launch_training.sh action_following_mix3
  bash scripts/launch_training.sh action_following_mix1
  bash scripts/launch_training.sh s1_c_3to1to1to1
  bash scripts/launch_training.sh s1_c_ee_head
  bash scripts/launch_training.sh s1_a_single_task place_object_basket
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

recipe="$1"
task="${2:-}"

case "${recipe}" in
  action_following_mix3)
    exec bash "${SCRIPT_DIR}/train_action_following_mix.sh" mix_3to1to1to1
    ;;
  action_following_mix1)
    exec bash "${SCRIPT_DIR}/train_action_following_mix.sh" mix_1to1to1to1
    ;;
  action_following_clean)
    INCLUDE_ENHANCED=0 exec bash "${SCRIPT_DIR}/train_action_following_mix.sh" clean_only
    ;;
  all50_headwrist)
    exec bash "${SCRIPT_DIR}/deprecated/train_ctrlworld_8gpu_delta_ee_all50_nf16_60k_headwrist.sh"
    ;;
  s1_a_expert)
    exec bash "${SCRIPT_DIR}/deprecated/train_s1_a_expert_only_headwrist.sh"
    ;;
  s1_b_expert_pca)
    exec bash "${SCRIPT_DIR}/deprecated/train_s1_b_expert_sliding_pca_single_headwrist.sh"
    ;;
  s1_c_3to1to1to1)
    exec bash "${SCRIPT_DIR}/deprecated/train_s1_c_expert_pca_raw_rf_family_balanced_headwrist.sh"
    ;;
  s1_c_ee_head)
    exec bash "${SCRIPT_DIR}/deprecated/train_s1_c_expert_pca_raw_rf_family_balanced_headwrist_ee_head.sh"
    ;;
  s1_a_single_task)
    if [[ -z "${task}" ]]; then
      echo "[ERROR] s1_a_single_task requires a task name."
      usage
      exit 2
    fi
    exec bash "${SCRIPT_DIR}/deprecated/train_s1_a_expert_only_headwrist_single_task.sh" "${task}"
    ;;
  *)
    echo "[ERROR] Unknown recipe: ${recipe}"
    usage
    exit 2
    ;;
esac
