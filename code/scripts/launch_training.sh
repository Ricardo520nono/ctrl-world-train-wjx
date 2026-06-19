#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ctrlworld_train_env.sh"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/launch_training.sh <recipe> [task]

Recipes:
  all50_headwrist       50 tasks, expert, chunk16, 60k steps
  s1_a_expert           5 tasks, expert-only, chunk16, 40k steps
  s1_b_expert_pca       5 tasks, expert + PCA, chunk16, 40k steps
  s1_c_3to1to1to1       5 tasks, expert + PCA + raw + random-feasible, 3:1:1:1, chunk16, 40k steps
  s1_c_ee_head          S1-C with EE trajectory auxiliary head enabled
  s1_a_single_task      one S1 task expert-only, chunk16, 40k steps; requires [task]

Single-task names:
  click_alarmclock
  click_bell
  place_object_basket
  open_laptop
  stack_blocks_two

Examples:
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
  all50_headwrist)
    exec bash "${SCRIPT_DIR}/train_ctrlworld_8gpu_delta_ee_all50_nf16_60k_headwrist.sh"
    ;;
  s1_a_expert)
    exec bash "${SCRIPT_DIR}/train_s1_a_expert_only_headwrist.sh"
    ;;
  s1_b_expert_pca)
    exec bash "${SCRIPT_DIR}/train_s1_b_expert_sliding_pca_single_headwrist.sh"
    ;;
  s1_c_3to1to1to1)
    exec bash "${SCRIPT_DIR}/train_s1_c_expert_pca_raw_rf_family_balanced_headwrist.sh"
    ;;
  s1_c_ee_head)
    exec bash "${SCRIPT_DIR}/train_s1_c_expert_pca_raw_rf_family_balanced_headwrist_ee_head.sh"
    ;;
  s1_a_single_task)
    if [[ -z "${task}" ]]; then
      echo "[ERROR] s1_a_single_task requires a task name."
      usage
      exit 2
    fi
    exec bash "${SCRIPT_DIR}/train_s1_a_expert_only_headwrist_single_task.sh" "${task}"
    ;;
  *)
    echo "[ERROR] Unknown recipe: ${recipe}"
    usage
    exit 2
    ;;
esac
