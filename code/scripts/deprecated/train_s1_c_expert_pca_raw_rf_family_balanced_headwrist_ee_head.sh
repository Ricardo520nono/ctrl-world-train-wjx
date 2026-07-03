#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export USE_EE_HEAD=1
export EE_LOSS_WEIGHT="${EE_LOSS_WEIGHT:-0.05}"
export EE_HEAD_HIDDEN_DIM="${EE_HEAD_HIDDEN_DIM:-256}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-s1_C_3to1to1to1_family_balanced_headwrist_eehead_w${EE_LOSS_WEIGHT}}"

exec bash "${SCRIPT_DIR}/train_s1_c_expert_pca_raw_rf_family_balanced_headwrist.sh"
