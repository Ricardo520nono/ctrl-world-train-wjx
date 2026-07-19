#!/usr/bin/env bash
set -euo pipefail

# Deprecated: retained only to reproduce the legacy StarVLA/Ctrl-World rollout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${CODE_ROOT}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

STARVLA_ROOT="${STARVLA_ROOT:-/mnt/dataset/csx_workspace/Ideas/AF3/code/starVLA_dev}"
STARVLA_PYTHON="${STARVLA_PYTHON:-${STARVLA_ROOT}/.venv/bin/python}"
CTRLWORLD_PYTHON="${CTRLWORLD_PYTHON:-/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python}"

POLICY_CKPT="${POLICY_CKPT:-/mnt/dataset/public_data/cscsx_projects/AF3/starVLA_dev/checkpoints/starvla_oft_deltaee32_50clean_500randomized_official_48g_bs4/ACWM_T2A_starvla_oft_deltaee32_50clean_500randomized_official_48g_bs4_retry1_20260627/final_model/pytorch_model.pt}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5694}"
POLICY_GPU="${POLICY_GPU:-0}"
WM_GPU="${WM_GPU:-0}"
START_POLICY_SERVER="${START_POLICY_SERVER:-1}"
POLICY_NUM_DDIM_STEPS="${POLICY_NUM_DDIM_STEPS:-10}"
ROBOTWIN_USE_BF16="${ROBOTWIN_USE_BF16:-1}"

RUN_DIR="/mnt/dataset/public_data/cscsx_projects/AF3/starVLA_dev/legacy_flat_results_20260706/ctrl-world/outputs/ACWM_ctrlworld_action_following_mix4_test4v2_chunk32_8gpu_rgbfix_explore_resume15000_20260706"
WM_CKPT="${WM_CKPT:-${RUN_DIR}/checkpoint-step25000-epoch2.06.pt}"
LATENT_ROOT="${LATENT_ROOT:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/latents/action_following_chunk32_clean_enhanced_explore_test4v2_20260705}"
MANIFEST_PATH="${MANIFEST_PATH:-${LATENT_ROOT}/manifests/clean_train.jsonl}"
STAT_PATH="${STAT_PATH:-/mnt/dataset/public_data/cscsx_projects/ctrl_world_train/dataset_meta_info/action_following_chunk32_clean_enhanced_explore_test4v2_20260705/stat.json}"

TASK="${TASK:-rotate_qrcode}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/rollout_policy_server_rotate_qrcode/checkpoint-step25000-epoch2.06_starvla_retry1_sample0_steps20}"
SERVER_LOG="${SERVER_LOG:-${OUT_DIR}/policy_server.log}"
CLIENT_LOG="${CLIENT_LOG:-${OUT_DIR}/policy_rollout.log}"

mkdir -p "${OUT_DIR}"

for path in "${STARVLA_ROOT}" "${STARVLA_PYTHON}" "${CTRLWORLD_PYTHON}" "${WM_CKPT}" "${LATENT_ROOT}" "${MANIFEST_PATH}" "${STAT_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 1
  fi
done

if [[ "${START_POLICY_SERVER}" == "1" && ! -f "${POLICY_CKPT}" ]]; then
  echo "[ERROR] POLICY_CKPT does not exist: ${POLICY_CKPT}" >&2
  echo "[ERROR] Set POLICY_CKPT=/actual/path/to/pytorch_model.pt, or mount the requested checkpoint path." >&2
  exit 1
fi

port_open() {
  "${CTRLWORLD_PYTHON}" - "${POLICY_HOST}" "${POLICY_PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
else:
    sys.exit(0)
finally:
    sock.close()
PY
}

wait_for_server() {
  local deadline=$((SECONDS + ${POLICY_SERVER_WAIT_SECONDS:-900}))
  while (( SECONDS < deadline )); do
    if port_open; then
      echo "[INFO] Policy server is reachable at ${POLICY_HOST}:${POLICY_PORT}"
      return 0
    fi
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[ERROR] Policy server process exited before becoming reachable. Log tail:" >&2
      tail -80 "${SERVER_LOG}" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "[ERROR] Timed out waiting for policy server at ${POLICY_HOST}:${POLICY_PORT}. Log tail:" >&2
  tail -80 "${SERVER_LOG}" >&2 || true
  return 1
}

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT

if [[ "${START_POLICY_SERVER}" == "1" ]]; then
  if port_open; then
    echo "[ERROR] ${POLICY_HOST}:${POLICY_PORT} is already open. Use a free POLICY_PORT or set START_POLICY_SERVER=0 to reuse it." >&2
    exit 1
  fi

  echo "[INFO] Starting StarVLA policy server"
  echo "[INFO] policy_ckpt=${POLICY_CKPT}"
  echo "[INFO] policy_gpu=${POLICY_GPU} port=${POLICY_PORT}"
  (
    cd "${STARVLA_ROOT}"
    export REPO_ROOT="${STARVLA_ROOT}"
    export STARVLA_PYTHON="${STARVLA_PYTHON}"
    export ROBOTWIN_USE_BF16="${ROBOTWIN_USE_BF16}"
    export ROBOTWIN_NUM_INFERENCE_TIMESTEPS="${ROBOTWIN_NUM_INFERENCE_TIMESTEPS:-}"
    bash examples/Robotwin/eval_files/run_policy_server.sh "${POLICY_CKPT}" "${POLICY_GPU}" "${POLICY_PORT}"
  ) >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  wait_for_server
else
  echo "[INFO] Reusing existing policy server at ${POLICY_HOST}:${POLICY_PORT}"
  wait_for_server
fi

echo "[INFO] Running Ctrl-World autoregressive rollout with policy-server actions"
echo "[INFO] wm_ckpt=${WM_CKPT}"
echo "[INFO] stat=${STAT_PATH}"
echo "[INFO] out=${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${WM_GPU}" "${CTRLWORLD_PYTHON}" scripts/replay_policy_server_autoreg.py \
  --ckpt "${WM_CKPT}" \
  --latent_root "${LATENT_ROOT}" \
  --manifest "${MANIFEST_PATH}" \
  --stat "${STAT_PATH}" \
  --task "${TASK}" \
  --sample_index "${SAMPLE_INDEX}" \
  --steps 20 \
  --max_actions 0 \
  --chunk_size 32 \
  --num_history 6 \
  --num_frames 26 \
  --action_chunk_size 32 \
  --action_dim 14 \
  --decode_chunk 7 \
  --policy_host "${POLICY_HOST}" \
  --policy_port "${POLICY_PORT}" \
  --policy_mode vla \
  --policy_num_ddim_steps "${POLICY_NUM_DDIM_STEPS}" \
  --policy_bridge_python "${STARVLA_PYTHON}" \
  --starvla_root "${STARVLA_ROOT}" \
  --out "${OUT_DIR}" \
  2>&1 | tee "${CLIENT_LOG}"
