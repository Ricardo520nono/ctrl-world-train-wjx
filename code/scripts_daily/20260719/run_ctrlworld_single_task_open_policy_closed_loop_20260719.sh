#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}"

TASK="${TASK:?TASK is required}"
WM_CKPT="${WM_CKPT:?WM_CKPT is required}"
LATENT_ROOT="${LATENT_ROOT:?LATENT_ROOT is required}"
STAT_PATH="${STAT_PATH:?STAT_PATH is required}"
MANIFEST_PATH="${MANIFEST_PATH:-${LATENT_ROOT}/manifests/clean_train.jsonl}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"

POLICY_CKPT="${POLICY_CKPT:-/mnt/dataset/csx_workspace/Ideas/data_AF3/starVLA_dev/checkpoints/qwenoft_rot6d20_clean50_randomized500_decord_lru8_nofuture_locality_b4_official_48g_bs4/ACWM_T2A_qwenoft_rot6d20_clean50_randomized500_decord_lru8_nofuture_locality_b4_official_48g_bs4_20260709/checkpoints/steps_90000_pytorch_model.pt}"
POLICY_RUN_DIR="$(cd "$(dirname "${POLICY_CKPT}")/.." && pwd)"
POLICY_STAT_PATH="${POLICY_STAT_PATH:-${POLICY_RUN_DIR}/dataset_statistics.json}"
POLICY_STAT_KEY="${POLICY_STAT_KEY:-new_embodiment}"
POLICY_VIEW_ORDER="${POLICY_VIEW_ORDER:-cam_high,cam_left_wrist,cam_right_wrist}"
POLICY_BASE_VLM="${POLICY_BASE_VLM:-/mnt/dataset/public_data/Qwen3-VL-4B-Instruct}"
POLICY_LEGACY_BASE_VLM="${POLICY_LEGACY_BASE_VLM:-/mnt/public_ckp/Qwen3-VL-4B-Instruct}"
STARVLA_ROOT="${STARVLA_ROOT:-/mnt/dataset/csx_workspace/Ideas/AF3/code/starVLA_dev}"
STARVLA_PYTHON="${STARVLA_PYTHON:-${STARVLA_ROOT}/.venv/bin/python}"
CTRLWORLD_PYTHON="${CTRLWORLD_PYTHON:-/mnt/dataset/csx_workspace/Ideas/WMProbe/.venv/bin/python}"

SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
WM_STEPS="${WM_STEPS:-20}"
POLICY_DDIM_STEPS="${POLICY_DDIM_STEPS:-10}"
SEED="${SEED:-20260719}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5694}"
POLICY_GPU="${POLICY_GPU:-0}"
OPEN_GPU="${OPEN_GPU:-1}"
CLOSED_GPU="${CLOSED_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"

OPEN_OUT="${OUT_ROOT}/open_loop_expert_actions"
CLOSED_OUT="${OUT_ROOT}/closed_loop_qwenoft_step90000"
SERVER_LOG="${OUT_ROOT}/policy_server.log"
OPEN_LOG="${OUT_ROOT}/open_loop.log"
CLOSED_LOG="${OUT_ROOT}/closed_loop.log"

for path in \
  "${WM_CKPT}" \
  "${LATENT_ROOT}" \
  "${MANIFEST_PATH}" \
  "${STAT_PATH}" \
  "${POLICY_CKPT}" \
  "${POLICY_STAT_PATH}" \
  "${POLICY_BASE_VLM}" \
  "${STARVLA_ROOT}" \
  "${STARVLA_PYTHON}" \
  "${CTRLWORLD_PYTHON}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing required path: ${path}" >&2
    exit 2
  fi
done

case "${OUT_ROOT}" in
  /mnt/dataset/public_data/cscsx_projects/AF3/*) ;;
  *)
    if [[ "${DRY_RUN}" != "1" ]]; then
      echo "[ERROR] Formal output must be under /mnt/dataset/public_data/cscsx_projects/AF3: ${OUT_ROOT}" >&2
      exit 2
    fi
    ;;
esac

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] task=${TASK} sample_index=${SAMPLE_INDEX}"
  echo "[DRY_RUN] wm_ckpt=${WM_CKPT}"
  echo "[DRY_RUN] policy_ckpt=${POLICY_CKPT}"
  echo "[DRY_RUN] policy_stat=${POLICY_STAT_PATH} key=${POLICY_STAT_KEY}"
  echo "[DRY_RUN] policy_view_order=${POLICY_VIEW_ORDER}"
  echo "[DRY_RUN] policy_base_vlm=${POLICY_BASE_VLM} legacy_path=${POLICY_LEGACY_BASE_VLM}"
  echo "[DRY_RUN] latent_root=${LATENT_ROOT}"
  echo "[DRY_RUN] manifest=${MANIFEST_PATH}"
  echo "[DRY_RUN] stat=${STAT_PATH}"
  echo "[DRY_RUN] wm_steps=${WM_STEPS} policy_ddim_steps=${POLICY_DDIM_STEPS}"
  echo "[DRY_RUN] policy_gpu=${POLICY_GPU} open_gpu=${OPEN_GPU} closed_gpu=${CLOSED_GPU} port=${POLICY_PORT}"
  echo "[DRY_RUN] out_root=${OUT_ROOT}"
  exit 0
fi

mkdir -p "$(dirname "${POLICY_LEGACY_BASE_VLM}")"
if [[ -e "${POLICY_LEGACY_BASE_VLM}" || -L "${POLICY_LEGACY_BASE_VLM}" ]]; then
  if [[ "$(realpath "${POLICY_LEGACY_BASE_VLM}")" != "$(realpath "${POLICY_BASE_VLM}")" ]]; then
    echo "[ERROR] Existing legacy policy base path points elsewhere: ${POLICY_LEGACY_BASE_VLM}" >&2
    exit 2
  fi
else
  ln -s "${POLICY_BASE_VLM}" "${POLICY_LEGACY_BASE_VLM}"
fi

if [[ -d "${OUT_ROOT}" && -n "$(find "${OUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "[ERROR] Refusing non-empty output directory: ${OUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OPEN_OUT}" "${CLOSED_OUT}"

cat > "${OUT_ROOT}/eval_contract.txt" <<EOF
TASK=${TASK}
SAMPLE_INDEX=${SAMPLE_INDEX}
WM_CKPT=${WM_CKPT}
POLICY_CKPT=${POLICY_CKPT}
POLICY_STAT_PATH=${POLICY_STAT_PATH}
POLICY_STAT_KEY=${POLICY_STAT_KEY}
POLICY_VIEW_ORDER=${POLICY_VIEW_ORDER}
POLICY_BASE_VLM=${POLICY_BASE_VLM}
POLICY_LEGACY_BASE_VLM=${POLICY_LEGACY_BASE_VLM}
LATENT_ROOT=${LATENT_ROOT}
MANIFEST_PATH=${MANIFEST_PATH}
STAT_PATH=${STAT_PATH}
WM_STEPS=${WM_STEPS}
POLICY_DDIM_STEPS=${POLICY_DDIM_STEPS}
SEED=${SEED}
POLICY_GPU=${POLICY_GPU}
OPEN_GPU=${OPEN_GPU}
CLOSED_GPU=${CLOSED_GPU}
POLICY_PORT=${POLICY_PORT}
EOF

port_open() {
  "${STARVLA_PYTHON}" - "${POLICY_HOST}" "${POLICY_PORT}" <<'PY'
import os
import sys

for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(key, None)
try:
    from websockets.sync.client import connect

    conn = connect(
        f"ws://{sys.argv[1]}:{int(sys.argv[2])}",
        compression=None,
        max_size=None,
        open_timeout=1,
        close_timeout=1,
    )
    conn.recv(timeout=2)
    conn.close()
except Exception:
    raise SystemExit(1)
PY
}

SERVER_PID=""
OPEN_PID=""
CLOSED_PID=""
cleanup() {
  for pid in "${OPEN_PID}" "${CLOSED_PID}" "${SERVER_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

if port_open; then
  echo "[ERROR] Policy port is already in use: ${POLICY_HOST}:${POLICY_PORT}" >&2
  exit 2
fi

echo "[INFO] Starting QwenOFT policy server task=${TASK} gpu=${POLICY_GPU} port=${POLICY_PORT}"
(
  cd "${STARVLA_ROOT}"
  export REPO_ROOT="${STARVLA_ROOT}"
  export STARVLA_PYTHON
  export ROBOTWIN_USE_BF16="${ROBOTWIN_USE_BF16:-1}"
  bash examples/Robotwin/eval_files/run_policy_server.sh "${POLICY_CKPT}" "${POLICY_GPU}" "${POLICY_PORT}"
) > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + ${POLICY_SERVER_WAIT_SECONDS:-900}))
while ! port_open; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] Policy server exited before readiness: ${SERVER_LOG}" >&2
    tail -n 100 "${SERVER_LOG}" >&2
    exit 2
  fi
  if (( SECONDS >= deadline )); then
    echo "[ERROR] Timed out waiting for policy server: ${SERVER_LOG}" >&2
    exit 2
  fi
  sleep 5
done

echo "[INFO] Starting open-loop and policy closed-loop rollouts in parallel"
CUDA_VISIBLE_DEVICES="${OPEN_GPU}" "${CTRLWORLD_PYTHON}" scripts/replay_clean_expert_autoreg.py \
  --ckpt "${WM_CKPT}" \
  --latent_root "${LATENT_ROOT}" \
  --manifest "${MANIFEST_PATH}" \
  --stat "${STAT_PATH}" \
  --task "${TASK}" \
  --sample_index "${SAMPLE_INDEX}" \
  --steps "${WM_STEPS}" \
  --max_actions 0 \
  --chunk_size 32 \
  --num_history 1 \
  --num_frames 32 \
  --action_chunk_size 32 \
  --action_dim 20 \
  --decode_chunk 7 \
  --seed "${SEED}" \
  --out "${OPEN_OUT}" > "${OPEN_LOG}" 2>&1 &
OPEN_PID=$!

CUDA_VISIBLE_DEVICES="${CLOSED_GPU}" "${CTRLWORLD_PYTHON}" scripts/replay_policy_server_autoreg.py \
  --ckpt "${WM_CKPT}" \
  --latent_root "${LATENT_ROOT}" \
  --manifest "${MANIFEST_PATH}" \
  --stat "${STAT_PATH}" \
  --task "${TASK}" \
  --sample_index "${SAMPLE_INDEX}" \
  --steps "${WM_STEPS}" \
  --max_actions 0 \
  --chunk_size 32 \
  --num_history 1 \
  --num_frames 32 \
  --action_chunk_size 32 \
  --action_dim 20 \
  --decode_chunk 7 \
  --seed "${SEED}" \
  --policy_host "${POLICY_HOST}" \
  --policy_port "${POLICY_PORT}" \
  --policy_mode vla \
  --policy_num_ddim_steps "${POLICY_DDIM_STEPS}" \
  --policy_ckpt "${POLICY_CKPT}" \
  --policy_stat "${POLICY_STAT_PATH}" \
  --policy_stat_key "${POLICY_STAT_KEY}" \
  --policy_view_order "${POLICY_VIEW_ORDER}" \
  --policy_bridge_python "${STARVLA_PYTHON}" \
  --starvla_root "${STARVLA_ROOT}" \
  --out "${CLOSED_OUT}" > "${CLOSED_LOG}" 2>&1 &
CLOSED_PID=$!

set +e
wait "${OPEN_PID}"
open_status=$?
OPEN_PID=""
wait "${CLOSED_PID}"
closed_status=$?
CLOSED_PID=""
set -e

if [[ "${open_status}" -ne 0 || "${closed_status}" -ne 0 ]]; then
  echo "[ERROR] Evaluation failed: open_status=${open_status} closed_status=${closed_status}" >&2
  exit 2
fi

touch "${OUT_ROOT}/.complete"
echo "[INFO] Completed task=${TASK} out=${OUT_ROOT}"
