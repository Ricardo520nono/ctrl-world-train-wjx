#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ctrlworld_train_env.sh"

PROTOCOL="${1:-mix_4to1to1to1to1}"
if [[ "${PROTOCOL}" != "mix_4to1to1to1to1" && "${PROTOCOL}" != "mix_1to1to1to1to1" && "${PROTOCOL}" != "enhanced_1to1to1to1" && "${PROTOCOL}" != "mix_3to1to1to1" && "${PROTOCOL}" != "mix_1to1to1to1" && "${PROTOCOL}" != "clean_only" ]]; then
  echo "[ERROR] Unsupported protocol: ${PROTOCOL}" >&2
  exit 2
fi

TASKS_DEFAULT="place_burger_fries lift_pot dump_bin_bigbin rotate_qrcode"
TASKS="${TASKS:-${TASKS_DEFAULT}}"
RUN_NAME="${RUN_NAME:-ctrlworld_action_following_${PROTOCOL}_chunk32_$(date +%Y%m%d_%H%M%S)}"
INCLUDE_CLEAN="${INCLUDE_CLEAN:-1}"
INCLUDE_ENHANCED="${INCLUDE_ENHANCED:-1}"
if [[ "${PROTOCOL}" == "clean_only" ]]; then
  INCLUDE_ENHANCED=0
fi

ACTION_FOLLOWING_ROOT="${ACTION_FOLLOWING_ROOT:-/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingData/enhanced_v1_split}"
CLEAN_LEROBOT_ROOT="${CLEAN_LEROBOT_ROOT:-/mnt/dataset/csx_workspace/Ideas/data/ActionFollowingBench/data_lerobot/robotwin_delta_ee/demo_clean_zed2i_visible}"
CLEAN_LEROBOT_VIDEO_ROOT="${CLEAN_LEROBOT_VIDEO_ROOT:-${CLEAN_LEROBOT_ROOT}}"
if [[ "${PROTOCOL}" == "clean_only" ]]; then
  DEFAULT_LATENT_ROOT="${CACHE_ROOT}/action_following_chunk32_clean_only"
  DEFAULT_META_ROOT="${PROJECT_ROOT}/dataset_meta_info/action_following_chunk32_clean_only"
else
  DEFAULT_LATENT_ROOT="${CACHE_ROOT}/action_following_chunk32_clean_enhanced_explore"
  DEFAULT_META_ROOT="${PROJECT_ROOT}/dataset_meta_info/action_following_chunk32_clean_enhanced_explore"
fi
LATENT_ROOT="${LATENT_ROOT:-${DEFAULT_LATENT_ROOT}}"
META_ROOT="${META_ROOT:-${DEFAULT_META_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

SVD_PATH="${SVD_PATH:-${ASSET_ROOT}/stable-video-diffusion-img2vid}"
CLIP_PATH="${CLIP_PATH:-${ASSET_ROOT}/clip-vit-base-patch32}"
TASK_INSTRUCTION_ROOT="${TASK_INSTRUCTION_ROOT:-/mnt/dataset/csx_workspace/Ideas/AF3/code/RoboTwin/description/task_instruction}"

NUM_HISTORY="${NUM_HISTORY:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
NUM_FRAMES="${NUM_FRAMES:-${CHUNK_SIZE}}"
if [[ "${NUM_HISTORY}" -le 0 || "${NUM_FRAMES}" -le 0 ]]; then
  echo "[ERROR] NUM_HISTORY and NUM_FRAMES must be positive." >&2
  exit 2
fi
if [[ "${NUM_FRAMES}" -ne "${CHUNK_SIZE}" && "$((NUM_HISTORY + NUM_FRAMES))" -ne "${CHUNK_SIZE}" ]]; then
  echo "[ERROR] Unsupported ActionFollowing temporal layout: CHUNK_SIZE=${CHUNK_SIZE}, NUM_HISTORY=${NUM_HISTORY}, NUM_FRAMES=${NUM_FRAMES}." >&2
  echo "[ERROR] Use either current-frame layout NUM_FRAMES=CHUNK_SIZE or legacy layout NUM_HISTORY+NUM_FRAMES=CHUNK_SIZE." >&2
  exit 2
fi

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-40000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-0}"
CHECKPOINTING_EPOCHS="${CHECKPOINTING_EPOCHS:-1}"
VALIDATION_STEPS="${VALIDATION_STEPS:-2500}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
CKPT_PATH="${CKPT_PATH:-none}"
ACTION_DIM="${ACTION_DIM:-14}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29630}"
PRECOMPUTE_BATCH_SIZE="${PRECOMPUTE_BATCH_SIZE:-16}"
LIMIT_PER_FAMILY_TASK="${LIMIT_PER_FAMILY_TASK:-}"
CLEAN_LIMIT_PER_TASK="${CLEAN_LIMIT_PER_TASK:-}"
SAMPLER_AUDIT_SAMPLES="${SAMPLER_AUDIT_SAMPLES:-10000}"
SAMPLER_AUDIT_SEED="${SAMPLER_AUDIT_SEED:-20260630}"
SAMPLER_AUDIT_TOLERANCE="${SAMPLER_AUDIT_TOLERANCE:-0.02}"
REQUIRE_WANDB_ONLINE="${REQUIRE_WANDB_ONLINE:-0}"

if [[ "${REQUIRE_WANDB_ONLINE}" == "1" ]]; then
  if [[ "${WANDB_MODE:-online}" != "online" ]]; then
    echo "[ERROR] Formal training requires WANDB_MODE=online; got ${WANDB_MODE:-unset}." >&2
    exit 2
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[ERROR] Formal training requires WANDB_API_KEY from ${CTRLWORLD_ENV_FILE}." >&2
    exit 2
  fi
  echo "[INFO] W&B online preflight passed: project=ctrlworld_action_following run=${RUN_NAME}"
fi

mkdir -p "${LATENT_ROOT}" "${META_ROOT}" "${OUTPUT_DIR}"

need_quick=1
if [[ ! -f "${LATENT_ROOT}/manifests/train.jsonl" ]]; then
  echo "[INFO] Precomputing ActionFollowing train latents into ${LATENT_ROOT}"
  PRECOMPUTE_FLAGS=()
  if [[ "${INCLUDE_CLEAN}" == "1" ]]; then
    PRECOMPUTE_FLAGS+=(--include_clean)
  fi
  if [[ "${INCLUDE_ENHANCED}" == "1" ]]; then
    PRECOMPUTE_FLAGS+=(--include_enhanced)
  fi
  if [[ -n "${LIMIT_PER_FAMILY_TASK}" ]]; then
    PRECOMPUTE_FLAGS+=(--limit_per_family_task "${LIMIT_PER_FAMILY_TASK}")
  fi
  if [[ -n "${CLEAN_LIMIT_PER_TASK}" ]]; then
    PRECOMPUTE_FLAGS+=(--clean_limit_per_task "${CLEAN_LIMIT_PER_TASK}")
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/precompute_latents_action_following.py" \
    --svd_path "${SVD_PATH}" \
    --out_root "${LATENT_ROOT}" \
    --enhanced_split_root "${ACTION_FOLLOWING_ROOT}" \
    --clean_lerobot_root "${CLEAN_LEROBOT_ROOT}" \
    --clean_lerobot_video_root "${CLEAN_LEROBOT_VIDEO_ROOT}" \
    --task_instruction_root "${TASK_INSTRUCTION_ROOT}" \
    --tasks ${TASKS} \
    --split train \
    --batch_size "${PRECOMPUTE_BATCH_SIZE}" \
    "${PRECOMPUTE_FLAGS[@]}"
else
  echo "[INFO] Reusing existing ActionFollowing train manifest under ${LATENT_ROOT}"
fi

if [[ "${need_quick}" == "1" && ! -f "${LATENT_ROOT}/manifests/test_quick.jsonl" ]]; then
  echo "[INFO] Precomputing ActionFollowing test_quick latents into ${LATENT_ROOT}"
  QUICK_FLAGS=()
  if [[ -n "${LIMIT_PER_FAMILY_TASK}" ]]; then
    QUICK_FLAGS+=(--limit_per_family_task "${LIMIT_PER_FAMILY_TASK}")
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/precompute_latents_action_following.py" \
    --svd_path "${SVD_PATH}" \
    --out_root "${LATENT_ROOT}" \
    --enhanced_split_root "${ACTION_FOLLOWING_ROOT}" \
    --clean_lerobot_root "${CLEAN_LEROBOT_ROOT}" \
    --clean_lerobot_video_root "${CLEAN_LEROBOT_VIDEO_ROOT}" \
    --task_instruction_root "${TASK_INSTRUCTION_ROOT}" \
    --tasks ${TASKS} \
    --split test_quick \
    --include_enhanced \
    --batch_size "${PRECOMPUTE_BATCH_SIZE}" \
    "${QUICK_FLAGS[@]}"
elif [[ "${need_quick}" == "1" ]]; then
  echo "[INFO] Reusing existing ActionFollowing test_quick manifest under ${LATENT_ROOT}"
fi

if [[ "${PROTOCOL}" == "clean_only" ]]; then
  REQUIRED_TRAIN_FAMILIES="clean"
elif [[ "${PROTOCOL}" == "mix_4to1to1to1to1" || "${PROTOCOL}" == "mix_1to1to1to1to1" ]]; then
  REQUIRED_TRAIN_FAMILIES="clean,perturbed,random_feasible,counterfactual_replay,exploration"
elif [[ "${PROTOCOL}" == "enhanced_1to1to1to1" ]]; then
  REQUIRED_TRAIN_FAMILIES="perturbed,random_feasible,counterfactual_replay,exploration"
else
  REQUIRED_TRAIN_FAMILIES="clean,perturbed,random_feasible,counterfactual_replay"
fi
"${PYTHON_BIN}" - "${LATENT_ROOT}/manifests/train.jsonl" "${REQUIRED_TRAIN_FAMILIES}" <<'PY'
import json
import sys

manifest, required = sys.argv[1], [x for x in sys.argv[2].split(",") if x]

def canonical(rec):
    text = f"{rec.get('family', '')} {rec.get('asset_id', '')} {rec.get('subtype', '')}".lower()
    if "counterfactual" in text:
        return "counterfactual_replay"
    if "exploration" in text or "policy_rollout" in text:
        return "exploration"
    if "random_feasible" in text:
        return "random_feasible"
    if "perturbed" in text or "pca" in text or "raw" in text:
        return "perturbed"
    if "clean" in text:
        return "clean"
    return rec.get("family", "")

counts = {}
with open(manifest) as f:
    for line in f:
        if line.strip():
            fam = canonical(json.loads(line))
            counts[fam] = counts.get(fam, 0) + 1
missing = [fam for fam in required if counts.get(fam, 0) <= 0]
if missing:
    raise SystemExit(f"manifest {manifest} missing required families: {missing}; counts={counts}")
print("[INFO] train manifest family counts:", counts)
PY

if [[ "${PROTOCOL}" == "mix_4to1to1to1to1" || "${PROTOCOL}" == "mix_1to1to1to1to1" || "${PROTOCOL}" == "enhanced_1to1to1to1" ]]; then
  REQUIRED_VAL_FAMILIES="perturbed,random_feasible,counterfactual_replay,exploration"
else
  REQUIRED_VAL_FAMILIES="perturbed,random_feasible,counterfactual_replay"
fi
"${PYTHON_BIN}" - "${LATENT_ROOT}/manifests/${VAL_MANIFEST_NAME:-test_quick}.jsonl" "${REQUIRED_VAL_FAMILIES}" <<'PY'
import json
import sys

manifest, required = sys.argv[1], [x for x in sys.argv[2].split(",") if x]

def canonical(rec):
    text = f"{rec.get('family', '')} {rec.get('asset_id', '')} {rec.get('subtype', '')}".lower()
    if "counterfactual" in text:
        return "counterfactual_replay"
    if "exploration" in text or "policy_rollout" in text:
        return "exploration"
    if "random_feasible" in text:
        return "random_feasible"
    if "perturbed" in text or "pca" in text or "raw" in text:
        return "perturbed"
    if "clean" in text:
        return "clean"
    return rec.get("family", "")

counts = {}
with open(manifest) as f:
    for line in f:
        if line.strip():
            fam = canonical(json.loads(line))
            counts[fam] = counts.get(fam, 0) + 1
missing = [fam for fam in required if counts.get(fam, 0) <= 0]
if missing:
    raise SystemExit(f"manifest {manifest} missing required families: {missing}; counts={counts}")
print("[INFO] val manifest family counts:", counts)
PY

if [[ ! -f "${META_ROOT}/stat.json" ]]; then
  echo "[INFO] Computing ActionFollowing action stats into ${META_ROOT}"
  STAT_MANIFEST="${STAT_MANIFEST:-}"
  if [[ -z "${STAT_MANIFEST}" ]]; then
    if [[ "${PROTOCOL}" == "clean_only" && -f "${LATENT_ROOT}/manifests/clean_train.jsonl" ]]; then
      STAT_MANIFEST="${LATENT_ROOT}/manifests/clean_train.jsonl"
    else
      STAT_MANIFEST="${LATENT_ROOT}/manifests/train.jsonl"
    fi
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/compute_stat_action_following.py" \
    --latent_root "${LATENT_ROOT}" \
    --manifest "${STAT_MANIFEST}" \
    --out_dir "${META_ROOT}" \
    --action_dim "${ACTION_DIM}"
else
  echo "[INFO] Reusing existing stat: ${META_ROOT}/stat.json"
fi

echo "[INFO] Auditing ActionFollowing sampler protocol=${PROTOCOL}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/audit_action_following_sampler.py" \
  --latent_root "${LATENT_ROOT}" \
  --train_manifest "${LATENT_ROOT}/manifests/train.jsonl" \
  --val_manifest "${LATENT_ROOT}/manifests/${VAL_MANIFEST_NAME:-test_quick}.jsonl" \
  --stat_path "${META_ROOT}/stat.json" \
  --protocol "${PROTOCOL}" \
  --chunk_size "${CHUNK_SIZE}" \
  --num_history "${NUM_HISTORY}" \
  --num_frames "${NUM_FRAMES}" \
  --action_dim "${ACTION_DIM}" \
  --num_samples "${SAMPLER_AUDIT_SAMPLES}" \
  --seed "${SAMPLER_AUDIT_SEED}" \
  --tolerance "${SAMPLER_AUDIT_TOLERANCE}"

{
  echo "RUN_NAME=${RUN_NAME}"
  echo "PROTOCOL=${PROTOCOL}"
  echo "TASKS=${TASKS}"
  echo "ACTION_FOLLOWING_ROOT=${ACTION_FOLLOWING_ROOT}"
  echo "CLEAN_LEROBOT_ROOT=${CLEAN_LEROBOT_ROOT}"
  echo "CLEAN_LEROBOT_VIDEO_ROOT=${CLEAN_LEROBOT_VIDEO_ROOT}"
  echo "LATENT_ROOT=${LATENT_ROOT}"
  echo "META_ROOT=${META_ROOT}"
  echo "TASK_INSTRUCTION_ROOT=${TASK_INSTRUCTION_ROOT}"
  echo "STAT_MANIFEST=${STAT_MANIFEST:-}"
  echo "INCLUDE_CLEAN=${INCLUDE_CLEAN}"
  echo "INCLUDE_ENHANCED=${INCLUDE_ENHANCED}"
  echo "LIMIT_PER_FAMILY_TASK=${LIMIT_PER_FAMILY_TASK}"
  echo "CLEAN_LIMIT_PER_TASK=${CLEAN_LIMIT_PER_TASK}"
  echo "CHUNK_SIZE=${CHUNK_SIZE}"
  echo "NUM_HISTORY=${NUM_HISTORY}"
  echo "NUM_FRAMES=${NUM_FRAMES}"
  echo "ACTION_DIM=${ACTION_DIM}"
  echo "SAMPLER_AUDIT_SAMPLES=${SAMPLER_AUDIT_SAMPLES}"
  echo "SAMPLER_AUDIT_SEED=${SAMPLER_AUDIT_SEED}"
  echo "SAMPLER_AUDIT_TOLERANCE=${SAMPLER_AUDIT_TOLERANCE}"
  echo "WANDB_MODE=${WANDB_MODE:-online}"
  echo "WANDB_PROJECT=ctrlworld_action_following"
  echo "WANDB_RUN_NAME=${RUN_NAME}"
  echo "CKPT_PATH=${CKPT_PATH}"
} > "${OUTPUT_DIR}/launch_cmd.txt"

echo "[INFO] Launching ${RUN_NAME}"
"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT_ROOT}/scripts/train_delta_ee.py" \
  --dataset_type action_following \
  --svd_model_path "${SVD_PATH}" \
  --clip_model_path "${CLIP_PATH}" \
  --ckpt_path "${CKPT_PATH}" \
  --dataset_root_path "${LATENT_ROOT}" \
  --dataset_meta_info_path "$(dirname "${META_ROOT}")" \
  --dataset_cfgs "$(basename "${META_ROOT}")" \
  --dataset_names "$(echo "${TASKS}" | tr ' ' '+')" \
  --action_following_latent_root "${LATENT_ROOT}" \
  --action_following_train_manifest "${LATENT_ROOT}/manifests/train.jsonl" \
  --action_following_val_manifest "${LATENT_ROOT}/manifests/${VAL_MANIFEST_NAME:-test_quick}.jsonl" \
  --action_following_stat_path "${META_ROOT}/stat.json" \
  --action_following_sampling_protocol "${PROTOCOL}" \
  --action_following_chunk_size "${CHUNK_SIZE}" \
  --action_following_sampling_seed "${SAMPLER_AUDIT_SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  --wandb_project_name ctrlworld_action_following \
  --wandb_run_name "${RUN_NAME}" \
  --tag "${RUN_NAME}" \
  --action_dim "${ACTION_DIM}" \
  --height 240 \
  --num_history "${NUM_HISTORY}" \
  --num_frames "${NUM_FRAMES}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS}" \
  --checkpointing_epochs "${CHECKPOINTING_EPOCHS}" \
  --validation_steps "${VALIDATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --mixed_precision "${MIXED_PRECISION}" \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"
