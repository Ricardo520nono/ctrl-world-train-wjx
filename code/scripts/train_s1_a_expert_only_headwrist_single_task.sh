#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ctrlworld_train_env.sh"

# ============================================================================
# S1-A single-task: expert-only, 14D abs-joint, nf=16, 40k steps.
# Camera set: head_camera + left_camera + right_camera (head + dual wrist),
# stacked vertically top->bottom -> latent (T,4,90,40).
# Checkpointing: every 1 completed epoch (step-based saving disabled).
#
# Usage:
#   bash scripts/train_s1_a_expert_only_headwrist_single_task.sh <task_name>
# ============================================================================

if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 <task_name>"
  echo "Valid tasks: click_alarmclock click_bell place_object_basket open_laptop stack_blocks_two"
  exit 2
fi

TASK="$1"
case "${TASK}" in
  click_alarmclock|click_bell|place_object_basket|open_laptop|stack_blocks_two)
    ;;
  *)
    echo "[ERROR] Unknown S1 task: ${TASK}"
    echo "Valid tasks: click_alarmclock click_bell place_object_basket open_laptop stack_blocks_two"
    exit 2
    ;;
esac

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

ENV_FILE="${CTRLWORLD_ENV_FILE}"
if [[ -f "${ENV_FILE}" ]]; then
  set +u
  source "${ENV_FILE}"
  set -u
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[ERROR] WANDB_API_KEY is empty."
  exit 1
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="s1_A_expert_only_${TASK}_headwrist_${RUN_TS}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

export WANDB_MODE="online"
export WANDB_PROJECT="ctrlworld_s1"
export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_NAME="${RUN_NAME}"
export SWANLAB_MODE="local"
export TOKENIZERS_PARALLELISM="false"

# ---- dependency install ----
echo "[INFO] Installing required Python packages..."
"${PYTHON_BIN}" -m pip install -U pip setuptools wheel

install_pkg() {
  local pkg="$1"
  "${PYTHON_BIN}" -m pip install "$pkg" || \
    "${PYTHON_BIN}" -m pip install -i https://pypi.org/simple "$pkg"
}

for pkg in "transformers==4.48.1" "accelerate>=0.34.0" "peft==0.15.2" "safetensors==0.5.3" \
           sentencepiece einops decord imageio mediapy omegaconf wandb h5py; do
  install_pkg "$pkg"
done

"${PYTHON_BIN}" -m pip install -i https://pypi.org/simple "diffusers==0.34.0"
"${PYTHON_BIN}" -m pip uninstall -y swanlab || true

echo "[INFO] Verifying runtime dependencies..."
"${PYTHON_BIN}" -c "import diffusers,transformers,accelerate,torch,h5py; from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import TextToVideoSDPipelineOutput; print('deps_ok', diffusers.__version__, torch.__version__)"

# ---- camera set & paths ----
CAMERAS="head_camera,left_camera,right_camera"
SVD_PATH="${ASSET_ROOT}/stable-video-diffusion-img2vid"
EXPERT_LATENT_ROOT="${CACHE_ROOT}/precomputed_latents_s1A_5tasks_14d_headwrist"
DATA_ROOT_ORIG="/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/data_delta_ee/demo_clean_zed2i_visible"
META_INFO_BASE="${PROJECT_ROOT}/dataset_meta_info"
DATASET_CFGS="s1_A_expert_only"

# ---- pre-encode expert latents for this S1 task with headwrist cameras ----
echo "[INFO] Pre-encoding expert latents (cameras=${CAMERAS}) for task=${TASK}..."
TASK_LATENT_DIR="${EXPERT_LATENT_ROOT}/${TASK}"
if [[ -f "${TASK_LATENT_DIR}/meta.json" ]]; then
  echo "[INFO] Expert latents already exist for ${TASK}, skipping."
else
  "${PYTHON_BIN}" ${PROJECT_ROOT}/scripts/precompute_latents_delta_ee.py \
    --data_dir  "${DATA_ROOT_ORIG}/${TASK}/data" \
    --out_dir   "${TASK_LATENT_DIR}" \
    --svd_path  "${SVD_PATH}" \
    --task_name "${TASK}" \
    --cameras   "${CAMERAS}"
fi

# ---- action stat: reuse the original S1-A stat for consistency with 5-task run ----
if [[ ! -f "${META_INFO_BASE}/${DATASET_CFGS}/stat.json" ]]; then
  echo "[INFO] Computing S1-A stat because ${META_INFO_BASE}/${DATASET_CFGS}/stat.json is missing..."
  "${PYTHON_BIN}" ${PROJECT_ROOT}/scripts/compute_stat_s1.py \
    --group         A \
    --expert_root   "${EXPERT_LATENT_ROOT}" \
    --tasks         click_alarmclock click_bell place_object_basket open_laptop stack_blocks_two \
    --episode_split 0-39 \
    --out_dir       "${META_INFO_BASE}/${DATASET_CFGS}" \
    --action_dim    14
else
  echo "[INFO] S1-A stat.json already exists (reusing, action is camera-independent)."
fi

# ---- training ----
DATASET_NAMES="${TASK}"

printf '%s\n' "$(date): ${RUN_NAME} (task=${TASK}, cameras=${CAMERAS}, ckpt=per-epoch)" > "${OUTPUT_DIR}/launch_cmd.txt"
echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] TASK=${TASK}"
echo "[INFO] CAMERAS=${CAMERAS}"
echo "[INFO] Launching S1-A single-task training (expert only, nf=16, head+dual-wrist, 40k steps, ckpt per-epoch)..."

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port=29615 \
  ${PROJECT_ROOT}/scripts/train_delta_ee.py \
  --svd_model_path    "${SVD_PATH}" \
  --clip_model_path   ${ASSET_ROOT}/clip-vit-base-patch32 \
  --ckpt_path         none \
  --dataset_root_path "${EXPERT_LATENT_ROOT}" \
  --dataset_meta_info_path "${META_INFO_BASE}" \
  --dataset_names     "${DATASET_NAMES}" \
  --dataset_cfgs      "${DATASET_CFGS}" \
  --episode_split     0-39 \
  --val_episode_split 40-49 \
  --output_dir        "${OUTPUT_DIR}" \
  --wandb_project_name ctrlworld_s1 \
  --wandb_run_name    "${RUN_NAME}" \
  --tag               "${RUN_NAME}" \
  --action_dim        14 \
  --height            240 \
  --num_history       6 \
  --num_frames        16 \
  --train_batch_size  1 \
  --gradient_accumulation_steps 2 \
  --max_train_steps   40000 \
  --checkpointing_steps 0 \
  --checkpointing_epochs 1 \
  --validation_steps  2500 \
  --learning_rate     1e-5 \
  --mixed_precision   bf16 \
  --use_abs_joint_action \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"
