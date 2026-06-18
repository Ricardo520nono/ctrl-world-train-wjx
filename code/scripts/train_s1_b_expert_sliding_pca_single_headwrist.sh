#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ctrlworld_train_env.sh"

# ============================================================================
# S1-B: expert(sliding) + PCA-perturbed(single window/episode), 5 tasks,
# 14D abs-joint, nf=16, 40k steps.
# Camera set: head_camera + left_camera + right_camera (head + dual wrist),
# stacked vertically top->bottom -> latent (T,4,90,40).
# Checkpointing: every 1 completed epoch (step-based saving disabled).
# Derived from train_s1_b_expert_sliding_pca_single.sh; only camera set,
# latent paths, run name and checkpoint policy differ.
# ============================================================================

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
RUN_NAME="s1_B_expert_sliding_pca_single_headwrist_${RUN_TS}"
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

for pkg in "transformers>=4.46.0" "accelerate>=0.34.0" "peft==0.15.2" "safetensors==0.5.3" \
           sentencepiece einops decord imageio mediapy omegaconf wandb h5py; do
  install_pkg "$pkg"
done

"${PYTHON_BIN}" -m pip install -i https://pypi.org/simple "diffusers==0.34.0"
"${PYTHON_BIN}" -m pip uninstall -y swanlab || true

echo "[INFO] Verifying runtime dependencies..."
"${PYTHON_BIN}" -c "import diffusers,transformers,accelerate,torch,h5py; from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import TextToVideoSDPipelineOutput; print('deps_ok', diffusers.__version__, torch.__version__)"

# ---- camera set & paths ----
CAMERAS="head_camera,left_camera,right_camera"   # head + dual wrist, top->bottom
SVD_PATH="${ASSET_ROOT}/stable-video-diffusion-img2vid"
# Dedicated roots (NOT shared with the all50 headwrist job nor with S1-A) to avoid
# concurrent-write corruption when both jobs are submitted at the same time.
EXPERT_LATENT_ROOT="${CACHE_ROOT}/precomputed_latents_s1B_5tasks_14d_headwrist"
PCA_LATENT_ROOT="${CACHE_ROOT}/s1B_latents_pca_train_headwrist"
DATA_ROOT_ORIG="/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/data_delta_ee/demo_clean_zed2i_visible"
DATA_ROOT_PCA="/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/EnhancedData/perturbed_pca_gaussian/c_8_sigma_0p05"
META_INFO_BASE="${PROJECT_ROOT}/dataset_meta_info"
S1_TASKS="click_alarmclock click_bell place_object_basket open_laptop stack_blocks_two"

# ---- pre-encode expert latents for the 5 S1 tasks with new cameras (skip if exist) ----
echo "[INFO] Pre-encoding expert latents (cameras=${CAMERAS}) for S1 tasks..."
for TASK in ${S1_TASKS}; do
  TASK_LATENT_DIR="${EXPERT_LATENT_ROOT}/${TASK}"
  if [[ -f "${TASK_LATENT_DIR}/meta.json" ]]; then
    echo "[INFO] Expert latents already exist for ${TASK}, skipping."
    continue
  fi
  "${PYTHON_BIN}" ${PROJECT_ROOT}/scripts/precompute_latents_delta_ee.py \
    --data_dir  "${DATA_ROOT_ORIG}/${TASK}/data" \
    --out_dir   "${TASK_LATENT_DIR}" \
    --svd_path  "${SVD_PATH}" \
    --task_name "${TASK}" \
    --cameras   "${CAMERAS}"
done

# ---- pre-encode PCA-perturbed latents (ep0-39) with new cameras (skip if exist) ----
echo "[INFO] Pre-encoding PCA-perturbed latents (cameras=${CAMERAS}, ep0-39) for S1 tasks..."
"${PYTHON_BIN}" ${PROJECT_ROOT}/scripts/precompute_latents_s1_pca.py \
  --data_root   "${DATA_ROOT_PCA}" \
  --out_root    "${PCA_LATENT_ROOT}" \
  --svd_path    "${SVD_PATH}" \
  --tasks       ${S1_TASKS} \
  --episode_min 0 --episode_max 39 \
  --cameras     "${CAMERAS}"

# ---- action stat: reuse (action is camera-independent) ----
if [[ ! -f "${META_INFO_BASE}/s1_B_expert_plus_pca/stat.json" ]]; then
  echo "[ERROR] Expected reusable stat not found at ${META_INFO_BASE}/s1_B_expert_plus_pca/stat.json"
  exit 1
fi
echo "[INFO] Reusing s1_B_expert_plus_pca stat (action is camera-independent)."

# ---- training ----
DATASET_NAMES="click_alarmclock+click_bell+place_object_basket+open_laptop+stack_blocks_two"
DATASET_ROOT="${EXPERT_LATENT_ROOT}+${PCA_LATENT_ROOT}"

printf '%s\n' "$(date): ${RUN_NAME} (cameras=${CAMERAS}, ckpt=per-epoch)" > "${OUTPUT_DIR}/launch_cmd.txt"
echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] CAMERAS=${CAMERAS}"
echo "[INFO] Launching S1-B (expert sliding + pca single, 5 tasks, nf=16, head+dual-wrist, 40k steps, ckpt per-epoch)..."

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port=29617 \
  ${PROJECT_ROOT}/scripts/train_delta_ee.py \
  --svd_model_path    "${SVD_PATH}" \
  --clip_model_path   ${ASSET_ROOT}/clip-vit-base-patch32 \
  --ckpt_path         none \
  --dataset_root_path "${DATASET_ROOT}" \
  --dataset_root_sampling "sliding+single" \
  --dataset_meta_info_path "${META_INFO_BASE}" \
  --dataset_names     "${DATASET_NAMES}" \
  --dataset_cfgs      s1_B_expert_plus_pca \
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
