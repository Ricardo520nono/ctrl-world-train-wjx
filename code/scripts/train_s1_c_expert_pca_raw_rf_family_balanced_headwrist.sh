#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ctrlworld_train_env.sh"

# ============================================================================
# S1-C main experiment: expert + PCA + raw + random-feasible, 3:1:1:1
# family-balanced sampler, 5 tasks, 14D abs-joint, nf=16, 40k steps.
#
# Camera set: head_camera + left_camera + right_camera, stacked top->bottom.
# Intended for Baidu 8-GPU training.
# ============================================================================

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

ENV_FILE="${CTRLWORLD_ENV_FILE}"
if [[ -f "${ENV_FILE}" ]]; then
  set +u
  source "${ENV_FILE}"
  set -u
fi

WANDB_MODE="${WANDB_MODE:-online}"
if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "[ERROR] WANDB_API_KEY is empty."
  exit 1
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-s1_C_3to1to1to1_family_balanced_headwrist}"
RUN_NAME="${RUN_NAME_PREFIX}_${RUN_TS}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/run.log") 2>&1

export WANDB_MODE
export WANDB_PROJECT="ctrlworld_s1"
export WANDB_NAME="${RUN_NAME}"
export WANDB_RUN_NAME="${RUN_NAME}"
export SWANLAB_MODE="local"
export TOKENIZERS_PARALLELISM="false"

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

PRECOMPUTE_GPUS="${PRECOMPUTE_GPUS:-8}"

CAMERAS="head_camera,left_camera,right_camera"
SVD_PATH="${ASSET_ROOT}/stable-video-diffusion-img2vid"
META_INFO_BASE="${PROJECT_ROOT}/dataset_meta_info"
DATASET_CFGS="s1_C_expert_pca_raw_rf_3to1to1to1"

S1_TASKS="click_alarmclock click_bell place_object_basket open_laptop stack_blocks_two"
DATASET_NAMES="click_alarmclock+click_bell+place_object_basket+open_laptop+stack_blocks_two"

# Existing headwrist roots from previous S1 runs.
EXPERT_LATENT_ROOT="${CACHE_ROOT}/precomputed_latents_s1A_5tasks_14d_headwrist"
PCA_LATENT_ROOT="${CACHE_ROOT}/s1B_latents_pca_train_headwrist"

# New roots for this experiment.
RAW_LATENT_ROOT="${CACHE_ROOT}/s1C_latents_raw_s0025_train_headwrist"
RF_UNIFORM_LATENT_ROOT="${CACHE_ROOT}/s1C_latents_rf300_uniform_train_headwrist"
RF_WEIGHTED_LATENT_ROOT="${CACHE_ROOT}/s1C_latents_rf300_weighted_train_headwrist"

ACTION_FOLLOWING_BENCH_ROOT="${ACTION_FOLLOWING_BENCH_ROOT:-/mnt/dataset/public_data/cscsx_projects/data/ActionFollowingBench}"
DATA_ROOT_ORIG="${DATA_ROOT_ORIG:-${ACTION_FOLLOWING_BENCH_ROOT}/data_delta_ee/demo_clean_zed2i_visible}"
DATA_ROOT_PCA="${DATA_ROOT_PCA:-${ACTION_FOLLOWING_BENCH_ROOT}/EnhancedData/perturbed_pca_gaussian/c_8_sigma_0p05}"
DATA_ROOT_RAW="${DATA_ROOT_RAW:-${ACTION_FOLLOWING_BENCH_ROOT}/EnhancedData/perturbed_raw_gaussian/sigma_0p0025}"
RF_BASE="${RF_BASE:-${ACTION_FOLLOWING_BENCH_ROOT}/EnhancedData/random_feasible_300step_5task_2ep5start_formal_v1/random_feasible_random_walk}"
DATA_ROOT_RF_UNIFORM="${DATA_ROOT_RF_UNIFORM:-${RF_BASE}/rf_5task_300step_2ep5start_formal_uniform_10seed_v1}"
DATA_ROOT_RF_WEIGHTED="${DATA_ROOT_RF_WEIGHTED:-${RF_BASE}/rf_5task_300step_2ep5start_formal_weighted_10seed_v1}"

wait_for_background_jobs() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  return "${status}"
}

echo "[INFO] Preparing expert headwrist latents..."
expert_pids=()
expert_idx=0
for TASK in ${S1_TASKS}; do
  TASK_LATENT_DIR="${EXPERT_LATENT_ROOT}/${TASK}"
  if [[ -f "${TASK_LATENT_DIR}/meta.json" ]]; then
    echo "[INFO] Expert latents already exist for ${TASK}, skipping."
    continue
  fi
  gpu_id=$((expert_idx % PRECOMPUTE_GPUS))
  expert_idx=$((expert_idx + 1))
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    echo "[INFO] Expert precompute task=${TASK} gpu=${CUDA_VISIBLE_DEVICES}"
    "${PYTHON_BIN}" ${PROJECT_ROOT}/scripts/precompute_latents_delta_ee.py \
      --data_dir  "${DATA_ROOT_ORIG}/${TASK}/data" \
      --out_dir   "${TASK_LATENT_DIR}" \
      --svd_path  "${SVD_PATH}" \
      --task_name "${TASK}" \
      --cameras   "${CAMERAS}"
  ) &
  expert_pids+=("$!")
done
wait_for_background_jobs "${expert_pids[@]}"

precompute_enhanced_root() {
  local name="$1"
  local data_root="$2"
  local out_root="$3"

  echo "[INFO] Pre-encoding ${name} latents: ${out_root}"
  enhanced_pids=()
  enhanced_idx=0
  for TASK in ${S1_TASKS}; do
    TASK_LATENT_DIR="${out_root}/${TASK}"
    if [[ -f "${TASK_LATENT_DIR}/meta.json" ]]; then
      echo "[INFO] ${name} latents already exist for ${TASK}, skipping."
      continue
    fi
    gpu_id=$((enhanced_idx % PRECOMPUTE_GPUS))
    enhanced_idx=$((enhanced_idx + 1))
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      echo "[INFO] ${name} precompute task=${TASK} gpu=${CUDA_VISIBLE_DEVICES}"
      "${PYTHON_BIN}" ${PROJECT_ROOT}/scripts/precompute_latents_s1_pca.py \
        --data_root   "${data_root}" \
        --out_root    "${out_root}" \
        --svd_path    "${SVD_PATH}" \
        --tasks       ${S1_TASKS} \
        --task_name   "${TASK}" \
        --episode_min 0 --episode_max 39 \
        --cameras     "${CAMERAS}"
    ) &
    enhanced_pids+=("$!")
  done
  wait_for_background_jobs "${enhanced_pids[@]}"
}

precompute_enhanced_root "pca_c8_sigma0p05" "${DATA_ROOT_PCA}" "${PCA_LATENT_ROOT}"
precompute_enhanced_root "raw_sigma0p0025" "${DATA_ROOT_RAW}" "${RAW_LATENT_ROOT}"
precompute_enhanced_root "random_feasible_uniform" "${DATA_ROOT_RF_UNIFORM}" "${RF_UNIFORM_LATENT_ROOT}"
precompute_enhanced_root "random_feasible_weighted" "${DATA_ROOT_RF_WEIGHTED}" "${RF_WEIGHTED_LATENT_ROOT}"

if [[ "${USE_EE_HEAD:-0}" == "1" ]]; then
  echo "[INFO] Ensuring EE targets exist in latent caches..."
  "${PYTHON_BIN}" ${PROJECT_ROOT}/scripts/backfill_ee_targets.py \
    --latent_root "${EXPERT_LATENT_ROOT}" \
    --data_root "${DATA_ROOT_ORIG}" \
    --tasks ${S1_TASKS} \
    --layout clean
  "${PYTHON_BIN}" ${PROJECT_ROOT}/scripts/backfill_ee_targets.py \
    --latent_root "${PCA_LATENT_ROOT}" \
    --data_root "${DATA_ROOT_PCA}" \
    --tasks ${S1_TASKS} \
    --layout enhanced
  "${PYTHON_BIN}" ${PROJECT_ROOT}/scripts/backfill_ee_targets.py \
    --latent_root "${RAW_LATENT_ROOT}" \
    --data_root "${DATA_ROOT_RAW}" \
    --tasks ${S1_TASKS} \
    --layout enhanced
  "${PYTHON_BIN}" ${PROJECT_ROOT}/scripts/backfill_ee_targets.py \
    --latent_root "${RF_UNIFORM_LATENT_ROOT}" \
    --data_root "${DATA_ROOT_RF_UNIFORM}" \
    --tasks ${S1_TASKS} \
    --layout enhanced
  "${PYTHON_BIN}" ${PROJECT_ROOT}/scripts/backfill_ee_targets.py \
    --latent_root "${RF_WEIGHTED_LATENT_ROOT}" \
    --data_root "${DATA_ROOT_RF_WEIGHTED}" \
    --tasks ${S1_TASKS} \
    --layout enhanced
fi

STAT_PATH="${META_INFO_BASE}/${DATASET_CFGS}/stat.json"
if [[ ! -f "${STAT_PATH}" ]]; then
  echo "[INFO] Computing four-family action stat..."
  "${PYTHON_BIN}" ${PROJECT_ROOT}/scripts/compute_stat_family_roots.py \
    --latent_roots \
      "${EXPERT_LATENT_ROOT}" \
      "${PCA_LATENT_ROOT}" \
      "${RAW_LATENT_ROOT}" \
      "${RF_UNIFORM_LATENT_ROOT}" \
      "${RF_WEIGHTED_LATENT_ROOT}" \
    --tasks ${S1_TASKS} \
    --episode_split 0-39 \
    --out_dir "${META_INFO_BASE}/${DATASET_CFGS}" \
    --action_dim 14
else
  echo "[INFO] Reusing existing stat: ${STAT_PATH}"
fi

FAMILY_ROOT_PATHS="expert=${EXPERT_LATENT_ROOT};pca=${PCA_LATENT_ROOT};raw=${RAW_LATENT_ROOT};rf_uniform=${RF_UNIFORM_LATENT_ROOT};rf_weighted=${RF_WEIGHTED_LATENT_ROOT}"
FAMILY_SAMPLING="expert=0.5,pca=0.166667,raw=0.166667,random_feasible=0.166667"
# Matches the chunk16 static-manifest size from the sampling doc:
# 34214 + 11405 + 11405 + 11405 = 68429 windows.
FAMILY_DATASET_LENGTH=68429

EE_HEAD_ARGS=()
if [[ "${USE_EE_HEAD:-0}" == "1" ]]; then
  EE_LOSS_WEIGHT="${EE_LOSS_WEIGHT:?USE_EE_HEAD=1 requires EE_LOSS_WEIGHT}"
  EE_HEAD_HIDDEN_DIM="${EE_HEAD_HIDDEN_DIM:?USE_EE_HEAD=1 requires EE_HEAD_HIDDEN_DIM}"
  EE_HEAD_ARGS+=(--use_ee_head)
  EE_HEAD_ARGS+=(--ee_loss_weight "${EE_LOSS_WEIGHT}")
  EE_HEAD_ARGS+=(--ee_head_hidden_dim "${EE_HEAD_HIDDEN_DIM}")
fi

printf '%s\n' "$(date): ${RUN_NAME} (3:1:1:1 family-balanced, cameras=${CAMERAS})" > "${OUTPUT_DIR}/launch_cmd.txt"
{
  echo "FAMILY_ROOT_PATHS=${FAMILY_ROOT_PATHS}"
  echo "FAMILY_SAMPLING=${FAMILY_SAMPLING}"
  echo "FAMILY_DATASET_LENGTH=${FAMILY_DATASET_LENGTH}"
  echo "USE_EE_HEAD=${USE_EE_HEAD:-0}"
  if [[ "${USE_EE_HEAD:-0}" == "1" ]]; then
    echo "EE_LOSS_WEIGHT=${EE_LOSS_WEIGHT}"
    echo "EE_HEAD_HIDDEN_DIM=${EE_HEAD_HIDDEN_DIM}"
  fi
} >> "${OUTPUT_DIR}/launch_cmd.txt"

echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] USE_EE_HEAD=${USE_EE_HEAD:-0}"
echo "[INFO] Launching S1-C 3:1:1:1 family-balanced training..."

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port=29619 \
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
  --use_family_balanced_sampler \
  --family_root_paths "${FAMILY_ROOT_PATHS}" \
  --family_sampling "${FAMILY_SAMPLING}" \
  --family_sampling_seed 20260610 \
  --family_dataset_length "${FAMILY_DATASET_LENGTH}" \
  "${EE_HEAD_ARGS[@]}" \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"
