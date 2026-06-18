#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ctrlworld_train_env.sh"

# ============================================================================
# CtrlWorld delta-ee, all 50 tasks, 14D abs-joint action, nf=16, 60k steps.
# Camera set: head_camera + left_camera + right_camera (head + dual wrist),
# stacked vertically top->bottom as head / left / right -> latent (T,4,90,40).
# Original (clean) data ONLY -- no enhanced/perturbed augmentation this run.
# Derived from train_ctrlworld_8gpu_delta_ee_all50_nf16_60k.sh; only the camera
# set, latent paths and run name differ -- all hyper-params kept identical.
# ============================================================================

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# load runtime envs (wandb/hf/cuda/proxy)
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
RUN_NAME="ctrlworld_delta_ee_all50_8gpu_nf16_60k_headwrist_${RUN_TS}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

export WANDB_MODE="online"
export WANDB_PROJECT="ctrlworld_delta_ee"
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

# ---- camera set & data paths ----
CAMERAS="head_camera,left_camera,right_camera"   # head + dual wrist, top->bottom
SVD_PATH="${ASSET_ROOT}/stable-video-diffusion-img2vid"
LATENT_ROOT_ORIG="${CACHE_ROOT}/precomputed_latents_delta_ee_all50_14d_headwrist"
DATA_ROOT_ORIG="/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/data_delta_ee/demo_clean_zed2i_visible"

ALL50_TASKS=(
  adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
  click_alarmclock click_bell dump_bin_bigbin grab_roller
  handover_block handover_mic hanging_mug lift_pot
  move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad
  open_laptop open_microwave pick_diverse_bottles pick_dual_bottles
  place_a2b_left place_a2b_right place_bread_basket place_bread_skillet
  place_burger_fries place_can_basket place_cans_plasticbox place_container_plate
  place_dual_shoes place_empty_cup place_fan place_mouse_pad
  place_object_basket place_object_scale place_object_stand place_phone_stand
  place_shoe press_stapler put_bottles_dustbin put_object_cabinet
  rotate_qrcode scan_object shake_bottle shake_bottle_horizontally
  stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two
  stamp_seal turn_switch
)

echo "[INFO] Step 1: Pre-encoding original latents with cameras=${CAMERAS} (skip if exist)..."
for TASK in "${ALL50_TASKS[@]}"; do
  TASK_LATENT_DIR="${LATENT_ROOT_ORIG}/${TASK}"
  if [[ -f "${TASK_LATENT_DIR}/meta.json" ]]; then
    echo "[INFO] Orig latents already exist for ${TASK}, skipping."
    continue
  fi
  echo "[INFO] Pre-encoding orig latents for ${TASK}..."
  "${PYTHON_BIN}" ${PROJECT_ROOT}/scripts/precompute_latents_delta_ee.py \
    --data_dir  "${DATA_ROOT_ORIG}/${TASK}/data" \
    --out_dir   "${TASK_LATENT_DIR}" \
    --svd_path  "${SVD_PATH}" \
    --task_name "${TASK}" \
    --cameras   "${CAMERAS}"
done

# ---- action stat: reuse original-only 14D stat (action is camera-independent) ----
DATASET_CFGS="delta_ee_all50_14d"
STAT_PATH="${PROJECT_ROOT}/dataset_meta_info/${DATASET_CFGS}/stat.json"
if [[ ! -f "${STAT_PATH}" ]]; then
  echo "[ERROR] Expected reusable stat not found at ${STAT_PATH}"
  exit 1
fi
echo "[INFO] Reusing original-only action stat: ${STAT_PATH}"

# ---- 8-GPU training (all 50 tasks, 14D, nf=16, original-only, head+dual-wrist, 60k steps) ----
DATASET_NAMES="adjust_bottle+beat_block_hammer+blocks_ranking_rgb+blocks_ranking_size+click_alarmclock+click_bell+dump_bin_bigbin+grab_roller+handover_block+handover_mic+hanging_mug+lift_pot+move_can_pot+move_pillbottle_pad+move_playingcard_away+move_stapler_pad+open_laptop+open_microwave+pick_diverse_bottles+pick_dual_bottles+place_a2b_left+place_a2b_right+place_bread_basket+place_bread_skillet+place_burger_fries+place_can_basket+place_cans_plasticbox+place_container_plate+place_dual_shoes+place_empty_cup+place_fan+place_mouse_pad+place_object_basket+place_object_scale+place_object_stand+place_phone_stand+place_shoe+press_stapler+put_bottles_dustbin+put_object_cabinet+rotate_qrcode+scan_object+shake_bottle+shake_bottle_horizontally+stack_blocks_three+stack_blocks_two+stack_bowls_three+stack_bowls_two+stamp_seal+turn_switch"
DATASET_ROOT="${LATENT_ROOT_ORIG}"

printf '%s\n' "$(date): ${RUN_NAME} (cameras=${CAMERAS}, original-only)" > "${OUTPUT_DIR}/launch_cmd.txt"

echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] CAMERAS=${CAMERAS}"
echo "[INFO] Launching 8-GPU training (delta-ee 14D, all50 tasks, nf=16, head+dual-wrist, original-only, 60k steps)..."

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port=29615 \
  ${PROJECT_ROOT}/scripts/train_delta_ee.py \
  --svd_model_path    "${SVD_PATH}" \
  --clip_model_path   ${ASSET_ROOT}/clip-vit-base-patch32 \
  --ckpt_path         none \
  --dataset_root_path "${DATASET_ROOT}" \
  --dataset_meta_info_path ${PROJECT_ROOT}/dataset_meta_info \
  --dataset_names     "${DATASET_NAMES}" \
  --dataset_cfgs      "${DATASET_CFGS}" \
  --output_dir        "${OUTPUT_DIR}" \
  --wandb_project_name ctrlworld_delta_ee \
  --wandb_run_name    "${RUN_NAME}" \
  --tag               "${RUN_NAME}" \
  --action_dim        14 \
  --height            240 \
  --num_history       6 \
  --num_frames        16 \
  --train_batch_size  1 \
  --gradient_accumulation_steps 2 \
  --max_train_steps   60000 \
  --checkpointing_steps 2500 \
  --validation_steps  5000 \
  --learning_rate     1e-5 \
  --mixed_precision   bf16 \
  --use_abs_joint_action \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"
