#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ctrlworld_train_env.sh"

echo "[INFO] Using PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] Installing Ctrl-World training dependencies..."

"${PYTHON_BIN}" -m pip install -U pip setuptools wheel

install_pkg() {
  local pkg="$1"
  "${PYTHON_BIN}" -m pip install "${pkg}" || \
    "${PYTHON_BIN}" -m pip install -i https://pypi.org/simple "${pkg}"
}

for pkg in \
  "transformers==4.48.1" \
  "accelerate>=0.34.0" \
  "peft==0.15.2" \
  "safetensors==0.5.3" \
  sentencepiece \
  einops \
  decord \
  imageio \
  mediapy \
  omegaconf \
  wandb \
  h5py; do
  install_pkg "${pkg}"
done

"${PYTHON_BIN}" -m pip install -i https://pypi.org/simple "diffusers==0.34.0"
"${PYTHON_BIN}" -m pip uninstall -y swanlab || true

"${PYTHON_BIN}" - <<'PY'
import diffusers, transformers, accelerate, torch, h5py
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import TextToVideoSDPipelineOutput
print("deps_ok", "diffusers", diffusers.__version__, "torch", torch.__version__)
PY

echo "[INFO] Done."
