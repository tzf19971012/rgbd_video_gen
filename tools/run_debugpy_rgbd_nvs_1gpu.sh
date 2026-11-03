#!/bin/bash

set -euo pipefail

PROJECT_DIR="${DVD_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${PROJECT_DIR}/.venv"
DEBUGPY_HOST="${DEBUGPY_HOST:-0.0.0.0}"
DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"
TRAIN_CONFIG="${1:-train_config/normal_config/blendedmvs_nvs_independent_quicktest_bs2_evalcheck.yaml}"

source ~/.bashrc
cd "${PROJECT_DIR}"

if [[ ! -d "${VENV}" ]]; then
  echo "ERROR: virtualenv not found: ${VENV}"
  exit 1
fi

source "${VENV}/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="${ZERO2CAD_LIBS:-$HOME/zero2cad/libs}":/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export PYGLET_HEADLESS=True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== debugpy launch start: $(date) ==="
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "DEBUGPY listen: ${DEBUGPY_HOST}:${DEBUGPY_PORT}"

nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

python -m debugpy \
  --listen "${DEBUGPY_HOST}:${DEBUGPY_PORT}" \
  --wait-for-client \
  -m examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs \
  --config "${TRAIN_CONFIG}"