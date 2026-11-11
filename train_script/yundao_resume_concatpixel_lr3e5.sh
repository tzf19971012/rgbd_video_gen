#!/usr/bin/env bash
set -euo pipefail
CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${CUR_DIR}/.." && pwd)"
export YUNDAO_RESUME_CKPT="${PROJ_DIR}/logs/gae_patch1_re10k20261104_202722/checkpoint-step-8000"
export YUNDAO_OUTPUT_DIR="${PROJ_DIR}/logs/gae_patch1_re10k_lr3e5_from8000"
export YUNDAO_VALIDATE_STEP=200
export YUNDAO_DATASETS=re10k
exec bash "${CUR_DIR}/yundao_train_gae_patch1_re10k.sh" \
    train_config/normal_config/gae_re10k_concatpixel_resume_lr3e5.yaml "$@"
