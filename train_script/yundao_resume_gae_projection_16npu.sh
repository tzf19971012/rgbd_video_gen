#!/usr/bin/env bash
# ROMA/Yundao:2 nodes x8 NPUs, GAE projection, all RE10K TRAIN scenes.
# Resume model+optimizer+scheduler at step8600; stop at global step30000.
# BS8/rank x16 ranks x accumulation1 = effective batch128; LR3e-5 unchanged.
# World size changes3->16: rank RNGs are reseeded; not a bitwise-exact replay.
# On EVERY node: bash train_script/yundao_resume_gae_projection_16npu.sh
# Check without training: ... --dry-run (requires ROMA vars or simulated vars).
# OOM fallback: ... batch_size=4 (global64) or batch_size=2 (global32).
set -euo pipefail
PROJ_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd -- "${PROJ_SCRIPT_DIR}/.." && pwd)"
export YUNDAO_NUM_HOSTS="${YUNDAO_NUM_HOSTS:-2}"
export YUNDAO_NUM_GPUS_PER_NODE="${YUNDAO_NUM_GPUS_PER_NODE:-8}"
if [[ ! "${YUNDAO_NUM_HOSTS}" =~ ^[1-9][0-9]*$ || ! "${YUNDAO_NUM_GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || (( YUNDAO_NUM_HOSTS * YUNDAO_NUM_GPUS_PER_NODE != 16 )); then
  echo 'This entry requires16 total ranks (default2 nodes x8 NPUs).' >&2; exit 2
fi
export YUNDAO_RESUME_CKPT="${YUNDAO_RESUME_CKPT:-${PROJ_ROOT}/reports/projection_gae_yundao_16npu_from8600/checkpoint-step-8600}"
export YUNDAO_OUTPUT_DIR="${YUNDAO_OUTPUT_DIR:-${PROJ_ROOT}/logs/projection_gae_yundao_16npu_from8600_to30000}"
export YUNDAO_VALIDATE_STEP=200
export YUNDAO_DATASETS=re10k
export YUNDAO_RE10K_ROOT=/mnt/sfs_turbo/t00845730/re10k/extracted
export YUNDAO_RE10K_MANIFEST=/mnt/sfs_turbo/t00845730/re10k/manifest_train_c2w_v1.jsonl
export YUNDAO_TRAIN_SCRIPT="${PROJ_ROOT}/reports/projection_ablation_20261109_113836/resume_3card_20261110_110818/code/train.py"
export YUNDAO_MASTER_PORT="${YUNDAO_MASTER_PORT:-29652}"
if [[ "${1:-}" == '--dry-run' ]]; then export YUNDAO_DRY_RUN=1; shift; fi
exec bash "${PROJ_SCRIPT_DIR}/yundao_train_gae_projection.sh" \
  train_config/normal_config/gae_projection_re10k_yundao_16npu_from8600.yaml "$@"
