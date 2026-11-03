#!/bin/bash
# CVUSA 5-view street -> satellite RGB-D NVS LoRA finetune.
# 基于 multi4_nvs_lora step-4500 权重，在 CVUSA 5视角->卫星 数据上微调。
#
# 用法:
#   bash train_script/train_cvusa5_sat_lora.sh
#   bash train_script/train_cvusa5_sat_lora.sh train_config/normal_config/cvusa5_sat_lora.yaml [key=value ...]
#
# NPU 选择: export ASCEND_RT_VISIBLE_DEVICES=<id>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${1:-train_config/normal_config/cvusa5_sat_lora.yaml}"
shift || true

PYTHON_BIN="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

echo "[train_cvusa5_sat_lora] PROJECT_ROOT = ${PROJECT_ROOT}"
echo "[train_cvusa5_sat_lora] config         = ${CONFIG}"
echo "[train_cvusa5_sat_lora] python         = ${PYTHON_BIN}"
echo "[train_cvusa5_sat_lora] overrides      = $*"

if [ "${USE_ACCELERATE:-0}" = "1" ]; then
    ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-train_config/accelerate_config/accelerate_npu_1gpu.yaml}"
    "${PYTHON_BIN}" -u -m accelerate.commands.launch --config_file "${ACCELERATE_CONFIG}" \
        -m examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs \
        --config "${CONFIG}" "$@"
else
    "${PYTHON_BIN}" -u -m examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs \
        --config "${CONFIG}" "$@"
fi
