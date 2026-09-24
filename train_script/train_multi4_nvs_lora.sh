#!/bin/bash
# 多数据集（BlendedMVS / GTA-SfM / MatrixCity / TartanAir）RGB-D NVS
# LoRA 微调（基于 bmvs_c_base.yaml）启动脚本。
#
# 用法:
#   bash train_script/train_multi4_nvs_lora.sh
#   bash train_script/train_multi4_nvs_lora.sh <config.yaml> [key=value ...]
#
# 例子（临时覆盖超参，无需改 yaml）:
#   bash train_script/train_multi4_nvs_lora.sh \
#       train_config/normal_config/multi4_nvs_lora.yaml \
#       batch_size=4 log_step=10 output_path=logs/_demo_lora
#
# 切换平台时：改 yaml 顶部的路径段（wan_model_root / sample_data_root / output_path）。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${1:-train_config/normal_config/multi4_nvs_lora.yaml}"
shift || true

# 本机 .venv 没有 accelerate 入口点（--system-site-packages 最小 venv），
# 与本机正在运行的作业一致：直接用 venv python 跑模块（单进程）；
# 需要 accelerate 多进程时: USE_ACCELERATE=1 ACCELERATE_CONFIG=... bash 本脚本
PYTHON_BIN="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

echo "[train_multi4_nvs_lora] PROJECT_ROOT = ${PROJECT_ROOT}"
echo "[train_multi4_nvs_lora] config         = ${CONFIG}"
echo "[train_multi4_nvs_lora] python         = ${PYTHON_BIN}"
echo "[train_multi4_nvs_lora] overrides      = $*"

if [ "${USE_ACCELERATE:-0}" = "1" ]; then
    ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-train_config/accelerate_config/accelerate_npu_1gpu.yaml}"
    echo "[train_multi4_nvs_lora] accelerate     = ${ACCELERATE_CONFIG}"
    accelerate launch --config_file "${ACCELERATE_CONFIG}" \
        -m examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs \
        --config "${CONFIG}" "$@"
else
    "${PYTHON_BIN}" -u -m examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs \
        --config "${CONFIG}" "$@"
fi
