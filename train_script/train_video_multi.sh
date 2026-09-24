#!/bin/bash
# 多数据集（BlendedMVS / GTA-SfM / MatrixCity / TartanAir，manifest.jsonl 驱动）
# 视频深度训练启动脚本。
#
# 用法:
#   bash train_script/train_video_multi.sh                          # 默认配置
#   bash train_script/train_video_multi.sh <config.yaml>            # 指定训练配置
#   bash train_script/train_video_multi.sh <config.yaml> <accelerate.yaml>  # 两者都指定
#
# 切换平台时：优先改 train_config/normal_config/video_config_multi.yaml 顶部的
# 路径段（wan_model_root / sample_data_root / output_path），
# 也可以通过第一个参数传入另一份配置，无需改动本脚本。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${1:-train_config/normal_config/video_config_multi.yaml}"
ACCELERATE_CONFIG="${2:-train_config/accelerate_config/accelerate_acc4.yaml}"

echo "[train_video_multi] PROJECT_ROOT = ${PROJECT_ROOT}"
echo "[train_video_multi] config         = ${CONFIG}"
echo "[train_video_multi] accelerate     = ${ACCELERATE_CONFIG}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" \
    -m examples.wanvideo.model_training.train_with_accelerate_video \
    --config "${CONFIG}"
