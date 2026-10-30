#!/bin/bash
# 多数据集（BlendedMVS / GTA-SfM / MatrixCity / TartanAir，manifest.jsonl 驱动）
# 视频深度估计 —— 全量微调（full fine-tune，非 LoRA）启动脚本。
#
# 用法:
#   bash train_script/train_video_multi_fullft.sh                          # 默认配置
#   bash train_script/train_video_multi_fullft.sh <config.yaml>            # 指定训练配置
#   bash train_script/train_video_multi_fullft.sh <config.yaml> <accelerate.yaml>
#
# 切换平台时：优先改 train_config/normal_config/video_config_multi_fullft.yaml
# 顶部的路径段（wan_model_root / sample_data_root / output_path）。
#
# 注意：全量微调显存占用约为 LoRA 的数倍（AdamW 状态 + 全量梯度），
# 1.3B DiT + batch_size=1 + 梯度检查点下每卡约需 21GB+，请确认单卡显存足够。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${1:-train_config/normal_config/video_config_multi_fullft.yaml}"
ACCELERATE_CONFIG="${2:-train_config/accelerate_config/accelerate_acc4.yaml}"

echo "[train_video_multi_fullft] PROJECT_ROOT = ${PROJECT_ROOT}"
echo "[train_video_multi_fullft] config         = ${CONFIG}"
echo "[train_video_multi_fullft] accelerate     = ${ACCELERATE_CONFIG}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" \
    -m examples.wanvideo.model_training.train_with_accelerate_video \
    --config "${CONFIG}"
