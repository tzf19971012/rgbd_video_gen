#!/usr/bin/env bash
# 实验二：geo/color 双流验证 — 几何 DA3(GAE-64) latent + 颜色 Wan-VAE latent。
# 默认 2 卡(2,3)，与实验一(0,1)并行共用四张空闲卡。
# 前置：wan latent 缓存已建（utils/cache_wan_latents.py -> re10k_ovf8_wan）。
# 用法: bash scripts/launch_local_geocolor_ovf.sh   [NPUS=2,3 PORT=29711 覆盖]
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/sfs_turbo/t00845730/rgbd_video_gen/.venv/bin/python}"
DA3_SRC="${DA3_SRC:-/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src}"
NPUS="${NPUS:-2,3}"
PORT="${PORT:-29711}"

export ASCEND_RT_VISIBLE_DEVICES="${NPUS}"
export PYTHONPATH="${DA3_SRC}"
export PYTHONUNBUFFERED=1

TS="$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')"
OUT="logs/ovf_geocolor_re10k_lora_${TS}"
mkdir -p "${OUT}/launch"
echo "${OUT}" > "${PROJ_DIR}/logs/.latest_geocolor"
echo "[launch] NPUS=${NPUS} OUT=${OUT}"

nohup setsid "${PYTHON_BIN}" -u -m accelerate.commands.launch \
  --multi_gpu --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port "${PORT}" \
  --num_processes 2 \
  -m examples.wanvideo.model_training.train_with_accelerate_geo_color_nvs \
  --config train_config/normal_config/geo_color_ovf_re10k_lora.yaml \
  "output_path=${OUT}" "$@" \
  > "${OUT}/launch/nohup.log" 2>&1 < /dev/null &
echo "[launch] pid=$!"
