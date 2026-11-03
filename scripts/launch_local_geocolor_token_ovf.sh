#!/usr/bin/env bash
# 实验五：geo/color 双流 v3 — token 级合并（分离 patchify）。
# geo 流用自有 Conv3d(64,1536) 在原生网格 28x48 patchify，color 流用自有
# Conv3d(16,1536)（pretrained Wan 种子）在 48x84 patchify，token 按帧拼接
# (336+1008=1344 tok/帧) 进原封 DiT，两速度头各读各的 token 段。
# 协议与 v1（ovf_geocolor_re10k_lora_20261102_104836）逐位一致，可配对比较。
# 前置：wan latent 缓存已建（utils/cache_wan_latents.py -> re10k_ovf8_wan）。
# 用法: bash scripts/launch_local_geocolor_token_ovf.sh
#   覆盖卡: NPUS=4,5 PORT=29735 bash scripts/launch_local_geocolor_token_ovf.sh
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/sfs_turbo/t00845730/rgbd_video_gen/.venv/bin/python}"
DA3_SRC="${DA3_SRC:-/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src}"
NPUS="${NPUS:-5,6}"
PORT="${PORT:-29733}"

export ASCEND_RT_VISIBLE_DEVICES="${NPUS}"
export PYTHONPATH="${DA3_SRC}"
export PYTHONUNBUFFERED=1

TS="$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')"
OUT="logs/ovf_geocolor_token_re10k_lora_${TS}"
mkdir -p "${OUT}/launch"
echo "${OUT}" > "${PROJ_DIR}/logs/.latest_geocolor_token"
echo "[launch] NPUS=${NPUS} OUT=${OUT}"

nohup setsid "${PYTHON_BIN}" -u -m accelerate.commands.launch \
  --multi_gpu --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port "${PORT}" \
  --num_processes 2 \
  -m examples.wanvideo.model_training.train_with_accelerate_geo_color_token_nvs \
  --config train_config/normal_config/geo_color_token_ovf_re10k_lora.yaml \
  "output_path=${OUT}" "$@" \
  > "${OUT}/launch/nohup.log" 2>&1 < /dev/null &
echo "[launch] pid=$!"
