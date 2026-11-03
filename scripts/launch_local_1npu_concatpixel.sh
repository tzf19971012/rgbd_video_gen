#!/usr/bin/env bash
# 单卡(910B3)启动 concatpixel arm（concat@pixel Plucker 注入，294ch）。
# bs1 x accum8 = 有效 batch 8（与 anchor bs2x2 卡 x accum2 等价；单卡显存装不下 bs2）。
# 用法: NPUS=7 bash scripts/launch_local_1npu_concatpixel.sh
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/sfs_turbo/t00845730/rgbd_video_gen/.venv/bin/python}"
DA3_SRC="${DA3_SRC:-/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src}"
NPUS="${NPUS:-7}"
PORT="${PORT:-29763}"

export ASCEND_RT_VISIBLE_DEVICES="${NPUS}"
export PYTHONPATH="${DA3_SRC}"
export PYTHONUNBUFFERED=1

TS="$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')"
OUT="logs/ovf_gae_re10k_concatpixel_${TS}"
mkdir -p "${OUT}/launch"
echo "${OUT}" > "${PROJ_DIR}/logs/.latest_concatpixel"
echo "[launch] NPUS=${NPUS} OUT=${OUT}"

nohup setsid "${PYTHON_BIN}" -u -m accelerate.commands.launch \
  --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port "${PORT}" \
  --num_processes 1 \
  -m examples.wanvideo.model_training.train_with_accelerate_gae_nvs \
  --config train_config/normal_config/gae_ovf_re10k_aux0_upsample2_concatpixel.yaml \
  "output_path=${OUT}" \
  "gradient_accumulation_steps=8" \
  "batch_size=1" \
  >> "${OUT}/launch/nohup.log" 2>&1 < /dev/null &
echo "[launch] pid=$!"
