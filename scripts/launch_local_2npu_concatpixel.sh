#!/usr/bin/env bash
# 本机 2 卡(910B3)启动相机注入强化 arm：gae_ovf_re10k_aux0_upsample2_concatpixel
# （concat@pixel Plucker 注入，294ch，其余协议同 aux0_upsample2 anchor）。
# 用法: bash scripts/launch_local_2npu_concatpixel.sh
#   覆盖卡: NPUS=0,1 bash scripts/launch_local_2npu_concatpixel.sh
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/sfs_turbo/t00845730/rgbd_video_gen/.venv/bin/python}"
DA3_SRC="${DA3_SRC:-/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src}"
NPUS="${NPUS:-0,1}"
PORT="${PORT:-29743}"

export ASCEND_RT_VISIBLE_DEVICES="${NPUS}"
export PYTHONPATH="${DA3_SRC}"
export PYTHONUNBUFFERED=1

TS="$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')"
OUT="logs/ovf_gae_re10k_concatpixel_${TS}"
mkdir -p "${OUT}/launch"
echo "${OUT}" > "${PROJ_DIR}/logs/.latest_concatpixel"
echo "[launch] NPUS=${NPUS} OUT=${OUT}"

nohup setsid "${PYTHON_BIN}" -u -m accelerate.commands.launch \
  --multi_gpu --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port "${PORT}" \
  --num_processes 2 \
  -m examples.wanvideo.model_training.train_with_accelerate_gae_nvs \
  --config train_config/normal_config/gae_ovf_re10k_aux0_upsample2_concatpixel.yaml \
  "output_path=${OUT}" \
  > "${OUT}/launch/nohup.log" 2>&1 < /dev/null &
echo "[launch] pid=$!"
