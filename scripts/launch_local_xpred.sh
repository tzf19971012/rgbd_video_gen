#!/usr/bin/env bash
# 本机 2 卡启动：patch1 + x-prediction（GAE 原版参数化，单变量对照 patch1）。
# 配置: train_config/normal_config/gae_ovf_re10k_aux0_patch1_xpred.yaml
# 用法: NPUS=4,5 PORT=29713 bash scripts/launch_local_xpred.sh
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/sfs_turbo/t00845730/rgbd_video_gen/.venv/bin/python}"
DA3_SRC="${DA3_SRC:-/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src}"
NPUS="${NPUS:-4,5}"
PORT="${PORT:-29713}"

export ASCEND_RT_VISIBLE_DEVICES="${NPUS}"
export PYTHONPATH="${DA3_SRC}"
export PYTHONUNBUFFERED=1

TS="$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')"
OUT="logs/ovf_gae_re10k_aux0_patch1_xpred_${TS}"
mkdir -p "${OUT}/launch"
echo "${OUT}" > "${PROJ_DIR}/logs/.latest_xpred"
echo "[launch] NPUS=${NPUS} OUT=${OUT}"

nohup setsid "${PYTHON_BIN}" -u -m accelerate.commands.launch \
  --multi_gpu --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port "${PORT}" \
  --num_processes 2 \
  -m examples.wanvideo.model_training.train_with_accelerate_gae_nvs \
  --config train_config/normal_config/gae_ovf_re10k_aux0_patch1_xpred.yaml \
  "output_path=${OUT}" \
  > "${OUT}/launch/nohup.log" 2>&1 < /dev/null &
echo "[launch] pid=$!"
