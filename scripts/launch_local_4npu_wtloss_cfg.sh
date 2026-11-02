#!/usr/bin/env bash
# 本机 4 卡(910B3, NPU 2,3,5,6)启动 gae_re10k_lora_wtloss_cfg。
# 跳过 NPU 7（外部服务）与 NPU 4（残留进程）。bs 8 x 6 = 有效 batch 48，
# 与 shift45 arm（16 x 3）严格一致。
# 用法: bash scripts/launch_local_6npu_wtloss_cfg.sh
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/sfs_turbo/t00845730/rgbd_video_gen/.venv/bin/python}"
DA3_SRC="${DA3_SRC:-/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src}"
NPUS="${NPUS:-2,3,5,6}"
PORT="${PORT:-29733}"

export ASCEND_RT_VISIBLE_DEVICES="${NPUS}"
export PYTHONPATH="${DA3_SRC}"
export PYTHONUNBUFFERED=1

TS="$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')"
OUT="logs/gae_re10k_wtloss_cfg_${TS}"
mkdir -p "${OUT}/launch"
echo "${OUT}" > "${PROJ_DIR}/logs/.latest_wtloss_cfg"
echo "[launch] NPUS=${NPUS} OUT=${OUT}"

nohup setsid "${PYTHON_BIN}" -u -m accelerate.commands.launch \
  --multi_gpu --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port "${PORT}" \
  --num_processes 4 \
  -m examples.wanvideo.model_training.train_with_accelerate_gae_nvs \
  --config train_config/normal_config/gae_re10k_lora_wtloss_cfg.yaml \
  "output_path=${OUT}" \
  > "${OUT}/launch/nohup.log" 2>&1 < /dev/null &
echo "[launch] pid=$!"
