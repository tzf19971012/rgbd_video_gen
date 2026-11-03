#!/usr/bin/env bash
# 单卡(910B3)续训 patch1：从 checkpoint-step-5600 恢复，bs1 x accum8 = 有效 batch 8
# （与原 bs2x2 卡 x accum2 等价；换 1 卡后显存装不下 bs2）。
# 用法: NPUS=3 bash scripts/launch_local_1npu_patch1_resume.sh
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/sfs_turbo/t00845730/rgbd_video_gen/.venv/bin/python}"
DA3_SRC="${DA3_SRC:-/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src}"
NPUS="${NPUS:-3}"
PORT="${PORT:-29754}"
CKPT="${CKPT:-logs/ovf_gae_re10k_aux0_patch1/checkpoint-step-5600}"
STEP="${STEP:-5600}"

export ASCEND_RT_VISIBLE_DEVICES="${NPUS}"
export PYTHONPATH="${DA3_SRC}"
export PYTHONUNBUFFERED=1

OUT="logs/ovf_gae_re10k_aux0_patch1"   # 续训写回同一目录（train.log append）
echo "[launch] NPUS=${NPUS} OUT=${OUT} resume_from=${CKPT}@step${STEP}"

nohup setsid "${PYTHON_BIN}" -u -m accelerate.commands.launch \
  --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port "${PORT}" \
  --num_processes 1 \
  -m examples.wanvideo.model_training.train_with_accelerate_gae_nvs \
  --config train_config/normal_config/gae_ovf_re10k_aux0_patch1.yaml \
  "output_path=${OUT}" \
  "gradient_accumulation_steps=8" \
  "batch_size=1" \
  "resume=true" \
  "training_state_dir=${CKPT}" \
  "global_step=${STEP}" \
  >> "${OUT}/launch/nohup_1npu_resume.log" 2>&1 < /dev/null &
echo "[launch] pid=$!"
