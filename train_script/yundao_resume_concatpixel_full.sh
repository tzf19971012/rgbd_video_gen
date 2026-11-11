#!/usr/bin/env bash
# =============================================================================
# 云道(ROMA)启动入口 — concat@pixel(14) x patch1 续训 + 全量数据
#
# 在 yundao_train_gae_patch1_re10k.sh 的协议上预设了本实验的环境变量：
#   配置        = train_config/normal_config/gae_ovf_re10k_patch1_concatpixel_full.yaml
#   恢复点      = logs/ovf_gae_re10k_patch1_concatpixel/checkpoint-step-2400
#                 （模型+优化器+scheduler+RNG 由 accelerator.load_state 全量恢复，
#                  global_step=2400 由脚本从目录名解析后以 dotlist 覆盖传入）
#   输出目录    = logs/ovf_gae_re10k_patch1_concatpixel_full（固定目录，不追时间戳，
#                 便于与本地实验对齐；要按时间戳新建则 export YUNDAO_OUTPUT_DIR=""）
#   数据        = re10k 全量（num_train_samples=null 已写进 config；
#                 YUNDAO_DATASETS=re10k 与默认一致，不再走 8 场景截断）
#   eval 节奏   = validate_step 200（与被续训 run 一致；脚本默认 300，这里显式回到 200）
#
# 用法（在 ROMA 作业的每个节点上执行）:
#   bash train_script/yundao_resume_concatpixel_full.sh
#
# 生效批量(batch semantics):
#   config 为 batch_size 4 x grad_accum 1。原 run 是 2 卡 -> 有效 batch 8。
#   云道作业请申请 2 NPU 单机（与原 run 对齐）；若申请 N 卡，请追加 dotlist
#   覆盖 batch_size，保持有效 batch=8：bash 本脚本 "batch_size=$((8/N))"
#   （追加参数会透传到 config 之后，最后覆盖生效）。
#
# 其他 YUNDAO_* 变量（seed / 端口 / 节点数等）语义与基底脚本一致，见
# yundao_train_gae_patch1_re10k.sh 头部注释。
# =============================================================================
set -euo pipefail

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${CUR_DIR}/.." && pwd)"

export YUNDAO_RESUME_CKPT="${PROJ_DIR}/logs/ovf_gae_re10k_patch1_concatpixel/checkpoint-step-2400"
export YUNDAO_OUTPUT_DIR="${PROJ_DIR}/logs/ovf_gae_re10k_patch1_concatpixel_full"
export YUNDAO_VALIDATE_STEP=200
export YUNDAO_DATASETS=re10k

exec bash "${CUR_DIR}/yundao_train_gae_patch1_re10k.sh" \
    train_config/normal_config/gae_ovf_re10k_patch1_concatpixel_full.yaml \
    "$@"
