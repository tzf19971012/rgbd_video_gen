#!/usr/bin/env bash
# 多机（ROMA 平台）启动：四数据集 RGB-D NVS LoRA 微调。
# 改编自 ORA-MoT 的 train_robot_sequence_roma.sh，流程一致：
#   检查平台注入的分布式参数 -> 激活 Python 环境 -> 跨节点同步实验时间戳
#   -> 创建实验目录并快照脚本 -> accelerate 多机启动训练。
#
# 用法（在 ROMA 作业里每个节点执行同一入口）:
#   bash train_script/yundao_train_multi4_nvs_lora.sh
#   bash train_script/yundao_train_multi4_nvs_lora.sh [config.yaml] [key=value ...]
#
# 例子（临时覆盖超参，无需改 yaml）:
#   bash train_script/yundao_train_multi4_nvs_lora.sh \
#       train_config/normal_config/multi4_nvs_lora.yaml \
#       batch_size=4 log_step=10
#
# 常用环境变量（都有默认值，一般不用设置）:
#   YUNDAO_CONDA_ENV        集群上的 Conda 环境名；设置后用该环境的 python。
#                           不设置则用 YUNDAO_PYTHON_BIN（默认项目 .venv）。
#   YUNDAO_PYTHON_BIN       Python 解释器路径；默认 .venv/bin/python（存在时）。
#   YUNDAO_RUN_PREFIX       实验目录名前缀，默认 multi4_nvs_lora。
#   YUNDAO_OUTPUT_DIR       精确指定输出目录（绝对路径），用于恢复已有实验；
#                           此时不再追加时间戳。
#   YUNDAO_LOG_ROOT         实验输出根目录，默认 <项目>/logs。
#   YUNDAO_MASTER_ADDR/PORT 覆盖主节点地址/端口；端口默认 29600。
#   YUNDAO_TIMEZONE         时间戳时区，默认 Asia/Shanghai。
#   YUNDAO_RUN_TIMESTAMP    固定时间戳（YYYYMMDD_HHMMSS），所有节点需一致。
#   YUNDAO_NO_LOGIN_SHELL=1 不经过 sudo 登录 shell 直接执行（本机调试用）。

set -euo pipefail

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${CUR_DIR}/.." && pwd)"

# =================================================================
# 1. 检查平台注入的分布式参数。每个节点都要执行这个入口。
# =================================================================
for roma_var in VC_WORKER_HOSTS MA_NUM_HOSTS MA_NUM_GPUS VC_TASK_INDEX; do
  if [[ -z "${!roma_var:-}" ]]; then
    echo "Missing ROMA environment variable: ${roma_var}" >&2
    echo "Run this script in a ROMA job on every node." >&2
    echo "For local training: bash train_script/train_multi4_nvs_lora.sh" >&2
    exit 1
  fi
done
[[ "${MA_NUM_HOSTS}" =~ ^[1-9][0-9]*$ && "${MA_NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "MA_NUM_HOSTS and MA_NUM_GPUS must be positive integers" >&2
  exit 1
}
[[ "${VC_TASK_INDEX}" =~ ^[0-9]+$ ]] && (( VC_TASK_INDEX < MA_NUM_HOSTS )) || {
  echo "VC_TASK_INDEX must be an integer in [0, MA_NUM_HOSTS)" >&2
  exit 1
}
MASTER_HOST="${VC_WORKER_HOSTS}"
MASTER_ADDR="${YUNDAO_MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}"
MASTER_PORT="${YUNDAO_MASTER_PORT:-29600}"
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] && (( MASTER_PORT > 0 && MASTER_PORT < 65536 )) || {
  echo "Invalid master port: ${MASTER_PORT}" >&2; exit 1;
}
WORLD_SIZE="$((MA_NUM_HOSTS * MA_NUM_GPUS))"
# 时间戳同步使用独立端口（训练端口 + 1），避免和 accelerate 的连接互相干扰。
# 仅多节点时才会真正用到；单节点直接读本机时钟。
TIMESTAMP_PORT="${YUNDAO_TIMESTAMP_PORT:-$((MASTER_PORT + 1))}"
if (( MA_NUM_HOSTS > 1 )); then
  [[ "${TIMESTAMP_PORT}" =~ ^[0-9]+$ ]] && (( TIMESTAMP_PORT > 0 && TIMESTAMP_PORT < 65536 && TIMESTAMP_PORT != MASTER_PORT )) || {
    echo "Invalid timestamp port (must differ from MASTER_PORT): ${TIMESTAMP_PORT}" >&2
    exit 1
  }
fi
echo "NNODES: ${MA_NUM_HOSTS}"
echo "MASTER_HOST: ${MASTER_HOST}"
echo "NODE_RANK: ${VC_TASK_INDEX}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "NGPUS_PER_NODE: ${MA_NUM_GPUS}"
echo "NUM_PROCESSES: ${WORLD_SIZE}"

# =================================================================
# 2. 读取训练参数。输出目录等时间戳同步后再确定。
# =================================================================
RUN_PREFIX="${YUNDAO_RUN_PREFIX:-multi4_nvs_lora}"
[[ "${RUN_PREFIX}" == */* ]] && {
  echo "YUNDAO_RUN_PREFIX must be a folder-name prefix; use YUNDAO_OUTPUT_DIR for a path." >&2
  exit 1
}
RUN_TIMEZONE="${YUNDAO_TIMEZONE:-Asia/Shanghai}"
RUN_TIMESTAMP="${YUNDAO_RUN_TIMESTAMP:-}"
[[ -z "${RUN_TIMESTAMP}" || "${RUN_TIMESTAMP}" =~ ^[0-9]{8}_[0-9]{6}$ ]] || {
  echo "YUNDAO_RUN_TIMESTAMP must have format YYYYMMDD_HHMMSS: ${RUN_TIMESTAMP}" >&2
  exit 1
}
OUTPUT_DIR_OVERRIDE="${YUNDAO_OUTPUT_DIR:-}"
if [[ -n "${OUTPUT_DIR_OVERRIDE}" && "${OUTPUT_DIR_OVERRIDE}" != /* ]]; then
  OUTPUT_DIR_OVERRIDE="${PROJ_DIR}/${OUTPUT_DIR_OVERRIDE}"
fi
LOG_ROOT="${YUNDAO_LOG_ROOT:-${PROJ_DIR}/logs}"
[[ "${LOG_ROOT}" == /* ]] || LOG_ROOT="${PROJ_DIR}/${LOG_ROOT}"

CONFIG="${1:-train_config/normal_config/multi4_nvs_lora.yaml}"
if [[ "${CONFIG}" != /* ]]; then CONFIG="${PROJ_DIR}/${CONFIG}"; fi
test -s "${CONFIG}" || { echo "Missing training config: ${CONFIG}" >&2; exit 1; }
shift || true
EXTRA_ARGS=("$@")

# Python 环境：指定 Conda 环境则激活后用其 python；否则用项目 .venv（存在时）。
CONDA_ENV="${YUNDAO_CONDA_ENV:-}"
if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_BIN="python"
elif [[ -x "${PROJ_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJ_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

MULTI_GPU_FLAG=""
if (( WORLD_SIZE > 1 )); then MULTI_GPU_FLAG="--multi_gpu"; fi

# 尚未确定最终实验目录，先单独记录环境 / 时间戳同步阶段的日志。
# 即使环境激活或同步失败，也能从这里找到错误；成功后复制到实验目录。
BOOTSTRAP_DIR="${PROJ_DIR}/logs/.yundao_launch_logs"
mkdir -p "${BOOTSTRAP_DIR}"
BOOTSTRAP_LOG="${BOOTSTRAP_DIR}/node_${VC_TASK_INDEX}_$(TZ="${RUN_TIMEZONE}" date '+%Y%m%d_%H%M%S')_$$.log"
echo "启动阶段日志: ${BOOTSTRAP_LOG}"

# =================================================================
# 3. 准备登录 shell。显式传入外层已算好的参数，避免 sudo 清掉环境变量。
# =================================================================
source "${CUR_DIR}/roma_environment.sh"
roma_capture_environment

# printf 对空数组会输出一个空串参数，因此先判空再拼接。
EXTRA_ARGS_Q=""
if (( ${#EXTRA_ARGS[@]} > 0 )); then printf -v EXTRA_ARGS_Q '%q ' "${EXTRA_ARGS[@]}"; fi
printf -v PROJ_DIR_Q '%q' "${PROJ_DIR}"
printf -v CUR_DIR_Q '%q' "${CUR_DIR}"
printf -v CONFIG_Q '%q' "${CONFIG}"
printf -v CONDA_ENV_Q '%q' "${CONDA_ENV}"
printf -v PYTHON_BIN_Q '%q' "${PYTHON_BIN}"
printf -v MASTER_ADDR_Q '%q' "${MASTER_ADDR}"
printf -v BOOTSTRAP_LOG_Q '%q' "${BOOTSTRAP_LOG}"
printf -v LOG_ROOT_Q '%q' "${LOG_ROOT}"
printf -v OUTPUT_DIR_OVERRIDE_Q '%q' "${OUTPUT_DIR_OVERRIDE}"
printf -v RUN_PREFIX_Q '%q' "${RUN_PREFIX}"
printf -v RUN_TIMESTAMP_Q '%q' "${RUN_TIMESTAMP}"
printf -v RUN_TIMEZONE_Q '%q' "${RUN_TIMEZONE}"
printf -v TIMESTAMP_TIMEOUT_Q '%q' "${YUNDAO_RENDEZVOUS_TIMEOUT:-600}"
printf -v TIMESTAMP_PORT_Q '%q' "${TIMESTAMP_PORT}"
printf -v SCRIPT_PATH_Q '%q' "${CUR_DIR}/${BASH_SOURCE[0]##*/}"
printf -v ENV_HELPER_Q '%q' "${CUR_DIR}/roma_environment.sh"
printf -v TIMESTAMP_HELPER_Q '%q' "${CUR_DIR}/sequence_run_timestamp.py"
ENV_ASSIGNMENTS_Q=""
if (( ${#ROMA_ENV_ASSIGNMENTS[@]} > 0 || ${#ROMA_PATH_ASSIGNMENTS[@]} > 0 )); then
  printf -v ENV_ASSIGNMENTS_Q '%q ' "${ROMA_ENV_ASSIGNMENTS[@]+"${ROMA_ENV_ASSIGNMENTS[@]}"}" \
                                    "${ROMA_PATH_ASSIGNMENTS[@]+"${ROMA_PATH_ASSIGNMENTS[@]}"}"
fi

# sudo 会清理当前环境；在外层把路径转义，再写入登录 shell 的命令。
# 不要改成 sudo -i bash -i -c：sudo -i 会再次转义参数，破坏多行命令。
# 已经是 root 或没有 sudo 的容器可直接走 bash 分支；YUNDAO_NO_LOGIN_SHELL=1 强制直跑。
if [[ "${YUNDAO_NO_LOGIN_SHELL:-0}" == "1" ]] || ! command -v sudo >/dev/null || [[ "$(id -u)" == "0" ]]; then
  LOGIN_SHELL=(bash -l -i -c)
else
  LOGIN_SHELL=(sudo -H bash -l -i -c)
fi
"${LOGIN_SHELL[@]}" "
  # 保存平台 stdout 到 fd 3；换日志文件后仍直接显示到平台控制台。
  exec 3>&1
  exec > >(tee -a ${BOOTSTRAP_LOG_Q} >&3) 2>&1
  # Conda 的激活脚本可能读取未设置的可选变量，不在这里启用 nounset。
  set +u
  set -eo pipefail
  YUNDAO_STAGE=environment
  trap 'rc=\$?; echo \"[YUNDAO ERROR] stage=\${YUNDAO_STAGE} exit=\${rc}\" >&2' ERR
  echo '[YUNDAO START] node=${VC_TASK_INDEX}/${MA_NUM_HOSTS}, processes=${WORLD_SIZE}'
  date -u
  source ${ENV_HELPER_Q}
  roma_log_environment login_before_restore
  roma_restore_environment --platform-paths-first ${ENV_ASSIGNMENTS_Q}
  roma_log_environment login_restored
  export PYTHONUNBUFFERED=1
  if [[ -n ${CONDA_ENV_Q} ]]; then
    YUNDAO_STAGE=conda
    echo '[YUNDAO 1/4] Activating Conda environment'
    if conda activate ${CONDA_ENV_Q}; then
      echo '[YUNDAO] Conda activated; Python:'
    else
      rc=\$?
      echo \"[YUNDAO ERROR] conda activate failed, exit=\${rc}\" >&2
      exit \"\${rc}\"
    fi
  else
    echo '[YUNDAO 1/4] Using project interpreter (no Conda env requested)'
  fi
  roma_log_environment conda_before_restore
  roma_restore_environment ${ENV_ASSIGNMENTS_Q}
  roma_log_environment conda_restored
  cd ${PROJ_DIR_Q}
  command -v ${PYTHON_BIN_Q}
  ${PYTHON_BIN_Q} --version

  # 主节点产生启动时间戳，其他节点读取同一个值，保证 OUT_DIR 一致。
  # 显式 OUTPUT_DIR 时沿用精确路径，可恢复已有实验；此时不另加时间戳。
  YUNDAO_STAGE=timestamp
  echo '[YUNDAO 2/4] Resolving experiment directory'
  if [[ -n ${OUTPUT_DIR_OVERRIDE_Q:+x} ]]; then
    OUT_DIR=${OUTPUT_DIR_OVERRIDE_Q}
  else
    if (( ${MA_NUM_HOSTS} == 1 )); then
      RUN_TIMESTAMP=\$(TZ=${RUN_TIMEZONE_Q} date '+%Y%m%d_%H%M%S')
      if [[ -n ${RUN_TIMESTAMP_Q} ]]; then RUN_TIMESTAMP=${RUN_TIMESTAMP_Q}; fi
    else
      RUN_TIMESTAMP=\$(${PYTHON_BIN_Q} ${TIMESTAMP_HELPER_Q} \
        --master-addr ${MASTER_ADDR_Q} --master-port ${TIMESTAMP_PORT_Q} \
        --num-nodes ${MA_NUM_HOSTS} --node-rank ${VC_TASK_INDEX} \
        --timezone ${RUN_TIMEZONE_Q} --timestamp ${RUN_TIMESTAMP_Q} \
        --timeout ${TIMESTAMP_TIMEOUT_Q})
    fi
    OUT_DIR=${LOG_ROOT_Q}/${RUN_PREFIX_Q}\${RUN_TIMESTAMP}
  fi

  # 创建实验目录，保存脚本快照，将后续日志切换到这个目录。
  echo "DEBUG: OUT_DIR=[${OUT_DIR}] RT=[${RUN_TIMESTAMP}]" >  YUNDAO_STAGE=output3
  YUNDAO_STAGE=output
  mkdir -p \"\${OUT_DIR}\"
  LOG_FILE=\"\${OUT_DIR}/yundao_node_${VC_TASK_INDEX}.log\"
  exec > >(tee -a \"\${LOG_FILE}\" >&3) 2>&1
  # 每节点的快照集中到子目录，避免训练结果根目录堆满脚本副本。
  SNAPSHOT_DIR=\"\${OUT_DIR}/launch/node_${VC_TASK_INDEX}\"
  mkdir -p \"\${SNAPSHOT_DIR}\"
  cp ${BOOTSTRAP_LOG_Q} \"\${SNAPSHOT_DIR}/startup.log\"
  cp ${SCRIPT_PATH_Q} \"\${SNAPSHOT_DIR}/yundao_train_multi4_nvs_lora.sh\"
  cp ${ENV_HELPER_Q} \"\${SNAPSHOT_DIR}/roma_environment.sh\"
  cp ${TIMESTAMP_HELPER_Q} \"\${SNAPSHOT_DIR}/sequence_run_timestamp.py\"
  cp ${CONFIG_Q} \"\${SNAPSHOT_DIR}/train_config.yaml\"
  cp examples/wanvideo/model_training/train_with_accelerate_rgbd_nvs.py \"\${SNAPSHOT_DIR}/\"
  roma_log_environment rank_launch
  echo \"[YUNDAO 3/4] output=\${OUT_DIR}; log=\${LOG_FILE}\"

  # 精度等训练超参全部来自 yaml；这里只负责分布式拓扑。
  # 与 train_multi4_nvs_lora.sh 一致，用 venv python 跑 accelerate 模块入口。
  YUNDAO_STAGE=accelerate
  echo '[YUNDAO 4/4] Launching accelerate with DDP; all ${MA_NUM_HOSTS} nodes must join'
  SCRIPT_ARGS=(--config ${CONFIG_Q} \"output_path=\${OUT_DIR}\" ${EXTRA_ARGS_Q})
  echo \"[YUNDAO] SCRIPT_ARGS: \${SCRIPT_ARGS[*]}\"
  ${PYTHON_BIN_Q} -u -m accelerate.commands.launch \\
    ${MULTI_GPU_FLAG} \\
    --num_machines ${MA_NUM_HOSTS} \\
    --machine_rank ${VC_TASK_INDEX} \\
    --main_process_ip ${MASTER_ADDR_Q} \\
    --main_process_port ${MASTER_PORT} \\
    --num_processes ${WORLD_SIZE} \\
    -m examples.wanvideo.model_training.train_with_accelerate_rgbd_nvs \\
    \"\${SCRIPT_ARGS[@]}\"
  echo '[YUNDAO DONE] Training command completed'
"
