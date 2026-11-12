#!/usr/bin/env bash
# ROMA projection-specific launcher, based on yundao_train_gae_patch1_re10k.sh.
# Use yundao_resume_gae_projection_16npu.sh; it sets the frozen code,16 ranks,
# full training manifest, pinned step8600 state and step30000 target.
# Model/optimizer/scheduler resume; rank RNGs are reseeded after3->16 ranks.
set -euo pipefail
# Dedicated projection variant of yundao_train_gae_patch1_re10k.sh.
# Keep the reference login-shell/environment restoration and distributed protocol.

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${CUR_DIR}/.." && pwd)"

# =================================================================
# 1. 检查平台注入的分布式参数。每个节点都要执行这个入口。
# =================================================================
for roma_var in VC_WORKER_HOSTS MA_NUM_HOSTS MA_NUM_GPUS VC_TASK_INDEX; do
  if [[ -z "${!roma_var:-}" ]]; then
    echo "Missing ROMA environment variable: ${roma_var}" >&2
    echo "Run this script in a ROMA job on every node." >&2
    echo "For local training: bash train_script/train_multi4_nvs_lora.sh train_config/normal_config/gae_re10k_patch1.yaml" >&2
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
# 训练拓扑以显式覆盖值为准，不直接信任平台注入值：平台注入值若大于 pod
# 实际卡数，多个 rank 会映射到同一块物理 NPU，HCCL 建组直接报
# Ranktable_Detect_Failed(EI0015) "same device physic id"（首次 6 卡作业
# 实测：pod 实际只可见 NPU 4,5,6,7 四张卡，6 进程在第一个 barrier 崩）。
# NUM_HOSTS 默认 1（单 pod）；NUM_GPUS_PER_NODE 默认自动探测 pod 可见的
# NPU 数（NPU_VISIBLE_DEVICES / ASCEND_VISIBLE_DEVICES / ASCEND_RT_VISIBLE_
# DEVICES 的逗号项数，退化则数 /dev/davinciN），可用 YUNDAO_NUM_GPUS_PER_NODE
# 强制覆盖；进程数超过可见卡数会直接报错退出。
NUM_HOSTS="${YUNDAO_NUM_HOSTS:-1}"
detect_visible_npu_count() {
  local vis="${NPU_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-}}}"
  local n=0
  if [[ -n "${vis}" ]]; then
    n=$(awk -F, '{k=0; for(i=1;i<=NF;i++){if($i~/^[0-9]+$/){k++}else if($i~/^[0-9]+-[0-9]+$/){split($i,r,"-");k+=r[2]-r[1]+1}};print k}' <<< "${vis}")
  fi
  if (( n == 0 )); then
    n=$(compgen -G "/dev/davinci[0-9]" | wc -l)
  fi
  echo "${n}"
}
AVAILABLE_NPUS="$(detect_visible_npu_count)"
NUM_GPUS_PER_NODE="${YUNDAO_NUM_GPUS_PER_NODE:-${AVAILABLE_NPUS:-0}}"
[[ "${NUM_HOSTS}" =~ ^[1-9][0-9]*$ && "${NUM_GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "YUNDAO_NUM_HOSTS and YUNDAO_NUM_GPUS_PER_NODE must be positive integers" >&2
  exit 1
}
if (( AVAILABLE_NPUS > 0 && NUM_GPUS_PER_NODE > AVAILABLE_NPUS )); then
  echo "NUM_GPUS_PER_NODE (${NUM_GPUS_PER_NODE}) exceeds the ${AVAILABLE_NPUS} NPU(s) visible on this pod." >&2
  echo "Extra ranks would map to an already-used physical device and HCCL init fails (EI0015)." >&2
  echo "Fix: set YUNDAO_NUM_GPUS_PER_NODE=${AVAILABLE_NPUS}, or request a larger job spec." >&2
  exit 1
fi
(( NUM_HOSTS <= MA_NUM_HOSTS )) || {
  echo "YUNDAO_NUM_HOSTS (${NUM_HOSTS}) cannot exceed MA_NUM_HOSTS (${MA_NUM_HOSTS})" >&2
  exit 1
}
# 平台在所有 pod 上执行同一入口；不参与训练的节点在这里直接退出。
if (( VC_TASK_INDEX >= NUM_HOSTS )); then
  echo "[YUNDAO] node ${VC_TASK_INDEX} not among the ${NUM_HOSTS} training nodes; exiting."
  exit 0
fi
WORLD_SIZE="$((NUM_HOSTS * NUM_GPUS_PER_NODE))"
# 时间戳同步使用独立端口（训练端口 + 1），避免和 accelerate 的连接互相干扰。
# 仅多节点时才会真正用到；单节点直接读本机时钟。
TIMESTAMP_PORT="${YUNDAO_TIMESTAMP_PORT:-$((MASTER_PORT + 1))}"
if (( NUM_HOSTS > 1 )); then
  [[ "${TIMESTAMP_PORT}" =~ ^[0-9]+$ ]] && (( TIMESTAMP_PORT > 0 && TIMESTAMP_PORT < 65536 && TIMESTAMP_PORT != MASTER_PORT )) || {
    echo "Invalid timestamp port (must differ from MASTER_PORT): ${TIMESTAMP_PORT}" >&2
    exit 1
  }
fi
echo "NNODES: ${NUM_HOSTS} (platform MA_NUM_HOSTS: ${MA_NUM_HOSTS})"
echo "MASTER_HOST: ${MASTER_HOST}"
echo "NODE_RANK: ${VC_TASK_INDEX}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "NGPUS_PER_NODE: ${NUM_GPUS_PER_NODE} (platform MA_NUM_GPUS: ${MA_NUM_GPUS})"
echo "NUM_PROCESSES: ${WORLD_SIZE}"

# =================================================================
# 2. 读取训练参数。输出目录等时间戳同步后再确定。
# =================================================================
RUN_PREFIX="${YUNDAO_RUN_PREFIX:-gae_patch1_re10k}"
[[ "${RUN_PREFIX}" == */* ]] && {
  echo "YUNDAO_RUN_PREFIX must be a folder-name prefix; use YUNDAO_OUTPUT_DIR for a path." >&2
  exit 1
}
RUN_TIMEZONE="${YUNDAO_TIMEZONE:-Asia/Shanghai}"
RUN_TIMESTAMP="${YUNDAO_RUN_TIMESTAMP:-}"
RUN_TIMESTAMP_SET=0
if [[ -n "${RUN_TIMESTAMP}" ]]; then
  RUN_TIMESTAMP_SET=1
fi
[[ -z "${RUN_TIMESTAMP}" || "${RUN_TIMESTAMP}" =~ ^[0-9]{8}_[0-9]{6}$ ]] || {
  echo "YUNDAO_RUN_TIMESTAMP must have format YYYYMMDD_HHMMSS: ${RUN_TIMESTAMP}" >&2
  exit 1
}
OUTPUT_DIR_OVERRIDE="${YUNDAO_OUTPUT_DIR:-}"
OUTPUT_DIR_OVERRIDE_SET=0
if [[ -n "${OUTPUT_DIR_OVERRIDE}" ]]; then
  OUTPUT_DIR_OVERRIDE_SET=1
  if [[ "${OUTPUT_DIR_OVERRIDE}" != /* ]]; then
    OUTPUT_DIR_OVERRIDE="${PROJ_DIR}/${OUTPUT_DIR_OVERRIDE}"
  fi
fi
LOG_ROOT="${YUNDAO_LOG_ROOT:-${PROJ_DIR}/logs}"
[[ "${LOG_ROOT}" == /* ]] || LOG_ROOT="${PROJ_DIR}/${LOG_ROOT}"

TRAIN_SCRIPT="${YUNDAO_TRAIN_SCRIPT:?GAE projection requires its frozen train.py}"
TRAIN_CODE_ROOT="$(dirname "${TRAIN_SCRIPT}")"
test -s "${TRAIN_SCRIPT}" || { echo "Missing projection training script: ${TRAIN_SCRIPT}" >&2; exit 1; }
printf -v RESUME_CKPT_Q '%q' "${YUNDAO_RESUME_CKPT:-}"
printf -v TRAIN_SCRIPT_Q '%q' "${TRAIN_SCRIPT}"
printf -v TRAIN_CODE_ROOT_Q '%q' "${TRAIN_CODE_ROOT}"
printf -v PREFLIGHT_Q '%q' "${PROJ_DIR}/train_script/check_gae_projection_yundao.py"
CONFIG="${1:-train_config/normal_config/gae_re10k_patch1.yaml}"
if [[ "${CONFIG}" != /* ]]; then CONFIG="${PROJ_DIR}/${CONFIG}"; fi
test -s "${CONFIG}" || { echo "Missing training config: ${CONFIG}" >&2; exit 1; }
shift || true
EXTRA_ARGS=("$@")

# 固定/更换随机种子（第二 seed 用 YUNDAO_SEED=43）；不设置则用 yaml 的 seed=42。
# 两 arm 各自的 seed 必须成对一致（rand-seed42 vs vggt-seed42, ...）。
SEED_ARGS=()
if [[ -n "${YUNDAO_SEED:-}" ]]; then
  [[ "${YUNDAO_SEED}" =~ ^[1-9][0-9]*$ ]] || {
    echo "YUNDAO_SEED must be a positive integer: ${YUNDAO_SEED}" >&2; exit 1;
  }
  SEED_ARGS=("seed=${YUNDAO_SEED}")
fi

# 从 accelerator.save_state 保存的 checkpoint-step-N 精确恢复（模型+优化器+
# 调度器+随机状态）。global_step 必须显式覆盖：训练脚本不从 checkpoint 读
# 步数，而是用 config 里的 global_step（train_with_accelerate_gae_nvs.py
# load_state 之后才执行 global_step = int(cfg.global_step)）。默认指向
# gae_patch1_re10k20261102_195757 崩溃前存下的 step-1000（当时首个 eval 懒
# 加载 GAE 时缺 depth_anything_3 而崩，已修）；YUNDAO_RESUME_CKPT="" 从头训。
RESUME_CKPT="${YUNDAO_RESUME_CKPT:-${PROJ_DIR}/logs/gae_patch1_re10k20261102_195757/checkpoint-step-1000}"
RESUME_ARGS=()
if [[ -n "${RESUME_CKPT}" ]]; then
  RESUME_STEP=""
  if [[ "$(basename "${RESUME_CKPT}")" =~ ^checkpoint-step-([0-9]+)$ ]]; then
    RESUME_STEP="${BASH_REMATCH[1]}"
  fi
  [[ -n "${RESUME_STEP}" ]] || {
    echo "YUNDAO_RESUME_CKPT must point at a checkpoint-step-<N> directory: ${RESUME_CKPT}" >&2
    exit 1
  }
  [[ -d "${RESUME_CKPT}" ]] || { echo "Resume checkpoint missing: ${RESUME_CKPT}" >&2; exit 1; }
  [[ -s "${RESUME_CKPT}/model.safetensors" ]] || {
    echo "Resume checkpoint incomplete (missing model.safetensors): ${RESUME_CKPT}" >&2
    exit 1
  }
  RESUME_ARGS=("resume=true" "training_state_dir=${RESUME_CKPT}" "global_step=${RESUME_STEP}")
  echo "Resume: ${RESUME_CKPT} (global_step=${RESUME_STEP})"
fi

# eval / checkpoint 节奏。config 默认 validate_step=1000；上次 run 首个 eval
# （step 1000）才暴露问题，改成 300 更早触发 eval，便于及时发现异常。
VALIDATE_STEP="${YUNDAO_VALIDATE_STEP:-300}"
[[ "${VALIDATE_STEP}" =~ ^[1-9][0-9]*$ ]] || {
  echo "YUNDAO_VALIDATE_STEP must be a positive integer: ${VALIDATE_STEP}" >&2; exit 1;
}
echo "Validate step: ${VALIDATE_STEP}"

# =================================================================
# 2.5 选择训练数据集。本消融约定 re10k-only（默认 re10k），以 dotlist 覆盖
# config 里的 kubric4d_root / manifest_path / scene_filter 三个对齐列表
# （scene_filter 置 null）。仍保留 blendmvs 分支，便于复验混训配方。
# =================================================================
DATASETS_SPEC="${YUNDAO_DATASETS:-re10k}"
DATASET_ARGS=()
if [[ "${DATASETS_SPEC}" != "none" ]]; then
  join_yaml_list() {
    local out="" item
    for item in "$@"; do out+="\"${item}\","; done
    echo "[${out%,}]"
  }
  IFS=',' read -r -a DATASET_LIST <<< "${DATASETS_SPEC}"
  DS_ROOTS=()
  DS_MANIFESTS=()
  for ds in "${DATASET_LIST[@]}"; do
    ds="${ds//[[:space:]]/}"
    case "${ds}" in
      blendmvs)
        DS_ROOTS+=("${YUNDAO_BMVS_ROOT:-/mnt/sfs_turbo/t00845730/data/blendedmvs/extracted}")
        DS_MANIFESTS+=("${YUNDAO_BMVS_MANIFEST:-/mnt/sfs_turbo/t00845730/data/blendedmvs/manifest_train_x9.jsonl}")
        ;;
      re10k)
        DS_ROOTS+=("${YUNDAO_RE10K_ROOT:-/mnt/sfs_turbo/t00845730/re10k/extracted}")
        DS_MANIFESTS+=("${YUNDAO_RE10K_MANIFEST:-/mnt/sfs_turbo/t00845730/re10k/manifest_train.jsonl}")
        ;;
      *)
        echo "Invalid YUNDAO_DATASETS entry: '${ds}' (expected blendmvs / re10k / none)" >&2
        exit 1
        ;;
    esac
  done
  # 启动前先校验数据在位，避免多机起跑后某节点才因缺数据崩溃。
  for p in "${DS_ROOTS[@]}"; do
    [[ -d "${p}" ]] || { echo "Dataset root missing: ${p}" >&2; exit 1; }
  done
  for p in "${DS_MANIFESTS[@]}"; do
    [[ -s "${p}" ]] || { echo "Dataset manifest missing: ${p}" >&2; exit 1; }
  done
  DATASET_ARGS=(
    "kubric4d_root=$(join_yaml_list "${DS_ROOTS[@]}")"
    "manifest_path=$(join_yaml_list "${DS_MANIFESTS[@]}")"
    "scene_filter=null"
  )
  echo "Datasets: ${DATASETS_SPEC}"
  echo "  kubric4d_root: ${DS_ROOTS[*]}"
  echo "  manifest_path: ${DS_MANIFESTS[*]}"
fi

# Python 环境：指定 Conda 环境则激活后用其 python；否则用项目 .venv（存在时）。
CONDA_ENV="${YUNDAO_CONDA_ENV:-}"
if [[ -n "${CONDA_ENV}" ]]; then
  CONDA_ENV_SET=1
  PYTHON_BIN="python"
else
  CONDA_ENV_SET=0
  PYTHON_BIN="${YUNDAO_PYTHON_BIN:-}"
  if [[ -z "${PYTHON_BIN}" && -x "${PROJ_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJ_DIR}/.venv/bin/python"
  fi
  [[ -n "${PYTHON_BIN}" ]] || PYTHON_BIN="python"
fi

MULTI_GPU_FLAG=""
if (( WORLD_SIZE > 1 )); then MULTI_GPU_FLAG="--multi_gpu"; fi

# Dry run validates files/config/topology without login shell, training or HCCL.
if [[ "${YUNDAO_DRY_RUN:-0}" == "1" ]]; then
  "${PYTHON_BIN}" "${PROJ_DIR}/train_script/check_gae_projection_yundao.py"     --config "${CONFIG}" --checkpoint "${RESUME_CKPT}"     --world-size "${WORLD_SIZE}" --code-root "${TRAIN_CODE_ROOT}"     --output-dir "${OUTPUT_DIR_OVERRIDE}" "${EXTRA_ARGS[@]}"
  printf '[DRY RUN] training entry: %s\n' "${TRAIN_SCRIPT}"
  exit 0
fi

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
DATASET_ARGS_Q=""
if (( ${#DATASET_ARGS[@]} > 0 )); then printf -v DATASET_ARGS_Q '%q ' "${DATASET_ARGS[@]}"; fi
SEED_ARGS_Q=""
if (( ${#SEED_ARGS[@]} > 0 )); then printf -v SEED_ARGS_Q '%q ' "${SEED_ARGS[@]}"; fi
RESUME_ARGS_Q=""
if (( ${#RESUME_ARGS[@]} > 0 )); then printf -v RESUME_ARGS_Q '%q ' "${RESUME_ARGS[@]}"; fi
printf -v VALIDATE_STEP_Q '%q' "${VALIDATE_STEP}"
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
  echo '[YUNDAO START] node=${VC_TASK_INDEX}/${NUM_HOSTS}, processes=${WORLD_SIZE}'
  date -u
  source ${ENV_HELPER_Q}
  roma_log_environment login_before_restore
  roma_restore_environment --platform-paths-first ${ENV_ASSIGNMENTS_Q}
  roma_log_environment login_restored
  export PYTHONUNBUFFERED=1
  # sudo 登录 shell 的默认 umask 会把实验目录建成 750、文件 640，平台诊断
  # 及其他用户读不到（曾导致 faultdiag parse 报 FileNotExist）；放开为 000，
  # 与其它 777 实验目录约定一致。
  umask 000
  # 注意：大字符串里内嵌的引号会截断外层引用，变量展开不受引号保护，
  # 因此判空一律用外层预先算好的 SET 标志位，值本身用 %q 变量传递。
  if (( CONDA_ENV_SET == 1 )); then
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
  export PYTHONPATH=${TRAIN_CODE_ROOT_Q}:${PROJ_DIR_Q}:\${PYTHONPATH:-}
  command -v ${PYTHON_BIN_Q}
  ${PYTHON_BIN_Q} --version

  # 主节点产生启动时间戳，其他节点读取同一个值，保证 OUT_DIR 一致。
  # 显式 OUTPUT_DIR 时沿用精确路径，可恢复已有实验；此时不另加时间戳。
  YUNDAO_STAGE=timestamp
  echo '[YUNDAO 2/4] Resolving experiment directory'
  if (( OUTPUT_DIR_OVERRIDE_SET == 1 )); then
    OUT_DIR=${OUTPUT_DIR_OVERRIDE_Q}
  else
    if (( ${NUM_HOSTS} == 1 )); then
      RUN_TIMESTAMP=\$(TZ=${RUN_TIMEZONE_Q} date '+%Y%m%d_%H%M%S')
      if (( RUN_TIMESTAMP_SET == 1 )); then RUN_TIMESTAMP=${RUN_TIMESTAMP_Q}; fi
    else
      RUN_TIMESTAMP=\$(${PYTHON_BIN_Q} ${TIMESTAMP_HELPER_Q} \
        --master-addr ${MASTER_ADDR_Q} --master-port ${TIMESTAMP_PORT_Q} \
        --num-nodes ${NUM_HOSTS} --node-rank ${VC_TASK_INDEX} \
        --timezone ${RUN_TIMEZONE_Q} --timestamp ${RUN_TIMESTAMP_Q} \
        --timeout ${TIMESTAMP_TIMEOUT_Q})
    fi
    OUT_DIR=${LOG_ROOT_Q}/${RUN_PREFIX_Q}\${RUN_TIMESTAMP}
  fi

  # 创建实验目录，保存脚本快照，将后续日志切换到这个目录。
  YUNDAO_STAGE=output
  mkdir -p \"\${OUT_DIR}\"
  LOG_FILE=\"\${OUT_DIR}/yundao_node_${VC_TASK_INDEX}.log\"
  exec > >(tee -a \"\${LOG_FILE}\" >&3) 2>&1
  # 每节点的快照集中到子目录，避免训练结果根目录堆满脚本副本。
  SNAPSHOT_DIR=\"\${OUT_DIR}/launch/node_${VC_TASK_INDEX}\"
  mkdir -p \"\${SNAPSHOT_DIR}\"
  cp ${BOOTSTRAP_LOG_Q} \"\${SNAPSHOT_DIR}/startup.log\"
  cp ${SCRIPT_PATH_Q} \"\${SNAPSHOT_DIR}/yundao_train_gae_projection.sh\"
  cp ${ENV_HELPER_Q} \"\${SNAPSHOT_DIR}/roma_environment.sh\"
  cp ${TIMESTAMP_HELPER_Q} \"\${SNAPSHOT_DIR}/sequence_run_timestamp.py\"
  cp ${CONFIG_Q} \"\${SNAPSHOT_DIR}/train_config.yaml\"
  cp ${TRAIN_SCRIPT_Q} \"\${SNAPSHOT_DIR}/train.py\"
  roma_log_environment rank_launch
  echo \"[YUNDAO 3/4] output=\${OUT_DIR}; log=\${LOG_FILE}\"

  # 精度等训练超参全部来自 yaml；这里只负责分布式拓扑。
  # 与 train_multi4_nvs_lora.sh 一致，用 venv python 跑 accelerate 模块入口。
  YUNDAO_STAGE=accelerate
  echo '[YUNDAO 4/4] Launching accelerate with DDP; all ${NUM_HOSTS} nodes must join'
  # seed 覆盖、数据集覆盖、eval 节奏、断点恢复参数放在 config 之后、用户
  # EXTRA_ARGS 之前：命令行显式 key=value 仍可最后覆盖这些设置。
  SCRIPT_ARGS=(--config ${CONFIG_Q} "output_path=\${OUT_DIR}" ${SEED_ARGS_Q} ${DATASET_ARGS_Q} "validate_step=${VALIDATE_STEP_Q}" ${RESUME_ARGS_Q} ${EXTRA_ARGS_Q})
  echo \"[YUNDAO] SCRIPT_ARGS: \${SCRIPT_ARGS[*]}\"
  ${PYTHON_BIN_Q} ${PREFLIGHT_Q} --config ${CONFIG_Q} \
    --checkpoint ${RESUME_CKPT_Q} --world-size ${WORLD_SIZE} \
    --code-root ${TRAIN_CODE_ROOT_Q} --output-dir \"\${OUT_DIR}\" \
    --check-hardware --local-processes ${NUM_GPUS_PER_NODE} ${EXTRA_ARGS_Q}
  ${PYTHON_BIN_Q} -u -m accelerate.commands.launch \\
    ${MULTI_GPU_FLAG} \\
    --num_machines ${NUM_HOSTS} \\
    --machine_rank ${VC_TASK_INDEX} \\
    --main_process_ip ${MASTER_ADDR_Q} \\
    --main_process_port ${MASTER_PORT} \\
    --num_processes ${WORLD_SIZE} \\
    ${TRAIN_SCRIPT_Q} \\
    \"\${SCRIPT_ARGS[@]}\"
  echo '[YUNDAO DONE] Training command completed'
"
