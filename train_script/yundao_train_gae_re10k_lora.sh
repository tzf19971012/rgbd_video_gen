#!/usr/bin/env bash
# GAE-64 latent + RGB aux loss，re10k-only LoRA 微调（云道/本机两用入口）。
# 改编自 yundao_train_re10k13b_rand.sh，训练模块换成 GAE latent 的
# train_with_accelerate_gae_nvs.py；数据、拓扑、时间戳目录约定保持一致。
#
# 用法（ROMA 作业里每个节点执行同一入口）:
#   bash train_script/yundao_train_gae_re10k_lora.sh
#   bash train_script/yundao_train_gae_re10k_lora.sh [config.yaml] [key=value ...]
#
# 例子（临时覆盖超参，无需改 yaml）:
#   bash train_script/yundao_train_gae_re10k_lora.sh \
#       train_config/normal_config/gae_re10k_lora_rgb_aux.yaml max_steps=3000
#
# 本机调试（无 ROMA 环境变量时自动走单进程直跑分支，卡由
# ASCEND_RT_VISIBLE_DEVICES 指定）:
#   ASCEND_RT_VISIBLE_DEVICES=2 bash train_script/yundao_train_gae_re10k_lora.sh
#
# 常用环境变量（都有默认值，一般不用设置）:
#   YUNDAO_SEED             覆盖随机种子；不设置则用 yaml 里的 seed=42。
#   YUNDAO_CONDA_ENV        集群上的 Conda 环境名；设置后用该环境的 python。
#   YUNDAO_PYTHON_BIN       Python 解释器路径；默认项目 .venv/bin/python。
#   YUNDAO_RUN_PREFIX       实验目录名前缀，默认 gae_re10k_rgb_aux。
#   YUNDAO_OUTPUT_DIR       精确指定输出目录（绝对路径），用于恢复已有实验。
#   YUNDAO_LOG_ROOT         实验输出根目录，默认 <项目>/logs。
#   YUNDAO_MASTER_ADDR/PORT 覆盖主节点地址/端口；端口默认 29700。
#   YUNDAO_NUM_HOSTS        参与训练的节点数；默认 1（单 pod 作业）。
#   YUNDAO_NUM_GPUS_PER_NODE 每节点训练进程数；默认自动探测 pod 可见的 NPU 数。
#   YUNDAO_TIMEZONE         时间戳时区，默认 Asia/Shanghai。
#   YUNDAO_RUN_TIMESTAMP    固定时间戳（YYYYMMDD_HHMMSS），所有节点需一致。
#   YUNDAO_NO_LOGIN_SHELL=1 不经过 sudo 登录 shell 直接执行。
#   YUNDAO_DATASETS         数据集选择，逗号分隔：re10k / blendmvs。默认 re10k。
#   YUNDAO_RE10K_ROOT       re10k 数据根目录（默认 .../re10k/extracted）。
#   YUNDAO_RE10K_MANIFEST   re10k manifest（默认 .../re10k/manifest_train.jsonl）。
#   YUNDAO_RESUME_FROM      续训 checkpoint 目录（checkpoint-step-N）；设为 none 关闭续训。
#                           默认从 gae_re10k_rgb_aux20261029_140212/checkpoint-step-3000
#                           续训，且输出目录同步默认回写该实验目录（断点续跑）。
#   YUNDAO_EVAL_NUM_INFERENCE_STEPS  eval 推理步数；默认 25。
#
# 注意：GAE trainer 的 eval 是训练场景浅拷贝（监控用）；冻结测试集走
# eval_script 单独评（re10k_test cache + manifest_test.jsonl）。

set -euo pipefail

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${CUR_DIR}/.." && pwd)"

CONFIG="${1:-train_config/normal_config/gae_re10k_lora_rgb_aux.yaml}"
if [[ "${CONFIG}" != /* ]]; then CONFIG="${PROJ_DIR}/${CONFIG}"; fi
test -s "${CONFIG}" || { echo "Missing training config: ${CONFIG}" >&2; exit 1; }
shift || true
EXTRA_ARGS=("$@")

RUN_PREFIX="${YUNDAO_RUN_PREFIX:-gae_re10k_rgb_aux}"
[[ "${RUN_PREFIX}" == */* ]] && {
  echo "YUNDAO_RUN_PREFIX must be a folder-name prefix; use YUNDAO_OUTPUT_DIR for a path." >&2
  exit 1
}
RUN_TIMEZONE="${YUNDAO_TIMEZONE:-Asia/Shanghai}"
LOG_ROOT="${YUNDAO_LOG_ROOT:-${PROJ_DIR}/logs}"
[[ "${LOG_ROOT}" == /* ]] || LOG_ROOT="${PROJ_DIR}/${LOG_ROOT}"

# 续训默认：输出回写到原实验目录，checkpoint 也从该目录取（都可用环境变量覆盖）。
YUNDAO_OUTPUT_DIR="${YUNDAO_OUTPUT_DIR:-${LOG_ROOT}/gae_re10k_rgb_aux20261029_140212}"
YUNDAO_RESUME_FROM="${YUNDAO_RESUME_FROM:-${YUNDAO_OUTPUT_DIR}/checkpoint-step-3000}"
YUNDAO_EVAL_NUM_INFERENCE_STEPS="${YUNDAO_EVAL_NUM_INFERENCE_STEPS:-25}"
OUTPUT_DIR_OVERRIDE="${YUNDAO_OUTPUT_DIR}"

# Python 环境：指定 Conda 环境则激活后用其 python；否则用项目 .venv。
CONDA_ENV="${YUNDAO_CONDA_ENV:-}"
if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="${YUNDAO_PYTHON_BIN:-}"
  if [[ -z "${PYTHON_BIN}" && -x "${PROJ_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJ_DIR}/.venv/bin/python"
  fi
  [[ -n "${PYTHON_BIN}" ]] || PYTHON_BIN="python"
fi

# =================================================================
# 2.5 选择训练数据集。GAE latent cache 按数据集分目录（gae_dataset_name），
# 切数据集 = 换 manifest + 换 cache 目录，两处必须成对改。
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
  DS_GAE_NAMES=()
  for ds in "${DATASET_LIST[@]}"; do
    ds="${ds//[[:space:]]/}"
    case "${ds}" in
      re10k)
        DS_ROOTS+=("${YUNDAO_RE10K_ROOT:-/mnt/sfs_turbo/t00845730/re10k/extracted}")
        DS_MANIFESTS+=("${YUNDAO_RE10K_MANIFEST:-/mnt/sfs_turbo/t00845730/re10k/manifest_train.jsonl}")
        DS_GAE_NAMES+=("${YUNDAO_GAE_RE10K_CACHE:-re10k_train}")
        ;;
      blendmvs)
        DS_ROOTS+=("${YUNDAO_BMVS_ROOT:-/mnt/sfs_turbo/t00845730/data/blendedmvs/extracted}")
        DS_MANIFESTS+=("${YUNDAO_BMVS_MANIFEST:-/mnt/sfs_turbo/t00845730/data/blendedmvs/manifest_all.jsonl}")
        DS_GAE_NAMES+=("${YUNDAO_GAE_BMVS_CACHE:-bmvs}")
        ;;
      *)
        echo "Invalid YUNDAO_DATASETS entry: '${ds}' (expected re10k / blendmvs / none)" >&2
        exit 1
        ;;
    esac
  done
  # 启动前校验数据与 GAE cache 在位。
  for p in "${DS_ROOTS[@]}"; do
    [[ -d "${p}" ]] || { echo "Dataset root missing: ${p}" >&2; exit 1; }
  done
  for p in "${DS_MANIFESTS[@]}"; do
    [[ -s "${p}" ]] || { echo "Dataset manifest missing: ${p}" >&2; exit 1; }
  done
  CACHE_ROOT="${YUNDAO_GAE_CACHE_ROOT:-/mnt/sfs_turbo/t00845730/gae_latent_cache}"
  [[ -d "${CACHE_ROOT}" ]] || { echo "GAE cache root missing: ${CACHE_ROOT}" >&2; exit 1; }
  for name in "${DS_GAE_NAMES[@]}"; do
    [[ -d "${CACHE_ROOT}/${name}" ]] || { echo "GAE cache missing: ${CACHE_ROOT}/${name}" >&2; exit 1; }
    [[ -s "${CACHE_ROOT}/${name}/latent_stats.pt" ]] || {
      echo "GAE latent_stats.pt missing: ${CACHE_ROOT}/${name}/latent_stats.pt" >&2; exit 1; }
  done
  DATASET_ARGS=(
    "kubric4d_root=$(join_yaml_list "${DS_ROOTS[@]}")"
    "manifest_path=$(join_yaml_list "${DS_MANIFESTS[@]}")"
    "scene_filter=null"
    "gae_cache_root=${CACHE_ROOT}"
    "gae_dataset_name=${DS_GAE_NAMES[0]}"
  )
  echo "Datasets: ${DATASETS_SPEC}  (GAE cache: ${DS_GAE_NAMES[*]})"
fi

SEED_ARGS=()
if [[ -n "${YUNDAO_SEED:-}" ]]; then
  [[ "${YUNDAO_SEED}" =~ ^[1-9][0-9]*$ ]] || {
    echo "YUNDAO_SEED must be a positive integer: ${YUNDAO_SEED}" >&2; exit 1; }
  SEED_ARGS=("seed=${YUNDAO_SEED}")
fi

# =================================================================
# 续训 + eval 推理步数：从 checkpoint-step-N 目录名解析 global_step，
# 组装 resume=true / training_state_dir / global_step / eval_num_inference_steps。
# =================================================================
RESUME_ARGS=()
if [[ -n "${YUNDAO_RESUME_FROM}" && "${YUNDAO_RESUME_FROM}" != "none" ]]; then
  RESUME_CKPT="${YUNDAO_RESUME_FROM%/}"
  [[ -d "${RESUME_CKPT}" ]] || { echo "Resume checkpoint dir missing: ${RESUME_CKPT}" >&2; exit 1; }
  RESUME_STEP="$(basename "${RESUME_CKPT}")"
  [[ "${RESUME_STEP}" =~ ^checkpoint-step-([0-9]+)$ ]] || {
    echo "Cannot parse global_step from checkpoint dir name: ${RESUME_CKPT}" >&2; exit 1; }
  RESUME_STEP="${BASH_REMATCH[1]}"
  RESUME_ARGS=("resume=true" "training_state_dir=${RESUME_CKPT}" "global_step=${RESUME_STEP}")
  echo "Resume from: ${RESUME_CKPT} (global_step=${RESUME_STEP})"
fi
[[ "${YUNDAO_EVAL_NUM_INFERENCE_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "YUNDAO_EVAL_NUM_INFERENCE_STEPS must be a positive integer: ${YUNDAO_EVAL_NUM_INFERENCE_STEPS}" >&2
  exit 1
}
RESUME_ARGS+=("eval_num_inference_steps=${YUNDAO_EVAL_NUM_INFERENCE_STEPS}")
echo "Eval inference steps: ${YUNDAO_EVAL_NUM_INFERENCE_STEPS}"

# =================================================================
# 本机直跑分支：无 ROMA 变量时单进程执行，不走 accelerate/HCCL。
# 卡的选择交给外层 ASCEND_RT_VISIBLE_DEVICES。
# =================================================================
if [[ -z "${VC_WORKER_HOSTS:-}" ]]; then
  echo "[YUNDAO-LOCAL] no ROMA env detected; running single-process"
  RUN_TIMESTAMP="${YUNDAO_RUN_TIMESTAMP:-$(TZ="${RUN_TIMEZONE}" date '+%Y%m%d_%H%M%S')}"
  if [[ -n "${OUTPUT_DIR_OVERRIDE}" ]]; then
    [[ "${OUTPUT_DIR_OVERRIDE}" == /* ]] || OUTPUT_DIR_OVERRIDE="${PROJ_DIR}/${OUTPUT_DIR_OVERRIDE}"
    OUT_DIR="${OUTPUT_DIR_OVERRIDE}"
  else
    OUT_DIR="${LOG_ROOT}/${RUN_PREFIX}${RUN_TIMESTAMP}"
  fi
  mkdir -p "${OUT_DIR}/launch/local"
  cp "${CONFIG}" "${OUT_DIR}/launch/local/train_config.yaml"
  cp "${BASH_SOURCE[0]}" "${OUT_DIR}/launch/local/yundao_train_gae_re10k_lora.sh"
  echo "[YUNDAO-LOCAL] output=${OUT_DIR}"
  SCRIPT_ARGS=(--config "${CONFIG}" "output_path=${OUT_DIR}" ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} ${DATASET_ARGS[@]+"${DATASET_ARGS[@]}"} ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
  echo "[YUNDAO-LOCAL] SCRIPT_ARGS: ${SCRIPT_ARGS[*]}"
  cd "${PROJ_DIR}"
  exec "${PYTHON_BIN}" -u -m examples.wanvideo.model_training.train_with_accelerate_gae_nvs "${SCRIPT_ARGS[@]}"
fi

# =================================================================
# 以下为 ROMA 平台多机分支（与 yundao_train_re10k13b_rand.sh 同约定）。
# =================================================================
for roma_var in VC_WORKER_HOSTS MA_NUM_HOSTS MA_NUM_GPUS VC_TASK_INDEX; do
  if [[ -z "${!roma_var:-}" ]]; then
    echo "Missing ROMA environment variable: ${roma_var}" >&2
    exit 1
  fi
done
[[ "${MA_NUM_HOSTS}" =~ ^[1-9][0-9]*$ && "${MA_NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "MA_NUM_HOSTS and MA_NUM_GPUS must be positive integers" >&2; exit 1; }
[[ "${VC_TASK_INDEX}" =~ ^[0-9]+$ ]] && (( VC_TASK_INDEX < MA_NUM_HOSTS )) || {
  echo "Invalid VC_TASK_INDEX: ${VC_TASK_INDEX}" >&2; exit 1; }
MASTER_HOST="${VC_WORKER_HOSTS}"
MASTER_ADDR="${YUNDAO_MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}"
MASTER_PORT="${YUNDAO_MASTER_PORT:-29700}"
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] && (( MASTER_PORT > 0 && MASTER_PORT < 65536 )) || {
  echo "Invalid master port: ${MASTER_PORT}" >&2; exit 1; }

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
  echo "YUNDAO_NUM_HOSTS/YUNDAO_NUM_GPUS_PER_NODE must be positive integers" >&2; exit 1; }
if (( AVAILABLE_NPUS > 0 && NUM_GPUS_PER_NODE > AVAILABLE_NPUS )); then
  echo "NUM_GPUS_PER_NODE (${NUM_GPUS_PER_NODE}) exceeds the ${AVAILABLE_NPUS} visible NPU(s); HCCL would fail (EI0015)." >&2
  echo "Fix: set YUNDAO_NUM_GPUS_PER_NODE=${AVAILABLE_NPUS}." >&2
  exit 1
fi
(( NUM_HOSTS <= MA_NUM_HOSTS )) || {
  echo "YUNDAO_NUM_HOSTS (${NUM_HOSTS}) cannot exceed MA_NUM_HOSTS (${MA_NUM_HOSTS})" >&2; exit 1; }
if (( VC_TASK_INDEX >= NUM_HOSTS )); then
  echo "[YUNDAO] node ${VC_TASK_INDEX} not among the ${NUM_HOSTS} training nodes; exiting."
  exit 0
fi
WORLD_SIZE="$((NUM_HOSTS * NUM_GPUS_PER_NODE))"
MASTER_PORT_TS=$((MASTER_PORT + 1))

echo "NNODES: ${NUM_HOSTS} (platform MA_NUM_HOSTS: ${MA_NUM_HOSTS})"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "NODE_RANK: ${VC_TASK_INDEX}"
echo "NGPUS_PER_NODE: ${NUM_GPUS_PER_NODE} (platform MA_NUM_GPUS: ${MA_NUM_GPUS})"
echo "NUM_PROCESSES: ${WORLD_SIZE}"

OUTPUT_DIR_OVERRIDE_SET=0
[[ -n "${OUTPUT_DIR_OVERRIDE}" ]] && OUTPUT_DIR_OVERRIDE_SET=1
RUN_TIMESTAMP="${YUNDAO_RUN_TIMESTAMP:-}"
RUN_TIMESTAMP_SET=0
[[ -n "${RUN_TIMESTAMP}" ]] && RUN_TIMESTAMP_SET=1
[[ -z "${RUN_TIMESTAMP}" || "${RUN_TIMESTAMP}" =~ ^[0-9]{8}_[0-9]{6}$ ]] || {
  echo "Invalid YUNDAO_RUN_TIMESTAMP: ${RUN_TIMESTAMP}" >&2; exit 1; }

BOOTSTRAP_DIR="${PROJ_DIR}/logs/.yundao_launch_logs"
mkdir -p "${BOOTSTRAP_DIR}"
BOOTSTRAP_LOG="${BOOTSTRAP_DIR}/node_${VC_TASK_INDEX}_$(TZ="${RUN_TIMEZONE}" date '+%Y%m%d_%H%M%S')_$$.log"
echo "启动阶段日志: ${BOOTSTRAP_LOG}"

source "${CUR_DIR}/roma_environment.sh"
roma_capture_environment

EXTRA_ARGS_Q=""
if (( ${#EXTRA_ARGS[@]} > 0 )); then printf -v EXTRA_ARGS_Q '%q ' "${EXTRA_ARGS[@]}"; fi
DATASET_ARGS_Q=""
if (( ${#DATASET_ARGS[@]} > 0 )); then printf -v DATASET_ARGS_Q '%q ' "${DATASET_ARGS[@]}"; fi
SEED_ARGS_Q=""
if (( ${#SEED_ARGS[@]} > 0 )); then printf -v SEED_ARGS_Q '%q ' "${SEED_ARGS[@]}"; fi
RESUME_ARGS_Q=""
if (( ${#RESUME_ARGS[@]} > 0 )); then printf -v RESUME_ARGS_Q '%q ' "${RESUME_ARGS[@]}"; fi
printf -v PROJ_DIR_Q '%q' "${PROJ_DIR}"
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
printf -v SCRIPT_PATH_Q '%q' "${CUR_DIR}/${BASH_SOURCE[0]##*/}"
printf -v ENV_HELPER_Q '%q' "${CUR_DIR}/roma_environment.sh"
printf -v TIMESTAMP_HELPER_Q '%q' "${CUR_DIR}/sequence_run_timestamp.py"
# Pod 镜像 env（PyTorch-2.7.1, py3.12）里没有 depth_anything_3；该包是纯
# Python，把 Depth-Anything-3/src 追加到 PYTHONPATH 即可。注意：不要把 GAE
# 仓库路径加进来 —— 其 src/utils 是常规包，会遮蔽本项目的命名空间包 utils
# （utils.projection 将报 ModuleNotFoundError）。
DA3_SRC="${YUNDAO_DA3_SRC:-/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src}"
EXTRA_PYPATH="${DA3_SRC}"
printf -v EXTRA_PYPATH_Q '%q' "${EXTRA_PYPATH}"

if [[ "${YUNDAO_NO_LOGIN_SHELL:-0}" == "1" ]] || ! command -v sudo >/dev/null || [[ "$(id -u)" == "0" ]]; then
  LOGIN_SHELL=(bash -l -i -c)
else
  LOGIN_SHELL=(sudo -H bash -l -i -c)
fi
"${LOGIN_SHELL[@]}" "
  exec 3>&1
  exec > >(tee -a ${BOOTSTRAP_LOG_Q} >&3) 2>&1
  set +u
  set -eo pipefail
  YUNDAO_STAGE=environment
  echo '[YUNDAO START] node=${VC_TASK_INDEX}/${NUM_HOSTS}, processes=${WORLD_SIZE}'
  date -u
  source ${ENV_HELPER_Q}
  roma_restore_environment
  export PYTHONUNBUFFERED=1
  export PYTHONPATH=\"\${PYTHONPATH:+\${PYTHONPATH}:}${EXTRA_PYPATH_Q}\"
  if [[ \"\${CONDA_ENV_SET:-0}\" == \"1\" ]]; then
    YUNDAO_STAGE=conda
    echo '[YUNDAO 1/4] Activating Conda environment'
    conda activate ${CONDA_ENV_Q} || { echo '[YUNDAO ERROR] conda activate failed' >&2; exit 1; }
  else
    echo '[YUNDAO 1/4] Using project interpreter (no Conda env requested)'
  fi
  cd ${PROJ_DIR_Q}
  ${PYTHON_BIN_Q} --version

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
        --master-addr ${MASTER_ADDR_Q} --master-port ${MASTER_PORT_TS} \
        --num-nodes ${NUM_HOSTS} --node-rank ${VC_TASK_INDEX} \
        --timezone ${RUN_TIMEZONE_Q} --timestamp ${RUN_TIMESTAMP_Q} \
        --timeout ${TIMESTAMP_TIMEOUT_Q})
    fi
    OUT_DIR=${LOG_ROOT_Q}/${RUN_PREFIX_Q}\${RUN_TIMESTAMP}
  fi

  YUNDAO_STAGE=output
  mkdir -p \"\${OUT_DIR}\"
  LOG_FILE=\"\${OUT_DIR}/yundao_node_${VC_TASK_INDEX}.log\"
  exec > >(tee -a \"\${LOG_FILE}\" >&3) 2>&1
  SNAPSHOT_DIR=\"\${OUT_DIR}/launch/node_${VC_TASK_INDEX}\"
  mkdir -p \"\${SNAPSHOT_DIR}\"
  cp ${BOOTSTRAP_LOG_Q} \"\${SNAPSHOT_DIR}/startup.log\"
  cp ${SCRIPT_PATH_Q} \"\${SNAPSHOT_DIR}/yundao_train_gae_re10k_lora.sh\"
  cp ${ENV_HELPER_Q} \"\${SNAPSHOT_DIR}/roma_environment.sh\"
  cp ${TIMESTAMP_HELPER_Q} \"\${SNAPSHOT_DIR}/sequence_run_timestamp.py\"
  cp ${CONFIG_Q} \"\${SNAPSHOT_DIR}/train_config.yaml\"
  cp examples/wanvideo/model_training/train_with_accelerate_gae_nvs.py \"\${SNAPSHOT_DIR}/\"
  cp examples/wanvideo/model_training/WanGAENVSTrainingModule.py \"\${SNAPSHOT_DIR}/\"
  cp utils/gae_codec.py \"\${SNAPSHOT_DIR}/\"
  echo \"[YUNDAO 3/4] output=\${OUT_DIR}; log=\${LOG_FILE}\"

  YUNDAO_STAGE=accelerate
  echo '[YUNDAO 4/4] Launching accelerate with DDP; all ${NUM_HOSTS} nodes must join'
  MULTI_GPU_FLAG=\"\"
  if (( WORLD_SIZE > 1 )); then MULTI_GPU_FLAG=\"--multi_gpu\"; fi
  SCRIPT_ARGS=(--config ${CONFIG_Q} \"output_path=\${OUT_DIR}\" ${SEED_ARGS_Q} ${DATASET_ARGS_Q} ${RESUME_ARGS_Q} ${EXTRA_ARGS_Q})
  echo \"[YUNDAO] SCRIPT_ARGS: \${SCRIPT_ARGS[*]}\"
  ${PYTHON_BIN_Q} -u -m accelerate.commands.launch \\
    \${MULTI_GPU_FLAG} \\
    --num_machines ${NUM_HOSTS} \\
    --machine_rank ${VC_TASK_INDEX} \\
    --main_process_ip ${MASTER_ADDR_Q} \\
    --main_process_port ${MASTER_PORT} \\
    --num_processes ${WORLD_SIZE} \\
    -m examples.wanvideo.model_training.train_with_accelerate_gae_nvs \\
    \"\${SCRIPT_ARGS[@]}\"
  echo '[YUNDAO DONE] Training command completed'
"
