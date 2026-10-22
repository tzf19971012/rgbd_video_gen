#!/usr/bin/env bash

ROMA_ENV_KEYS=(
  VC_WORKER_HOSTS VC_TASK_INDEX MA_NUM_HOSTS MA_NUM_GPUS
  HCCL_IF_BASE_PORT HCCL_HOST_SOCKET_PORT_RANGE HCCL_NPU_SOCKET_PORT_RANGE
  HCCL_SOCKET_IFNAME HCCL_SOCKET_FAMILY GLOO_SOCKET_IFNAME
  HCCL_LOGIC_SUPERPOD_ID HCCL_INTER_HCCS_DISABLE HCCL_INTRA_ROCE_ENABLE
  HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT HCCL_EVENT_TIMEOUT
  HCCL_RDMA_TIMEOUT HCCL_RDMA_RETRY_CNT HCCL_RDMA_TC HCCL_RDMA_SL
  HCCL_RDMA_QPS_PER_CONNECTION HCCL_RDMA_QP_PORT_CONFIG_PATH
  HCCL_WHITELIST_DISABLE HCCL_WHITELIST_FILE HCCL_ALGO HCCL_BUFFSIZE
  HCCL_OP_EXPANSION_MODE HCCL_OP_RETRY_ENABLE HCCL_DETERMINISTIC
  ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES NPU_VISIBLE_DEVICES
  ASCEND_DEVICE_ID ASCEND_GLOBAL_LOG_LEVEL ASCEND_SLOG_PRINT_TO_STDOUT
  ASCEND_PROCESS_LOG_PATH ASCEND_WORK_PATH ASCEND_HOME_PATH
  ASCEND_HOME ASCEND_INSTALL_PATH CANN_INSTALL_PATH
  ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME ASCEND_CUSTOM_OPP_PATH
  RANK_TABLE_FILE RANK_TABLE_FILE_V_1_0
  TORCH_HCCL_DEBUG_INFO_TEMP_FILE TORCH_HCCL_ASYNC_ERROR_HANDLING
  TORCH_HCCL_BLOCKING_WAIT TORCH_HCCL_DUMP_ON_TIMEOUT
  TORCH_HCCL_TRACE_BUFFER_SIZE TORCH_HCCL_DESYNC_DEBUG
  TORCH_HCCL_ENABLE_MONITORING TORCH_HCCL_HEARTBEAT_TIMEOUT_SEC
  ROBOT_SEQUENCE_STARTUP_TIMEOUT ROBOT_SEQUENCE_HCCL_CHECK_ONLY
)
ROMA_PATH_KEYS=(PATH LD_LIBRARY_PATH PYTHONPATH)

roma_capture_environment() {
  local name
  ROMA_ENV_ASSIGNMENTS=()
  ROMA_PATH_ASSIGNMENTS=()
  for name in "${ROMA_ENV_KEYS[@]}"; do
    if [[ -v ${name} ]]; then ROMA_ENV_ASSIGNMENTS+=("${name}=${!name}"); fi
  done
  for name in "${ROMA_PATH_KEYS[@]}"; do
    if [[ -v ${name} ]]; then ROMA_PATH_ASSIGNMENTS+=("${name}=${!name}"); fi
  done
}

roma_restore_environment() {
  local assignment name value current part merged paths_first=0 combined
  local -a parts
  if [[ "${1-}" == --platform-paths-first ]]; then paths_first=1; shift; fi
  for assignment in "$@"; do
    name="${assignment%%=*}"
    value="${assignment#*=}"
    if [[ " ${ROMA_ENV_KEYS[*]} " == *" ${name} "* ]]; then
      export "${name}=${value}"
    elif [[ " ${ROMA_PATH_KEYS[*]} " == *" ${name} "* ]]; then
      current="${!name-}"
      merged=""
      combined="${current}:${value}"
      if (( paths_first )); then combined="${value}:${current}"; fi
      IFS=: read -r -a parts <<< "${combined}"
      for part in "${parts[@]}"; do
        [[ -n "${part}" ]] || continue
        if [[ ":${merged}:" != *":${part}:"* ]]; then
          merged="${merged:+${merged}:}${part}"
        fi
      done
      export "${name}=${merged}"
    else
      echo "Refusing unlisted ROMA environment key: ${name}" >&2
      return 1
    fi
  done
}

roma_log_environment() {
  local stage="$1" name
  printf '[ROMA ENV] stage=%q uid=%s\n' "${stage}" "${EUID}"
  for name in "${ROMA_ENV_KEYS[@]}" "${ROMA_PATH_KEYS[@]}"; do
    printf '[ROMA ENV] %s=%q\n' "${name}" "${!name-<unset>}"
  done
  while IFS= read -r name; do
    if [[ " ${ROMA_ENV_KEYS[*]} " != *" ${name} "* ]]; then
      printf '[ROMA ENV WARNING] unlisted key not explicitly forwarded: %s\n' "${name}"
    fi
  done < <(compgen -e | LC_ALL=C sort | grep -E '^(HCCL_|TORCH_HCCL_|ASCEND_)' || true)
}
