#!/usr/bin/env bash
# 定时清理实验目录下的 checkpoint-step-* 目录。
# 保留策略：
#   1) step 最大的 KEEP 个 checkpoint
#   2) best_* 软链指向的 checkpoint
#   3) 最近 MTIME_GUARD_MIN 分钟内被写入的 checkpoint（防止误删正在写的）
# 用法: cleanup_ckpt.sh [实验目录] [保留个数]
# 例子: cleanup_ckpt.sh /mnt/sfs_turbo/t00845730/rgbd_video_gen/logs/gae_re10k_rgb_aux20261029_140212 2
set -euo pipefail

EXP_DIR="${1:-/mnt/sfs_turbo/t00845730/rgbd_video_gen/logs/gae_re10k_rgb_aux20261029_140212}"
KEEP="${2:-2}"
MTIME_GUARD_MIN="${CLEANUP_MTIME_GUARD_MIN:-30}"
LOG_FILE="${EXP_DIR%/}/ckpt_cleanup.log"

[[ -d "${EXP_DIR}" ]] || { echo "cleanup_ckpt: no such dir: ${EXP_DIR}" >&2; exit 0; }

# 收集 best_* 软链指向的 checkpoint 路径
declare -A PROTECT=()
for link in "${EXP_DIR}"/best_*; do
  [[ -L "${link}" ]] || continue
  tgt="$(readlink -f "${link}" 2>/dev/null || true)"
  [[ -n "${tgt}" ]] && PROTECT["${tgt}"]=1
done

# 按 step 升序列出 checkpoint-step-* 目录（只认数字后缀）
mapfile -t CKPTS < <(find "${EXP_DIR}" -maxdepth 1 -type d -name 'checkpoint-step-*' \
  | awk -F'checkpoint-step-' '$NF ~ /^[0-9]+$/ {print $NF, $0}' | sort -n | cut -d' ' -f2-)

TOTAL=${#CKPTS[@]}
(( TOTAL > KEEP )) || exit 0

now=$(date +%s)
deleted=0
freed=0
for (( i=0; i<TOTAL-KEEP; i++ )); do
  d="${CKPTS[i]}"
  [[ -n "${PROTECT[${d}]:-}" ]] && continue
  mtime=$(stat -c %Y "${d}")
  (( now - mtime < MTIME_GUARD_MIN * 60 )) && continue
  size=$(du -sk "${d}" 2>/dev/null | cut -f1)
  rm -rf "${d}"
  deleted=$((deleted + 1))
  freed=$((freed + size))
  echo "$(date '+%F %T') deleted ${d} (freed $((size/1024)) MB)"
done

(( deleted > 0 )) && echo "$(date '+%F %T') total deleted=${deleted} freed=$((freed/1024)) MB" >> "${LOG_FILE}"
exit 0
