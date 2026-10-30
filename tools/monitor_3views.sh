#!/bin/bash
# 阻塞等待 50 场景任务结束(最多 MAX_SEC 秒)，期间每 5 分钟打印一次进度。
LOG=logs/multi4_nvs_lora20261021_234558/re10k50_1img_3views_4800.log
JOB=825041
MAX_SEC=${1:-3300}
START=$(date +%s)
while true; do
    if grep -q "all done" "$LOG" 2>/dev/null; then
        echo "JOB_COMPLETE at $(date '+%F %T')"
        grep -cE "^\[[0-9]+/50\].*done" "$LOG"
        exit 0
    fi
    if ! kill -0 "$JOB" 2>/dev/null; then
        echo "JOB_DIED at $(date '+%F %T') (no 'all done')"
        tail -5 "$LOG"
        exit 1
    fi
    NOW=$(date +%s)
    if [ $((NOW - START)) -ge "$MAX_SEC" ]; then
        echo "STILL_RUNNING at $(date '+%F %T'): $(grep -cE '^\[[0-9]+/50\].*done' "$LOG")/50 scenes"
        exit 2
    fi
    sleep 60
done
