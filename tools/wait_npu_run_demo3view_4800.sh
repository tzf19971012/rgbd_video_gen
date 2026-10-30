#!/bin/bash
# 等待某块 NPU 空闲 HBM >= NEED_MB 后，在空闲卡上跑 demo3view (checkpoint-step-4800)。
cd /mnt/sfs_turbo/t00845730/rgbd_video_gen || exit 1

NEED_MB=${NEED_MB:-56000}
LOG_DIR=logs/multi4_nvs_lora20261021_234558

while true; do
    FREE_DEV=""
    while read -r dev used; do
        [ -z "$dev" ] && continue
        [[ "$used" =~ ^[0-9]+$ ]] || continue
        free=$((65536 - used))
        echo "$(date '+%F %T') npu$dev used=${used}MB free=${free}MB (need ${NEED_MB}MB)"
        if [ "$free" -ge "$NEED_MB" ]; then
            FREE_DEV=$dev
            break
        fi
    done < <(npu-smi info | awk '
        /\| *[0-9]+ +910B3/ {dev=$2}
        /\/ *65536/ {for(i=1;i<=NF;i++) if ($i ~ /\/$/ && $(i+1) == 65536) {sub(/\/$/,"",$i); print dev, $i}}')
    if [ -n "$FREE_DEV" ]; then
        echo "$(date '+%F %T') launching on NPU $FREE_DEV"
        ASCEND_RT_VISIBLE_DEVICES=$FREE_DEV exec .venv/bin/python -u -m examples.wanvideo.model_training.eval_rgbd_nvs_orbit \
            --config $LOG_DIR/args_re10k_infer.yaml \
            --checkpoint $LOG_DIR/checkpoint-step-4800 \
            --manifest_path /mnt/sfs_turbo/t00845730/re10k/manifest_test.jsonl \
            --split_path /mnt/sfs_turbo/t00845730/re10k/demo_split_indoor.json \
            --output_dir $LOG_DIR/demo3view_4800 \
            --tilt_degrees 90 \
            --yaw_degrees 90,180
        exit $?
    fi
    sleep 120
done
