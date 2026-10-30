#!/bin/bash
# job1 (PID=$1, 50场景3视角) 结束后: 等 NPU 空闲 -> 补跑 orbit demo3view_4800。
JOB_PID=$1
cd /mnt/sfs_turbo/t00845730/rgbd_video_gen || exit 1
LOG_DIR=logs/multi4_nvs_lora20261021_234558
NEED_MB=56000

while kill -0 "$JOB_PID" 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') job1 finished"

while true; do
    DEV=$(npu-smi info | awk '
        /\| *[0-9]+ +910B3/ {dev=$2}
        /\/ *65536/ {for(i=1;i<=NF;i++) if ($i ~ /\/$/ && $(i+1) == 65536) {sub(/\/$/,"",$i); print dev, $i}}' | \
    while read -r dev used; do
        [[ "$used" =~ ^[0-9]+$ ]] || continue
        if [ $((65536 - used)) -ge "$NEED_MB" ]; then echo "$dev"; break; fi
    done)
    if [ -n "$DEV" ]; then
        echo "$(date '+%F %T') starting orbit demo3view_4800 on NPU $DEV"
        ASCEND_RT_VISIBLE_DEVICES=$DEV exec .venv/bin/python -u -m examples.wanvideo.model_training.eval_rgbd_nvs_orbit \
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
