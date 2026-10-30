#!/bin/bash
# 等 NPU 空闲后依次跑:
#   1) re10k 50 场景 x 单图 x 3视角 (forward/bev/yaw) 网格推理 (checkpoint-step-4800)
#   2) 单场景 orbit demo3view_4800
cd /mnt/sfs_turbo/t00845730/rgbd_video_gen || exit 1

NEED_MB=${NEED_MB:-56000}
LOG_DIR=logs/multi4_nvs_lora20261021_234558
CKPT=$LOG_DIR/checkpoint-step-4800
CFG=$LOG_DIR/args_re10k_infer.yaml
MANIFEST=/mnt/sfs_turbo/t00845730/re10k/manifest_test.jsonl

free_dev() {
    npu-smi info | awk '
        /\| *[0-9]+ +910B3/ {dev=$2}
        /\/ *65536/ {for(i=1;i<=NF;i++) if ($i ~ /\/$/ && $(i+1) == 65536) {sub(/\/$/,"",$i); print dev, $i}}' | \
    while read -r dev used; do
        [[ "$used" =~ ^[0-9]+$ ]] || continue
        if [ $((65536 - used)) -ge "$NEED_MB" ]; then
            echo "$dev"
            break
        fi
    done
}

while true; do
    DEV=$(free_dev)
    if [ -n "$DEV" ]; then
        echo "$(date '+%F %T') NPU $DEV free enough, starting 50-scene 3views job"
        ASCEND_RT_VISIBLE_DEVICES=$DEV .venv/bin/python -u -m tools.infer_re10k_1src_3views \
            --config "$CFG" \
            --checkpoint "$CKPT" \
            --manifest_path "$MANIFEST" \
            --output_dir $LOG_DIR/re10k50_1img_3views_4800 \
            --num_scenes 50
        rc=$?
        echo "$(date '+%F %T') 50-scene job exited rc=$rc"
        if [ $rc -eq 0 ]; then
            DEV=$(free_dev)
            if [ -n "$DEV" ]; then
                echo "$(date '+%F %T') starting orbit demo3view_4800 on NPU $DEV"
                ASCEND_RT_VISIBLE_DEVICES=$DEV .venv/bin/python -u -m examples.wanvideo.model_training.eval_rgbd_nvs_orbit \
                    --config "$CFG" \
                    --checkpoint "$CKPT" \
                    --manifest_path "$MANIFEST" \
                    --split_path /mnt/sfs_turbo/t00845730/re10k/demo_split_indoor.json \
                    --output_dir $LOG_DIR/demo3view_4800 \
                    --tilt_degrees 90 \
                    --yaw_degrees 90,180
                echo "$(date '+%F %T') orbit demo exited rc=$?"
            fi
        fi
        exit 0
    fi
    echo "$(date '+%F %T') no NPU free >= ${NEED_MB}MB yet"
    sleep 120
done
