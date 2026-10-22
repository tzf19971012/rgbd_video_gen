#!/bin/bash
# 等 eval_testset.py 分片全部结束后，用 checkpoint-step-800 跑 infer_video_demo。
cd /mnt/sfs_turbo/t00845730/rgbd_video_gen || exit 1
export PATH="/data/l00580360/miniconda3/envs/wan14b/lib/python3.10/site-packages/imageio_ffmpeg/binaries:$PATH"

echo "[infer_demo] $(date '+%F %T') waiting for eval_testset.py shards to finish..."
while pgrep -f "eval_script/eval_testset.py" > /dev/null; do
    sleep 60
done
echo "[infer_demo] $(date '+%F %T') eval finished, starting inference"

.venv/bin/python -u -m utils.infer_video_demo \
    --config logs/multi4_nvs_lora20261020_094334/args.yaml \
    --checkpoint logs/multi4_nvs_lora20261020_094334/checkpoint-step-800 \
    --output_dir logs/multi4_nvs_lora20261020_094334/infer_video_s800 \
    --sample_idx 0,500,4500,9000
status=$?
echo "[infer_demo] $(date '+%F %T') inference exited with status ${status}"
