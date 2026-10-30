#!/bin/bash
# A/B speed test: new-code eval on a free NPU, 2 warmup + 10 timed samples.
set -u
cd /mnt/sfs_turbo/t00845730/rgbd_video_gen || exit 1
export PATH="/data/l00580360/miniconda3/envs/wan14b/lib/python3.10/site-packages/imageio_ffmpeg/binaries:$PATH"
export ASCEND_RT_VISIBLE_DEVICES=${NPU:-4}
exec .venv/bin/python -u eval_script/eval_testset.py \
    --run-config logs/multi4_nvs_lora20261021_234558/args.yaml \
    --ckpt logs/multi4_nvs_lora20261021_234558/checkpoint-step-1900 \
    --manifest /mnt/sfs_turbo/t00845730/re10k/manifest_test.jsonl \
    --kubric4d-root /mnt/sfs_turbo/t00845730/re10k/extracted \
    --output logs/test_eval/re10k_speed_newcode \
    --num-samples 12 --start-index 6 --batch-size 1 --samples-per-scene 1 \
    --view-seed 1234 --shard 0 --num-shards 1 --save-vis 0 \
    --dataloader-num-workers 2 --eval-step-tag 1900
