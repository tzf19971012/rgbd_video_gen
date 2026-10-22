#!/bin/bash
# Evaluate checkpoint-step-1250 on a test set (re10k or bmvs), replicating the
# exact settings of the s1000 runs in logs/test_eval.
# Usage: run_eval_s1250.sh <re10k|bmvs> <npu_id>
set -u
cd /mnt/sfs_turbo/t00845730/rgbd_video_gen || exit 1
export PATH="/data/l00580360/miniconda3/envs/wan14b/lib/python3.10/site-packages/imageio_ffmpeg/binaries:$PATH"

DS=${1:?re10k or bmvs}
NPU=${2:?npu id}

if [ "$DS" = "re10k" ]; then
    MANIFEST=/mnt/sfs_turbo/t00845730/re10k/manifest_test.jsonl
    ROOT=/mnt/sfs_turbo/t00845730/re10k/extracted
else
    MANIFEST=/mnt/sfs_turbo/t00845730/data/blendedmvs/manifest_test.jsonl
    ROOT=/mnt/sfs_turbo/t00845730/data/blendedmvs/extracted
fi

export ASCEND_RT_VISIBLE_DEVICES=$NPU
exec .venv/bin/python -u eval_script/eval_testset.py \
    --run-config logs/multi4_nvs_lora20261021_080018/args.yaml \
    --ckpt logs/multi4_nvs_lora20261021_080018/checkpoint-step-1250 \
    --manifest "$MANIFEST" \
    --kubric4d-root "$ROOT" \
    --output "logs/test_eval/${DS}_test_s1250" \
    --num-samples 50 --start-index 6 --batch-size 1 --samples-per-scene 1 \
    --view-seed 1234 --shard 0 --num-shards 1 --save-vis 44 \
    --dataloader-num-workers 2 --eval-step-tag 1250
