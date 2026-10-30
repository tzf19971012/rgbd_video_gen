#!/bin/bash
# Evaluate checkpoint-step-1600 of run multi4_nvs_lora20261021_234558 on a test
# set (re10k or bmvs), replicating the exact settings of the s1250 runs.
# Supports multi-NPU sharding: each shard writes its own output dir; merge
# afterwards with eval_script/merge_shard_metrics.py.
# Usage: run_eval_s1600.sh <re10k|bmvs> <npu_id> <shard> <num_shards>
set -u
cd /mnt/sfs_turbo/t00845730/rgbd_video_gen || exit 1
export PATH="/data/l00580360/miniconda3/envs/wan14b/lib/python3.10/site-packages/imageio_ffmpeg/binaries:$PATH"

DS=${1:?re10k or bmvs}
NPU=${2:?npu id}
SHARD=${3:-0}
NSHARDS=${4:-1}

if [ "$DS" = "re10k" ]; then
    MANIFEST=/mnt/sfs_turbo/t00845730/re10k/manifest_test.jsonl
    ROOT=/mnt/sfs_turbo/t00845730/re10k/extracted
else
    MANIFEST=/mnt/sfs_turbo/t00845730/data/blendedmvs/manifest_test.jsonl
    ROOT=/mnt/sfs_turbo/t00845730/data/blendedmvs/extracted
fi

OUT=logs/test_eval/${DS}_test_s1600
if [ "$NSHARDS" -gt 1 ]; then
    OUT=${OUT}_shard${SHARD}
fi

export ASCEND_RT_VISIBLE_DEVICES=$NPU
exec .venv/bin/python -u eval_script/eval_testset.py \
    --run-config logs/multi4_nvs_lora20261021_234558/args.yaml \
    --ckpt logs/multi4_nvs_lora20261021_234558/checkpoint-step-1600 \
    --manifest "$MANIFEST" \
    --kubric4d-root "$ROOT" \
    --output "$OUT" \
    --num-samples 50 --start-index 6 --batch-size 1 --samples-per-scene 1 \
    --view-seed 1234 --shard "$SHARD" --num-shards "$NSHARDS" --save-vis 44 \
    --dataloader-num-workers 2 --eval-step-tag 1600
