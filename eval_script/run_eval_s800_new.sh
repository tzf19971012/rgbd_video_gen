#!/bin/bash
# Evaluate checkpoint-step-800 of run multi4_nvs_lora20261021_234614 (vggt view
# sampling) on the same bmvs + re10k test sets as the s800 runs of
# multi4_nvs_lora20261020_094334, for apples-to-apples comparison.
# Waits until checkpoint-step-800/model.safetensors exists, then runs both
# evals sequentially on a single NPU.
# Usage: run_eval_s800_new.sh <npu_id>
set -u
cd /mnt/sfs_turbo/t00845730/rgbd_video_gen || exit 1
export PATH="/data/l00580360/miniconda3/envs/wan14b/lib/python3.10/site-packages/imageio_ffmpeg/binaries:$PATH"

NPU=${1:?npu id}
RUN=logs/multi4_nvs_lora20261021_234614
CKPT=$RUN/checkpoint-step-800

echo "[wait] waiting for $CKPT/model.safetensors ..."
while [ ! -f "$CKPT/model.safetensors" ]; do
    sleep 120
done
echo "[wait] checkpoint ready at $(date)"

run_one() {
    DS=$1
    if [ "$DS" = "re10k" ]; then
        MANIFEST=/mnt/sfs_turbo/t00845730/re10k/manifest_test.jsonl
        ROOT=/mnt/sfs_turbo/t00845730/re10k/extracted
    else
        MANIFEST=/mnt/sfs_turbo/t00845730/data/blendedmvs/manifest_test.jsonl
        ROOT=/mnt/sfs_turbo/t00845730/data/blendedmvs/extracted
    fi
    export ASCEND_RT_VISIBLE_DEVICES=$NPU
    .venv/bin/python -u eval_script/eval_testset.py \
        --run-config $RUN/args.yaml \
        --ckpt $CKPT \
        --manifest "$MANIFEST" \
        --kubric4d-root "$ROOT" \
        --output "logs/test_eval/${DS}_test_s800_new" \
        --num-samples 50 --start-index 6 --batch-size 1 --samples-per-scene 1 \
        --view-seed 1234 --shard 0 --num-shards 1 --save-vis 44 \
        --dataloader-num-workers 2 --eval-step-tag 800
}

echo "[eval] bmvs start $(date)"
run_one bmvs
echo "[eval] bmvs exit=$? $(date)"
echo "[eval] re10k start $(date)"
run_one re10k
echo "[eval] re10k exit=$? $(date)"
