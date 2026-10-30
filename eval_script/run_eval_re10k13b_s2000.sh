#!/bin/bash
# Compare step-2000 checkpoints of re10k13b_rand vs re10k13b_vggt on the SAME
# re10k test protocol. Both evals share ONE run-config (rand's args.yaml,
# view_order=random) so the only difference is the weights.
# Usage: run_eval_re10k13b_s2000.sh <rand|vggt> <npu_id>
set -u
cd /mnt/sfs_turbo/t00845730/rgbd_video_gen || exit 1
export PATH="/data/l00580360/miniconda3/envs/wan14b/lib/python3.10/site-packages/imageio_ffmpeg/binaries:$PATH"

WHICH=${1:?rand or vggt}
NPU=${2:?npu id}

RUN=/mnt/sfs_turbo/t00845730/rgbd_video_gen/logs/re10k13b_${WHICH}20261023_011322
if [ "$WHICH" = "rand" ]; then
    RUN=/mnt/sfs_turbo/t00845730/rgbd_video_gen/logs/re10k13b_rand20261023_011303
fi

# identical eval config for both: use rand's args.yaml (view_order=random)
CONFIG=/mnt/sfs_turbo/t00845730/rgbd_video_gen/logs/re10k13b_rand20261023_011303/args.yaml
MANIFEST=/mnt/sfs_turbo/t00845730/re10k/manifest_test.jsonl
ROOT=/mnt/sfs_turbo/t00845730/re10k/extracted
OUT=logs/test_eval/re10k13b_${WHICH}_s2000

export ASCEND_RT_VISIBLE_DEVICES=$NPU
exec .venv/bin/python -u eval_script/eval_testset.py \
    --run-config "$CONFIG" \
    --ckpt "$RUN/checkpoint-step-2000" \
    --manifest "$MANIFEST" \
    --kubric4d-root "$ROOT" \
    --output "$OUT" \
    --num-samples 50 --start-index 6 --batch-size 1 --samples-per-scene 1 \
    --view-seed 1234 --save-vis 8 \
    --dataloader-num-workers 2 --eval-step-tag 2000
