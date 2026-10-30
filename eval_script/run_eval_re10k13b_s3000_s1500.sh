#!/bin/bash
# Bucketed re10k test eval (same protocol as the v2 s2000/s2600 runs):
# rand's args.yaml (view_order=random), 20 scenes, view_seed=1234, 3 buckets.
# Arms: rand@3000, vggt@3000, randbi@1500. Run sequentially on one NPU.
# Usage: run_eval_re10k13b_s3000_s1500.sh <rand|vggt|randbi> <npu_id>
set -u
BASE=/mnt/sfs_turbo/t00845730/rgbd_video_gen
WHICH=${1:?rand or vggt or randbi}
NPU=${2:?npu id}
STEP=${3:?checkpoint step}

# per-arm working dir so concurrent runs don't share ./kernel_meta
WORKDIR=/tmp/opencode/eval_${WHICH}_s${STEP}
mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit 1
export PATH="/data/l00580360/miniconda3/envs/wan14b/lib/python3.10/site-packages/imageio_ffmpeg/binaries:$PATH"

RUN=$BASE/logs/re10k13b_${WHICH}20261023_011303
if [ "$WHICH" = "vggt" ]; then
    RUN=$BASE/logs/re10k13b_vggt20261023_011322
elif [ "$WHICH" = "randbi" ]; then
    RUN=$BASE/logs/re10k13b_randbi20261024_011225
fi

# identical eval config for all arms: rand's args.yaml (view_order=random)
CONFIG=$BASE/logs/re10k13b_rand20261023_011303/args.yaml
MANIFEST=/mnt/sfs_turbo/t00845730/re10k/manifest_test.jsonl
ROOT=/mnt/sfs_turbo/t00845730/re10k/extracted
OUT=$BASE/logs/test_eval/re10k13b_${WHICH}_s${STEP}_v2

export ASCEND_RT_VISIBLE_DEVICES=$NPU
exec $BASE/.venv/bin/python -u $BASE/eval_script/eval_testset_bucketed.py \
    --run-config "$CONFIG" \
    --ckpt "$RUN/checkpoint-step-$STEP" \
    --manifest "$MANIFEST" \
    --kubric4d-root "$ROOT" \
    --output "$OUT" \
    --num-scenes 20 --view-seed 1234 --save-vis 6 \
    --dataloader-num-workers 2 --eval-step-tag $STEP
