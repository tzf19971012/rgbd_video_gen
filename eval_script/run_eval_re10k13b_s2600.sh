#!/bin/bash
# Same bucketed re10k test protocol as the v2 (step-2000) eval, but with the
# step-2600 checkpoints. Identical config for both arms (rand's args.yaml,
# view_order=random), same 20-scene manifest, view_seed=1234.
# Usage: run_eval_re10k13b_s2600.sh <rand|vggt> <npu_id>
set -u
BASE=/mnt/sfs_turbo/t00845730/rgbd_video_gen
WHICH=${1:?rand or vggt}
NPU=${2:?npu id}

# per-arm working dir so concurrent runs don't share ./kernel_meta
WORKDIR=/tmp/opencode/eval_s2600_${WHICH}
mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit 1
export PATH="/data/l00580360/miniconda3/envs/wan14b/lib/python3.10/site-packages/imageio_ffmpeg/binaries:$PATH"

RUN=/mnt/sfs_turbo/t00845730/rgbd_video_gen/logs/re10k13b_${WHICH}20261023_011322
if [ "$WHICH" = "rand" ]; then
    RUN=/mnt/sfs_turbo/t00845730/rgbd_video_gen/logs/re10k13b_rand20261023_011303
fi

# identical eval config for both: use rand's args.yaml (view_order=random)
CONFIG=$BASE/logs/re10k13b_rand20261023_011303/args.yaml
MANIFEST=/mnt/sfs_turbo/t00845730/re10k/manifest_test.jsonl
ROOT=/mnt/sfs_turbo/t00845730/re10k/extracted
OUT=$BASE/logs/test_eval/re10k13b_${WHICH}_s2600_v2

export ASCEND_RT_VISIBLE_DEVICES=$NPU
exec $BASE/.venv/bin/python -u $BASE/eval_script/eval_testset_bucketed.py \
    --run-config "$CONFIG" \
    --ckpt "$RUN/checkpoint-step-2600" \
    --manifest "$MANIFEST" \
    --kubric4d-root "$ROOT" \
    --output "$OUT" \
    --num-scenes 20 --view-seed 1234 --save-vis 6 \
    --dataloader-num-workers 2 --eval-step-tag 2600
