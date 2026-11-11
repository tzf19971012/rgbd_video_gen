#!/usr/bin/env bash
# gae_re10k_concatpixel_span5to35_from8000 @ checkpoint-step-9800 轨迹能力测试
set -u
PROJ=/mnt/sfs_turbo/t00845730/rgbd_video_gen_gae
cd "$PROJ"
PY=/data/l00580360/miniconda3/envs/wan14b/bin/python
source /usr/local/Ascend/cann-8.5.2/set_env.sh
export PYTHONPATH="/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=3

RD=logs/gae_re10k_concatpixel_span5to35_from8000
CK=$RD/checkpoint-step-9800

echo "[$(date '+%F %T')] == scene=00028da87cc5a4c4"
"$PY" -u -m examples.wanvideo.model_training.infer_trajectory \
    --run_dir "$RD" --checkpoint "$CK" --scene_id 00028da87cc5a4c4

echo "[$(date '+%F %T')] == test scene=fe2fadf89a84e92a"
"$PY" -u -m examples.wanvideo.model_training.infer_trajectory \
    --run_dir "$RD" --checkpoint "$CK" --scene_id fe2fadf89a84e92a \
    --manifest /mnt/sfs_turbo/t00845730/re10k/manifest_test.jsonl \
    --cache_name re10k_test
echo "[$(date '+%F %T')] ALL DONE"
