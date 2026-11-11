#!/usr/bin/env bash
# ovf_gae_re10k_patch1_flippe_fixed_nodrop @ checkpoint-step-4000 相机控制（轨迹）测试
set -u
PROJ=/mnt/sfs_turbo/t00845730/rgbd_video_gen_gae
cd "$PROJ"
PY=/data/l00580360/miniconda3/envs/wan14b/bin/python
source /usr/local/Ascend/cann-8.5.2/set_env.sh
export PYTHONPATH="/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=7

RD=logs/ovf_gae_re10k_patch1_flippe_fixed_nodrop
CK=$RD/checkpoint-step-4000

echo "[$(date '+%F %T')] == scene=00028da87cc5a4c4 (train)"
"$PY" -u -m examples.wanvideo.model_training.infer_trajectory \
    --run_dir "$RD" --checkpoint "$CK" --scene_id 00028da87cc5a4c4

echo "[$(date '+%F %T')] == test scene=fe2fadf89a84e92a"
"$PY" -u -m examples.wanvideo.model_training.infer_trajectory \
    --run_dir "$RD" --checkpoint "$CK" --scene_id fe2fadf89a84e92a \
    --manifest /mnt/sfs_turbo/t00845730/re10k/manifest_test.jsonl \
    --cache_name re10k_test

echo "[$(date '+%F %T')] == lateral probe fe2fadf89a84e92a"
"$PY" -u -m tools.infer_gae_lateral_probe \
    --run_dir "$RD" --checkpoint "$CK" --scene_id fe2fadf89a84e92a

echo "[$(date '+%F %T')] == score lateral probe"
"$PY" -u tools/score_lateral_probe.py "$RD/lateral_probe_step4000/fe2fadf89a84e92a"

echo "[$(date '+%F %T')] ALL DONE"
