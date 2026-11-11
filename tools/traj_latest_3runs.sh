#!/usr/bin/env bash
# 测试三个 run 最新 ckpt 的相机轨迹生成能力（与 Nov5 的 traj_step9000/800 同一套命令）。
# 每个 ckpt 跑两个场景: 训练场景 00028da87cc5a4c4 + 留出测试场景 fe2fadf89a84e92a。
# NPU 0 串行执行。
set -u
PROJ=/mnt/sfs_turbo/t00845730/rgbd_video_gen_gae
cd "$PROJ"
PY=/data/l00580360/miniconda3/envs/wan14b/bin/python
source /usr/local/Ascend/cann-8.5.2/set_env.sh
export PYTHONPATH="/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0

TEST_MANIFEST=/mnt/sfs_turbo/t00845730/re10k/manifest_test.jsonl
TEST_CACHE=re10k_test
SCENE=00028da87cc5a4c4
TEST_SCENE=fe2fadf89a84e92a

run() {
    RD=$1; CK=$2
    echo "[$(date '+%F %T')] == $(basename "$RD") $(basename "$CK") scene=$SCENE"
    "$PY" -u -m examples.wanvideo.model_training.infer_trajectory \
        --run_dir "$RD" --checkpoint "$CK" --scene_id "$SCENE"
    echo "[$(date '+%F %T')] == $(basename "$RD") $(basename "$CK") test scene=$TEST_SCENE"
    "$PY" -u -m examples.wanvideo.model_training.infer_trajectory \
        --run_dir "$RD" --checkpoint "$CK" --scene_id "$TEST_SCENE" \
        --manifest "$TEST_MANIFEST" --cache_name "$TEST_CACHE"
}

run logs/gae_patch1_re10k20261105_154929 \
    logs/gae_patch1_re10k20261105_154929/checkpoint-step-10600
run logs/ovf_gae_re10k_patch1_flippe_fixed_nodrop \
    logs/ovf_gae_re10k_patch1_flippe_fixed_nodrop/checkpoint-step-3000
run logs/ovf_geocolor_token_re10k_lora_fixed_20261105_105036 \
    logs/ovf_geocolor_token_re10k_lora_fixed_20261105_105036/checkpoint-step-5600
echo "[$(date '+%F %T')] ALL DONE"
