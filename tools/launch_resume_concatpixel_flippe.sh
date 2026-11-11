#!/bin/bash
# =============================================================================
# 同时拉起两个续训任务，各占 2 张空闲 NPU：
#   A) concatpixel -> checkpoint-step-2400 续训，全量 7884 场景
#      日志: logs/ovf_gae_re10k_patch1_concatpixel_full/nohup.log
#   B) flippe_drop -> checkpoint-step-2200 续训，去掉 condition dropout
#      日志: logs/ovf_gae_re10k_patch1_flippe_nodrop/nohup.log
#
# 用法:  bash tools/launch_resume_concatpixel_flippe.sh
# 卡号可通过环境变量覆盖: NPUS_A="4,5" NPUS_B="6,7" bash tools/launch_...
# =============================================================================
set -e
cd "$(dirname "$0")/.."

PY=/data/l00580360/miniconda3/envs/wan14b/bin/python
NPUS_A="${NPUS_A:-4,5}"
NPUS_B="${NPUS_B:-6,7}"
CFG_A=train_config/normal_config/gae_ovf_re10k_patch1_concatpixel_full.yaml
CFG_B=train_config/normal_config/gae_ovf_re10k_patch1_flippe_nodrop.yaml
SCRIPT=examples/wanvideo/model_training/train_with_accelerate_gae_nvs.py

for cfg in "$CFG_A" "$CFG_B"; do
    out=$(python - "$cfg" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["output_path"])
EOF
)
    mkdir -p "$out"
done

echo "[A] concatpixel_full on NPU $NPUS_A"
nohup env ASCEND_RT_VISIBLE_DEVICES="$NPUS_A" \
    "$PY" -m accelerate.commands.launch --num_processes 2 --mixed_precision no \
    "$SCRIPT" --config "$CFG_A" \
    > logs/ovf_gae_re10k_patch1_concatpixel_full/nohup.log 2>&1 &
echo "    pid $!"

echo "[B] flippe_nodrop on NPU $NPUS_B"
nohup env ASCEND_RT_VISIBLE_DEVICES="$NPUS_B" \
    "$PY" -m accelerate.commands.launch --num_processes 2 --mixed_precision no \
    "$SCRIPT" --config "$CFG_B" \
    > logs/ovf_gae_re10k_patch1_flippe_nodrop/nohup.log 2>&1 &
echo "    pid $!"

echo "both launched; tail -f logs/*/nohup.log to watch"
