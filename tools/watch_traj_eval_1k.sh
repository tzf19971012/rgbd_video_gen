#!/usr/bin/env bash
# 每 1000 步 ckpt 自动跑相机轨迹测试（shared_target_noise，场景 00028da87cc5a4c4），
# 指标追加汇总到 logs/traj_eval_shared_noise/summary.csv。
# 启动: nohup setsid bash tools/watch_traj_eval_1k.sh > /dev/null 2>&1 &
# 停止: kill $(cat logs/.traj_eval_watcher.pid)
set -u
PROJ_DIR=/mnt/sfs_turbo/t00845730/rgbd_video_gen_gae
cd "$PROJ_DIR"

PY=/mnt/sfs_turbo/t00845730/rgbd_video_gen/.venv/bin/python
export PYTHONPATH=/mnt/sfs_turbo/t00845730/zhangyiheng/Depth-Anything-3/src

SCENE=00028da87cc5a4c4
RUNS=(ovf_gae_re10k_patch1_flippe_drop ovf_gae_re10k_patch1_concatpixel_drop ovf_gae_re10k_patch1_concatpixel)
OUT_ROOT=logs/traj_eval_shared_noise
SUMMARY="$OUT_ROOT/summary.csv"
WATCH_LOG="$OUT_ROOT/watcher.log"
POLL=180
NPU=2

if [[ -f logs/.traj_eval_watcher.pid ]] && kill -0 "$(cat logs/.traj_eval_watcher.pid)" 2>/dev/null; then
    echo "watcher already running: pid $(cat logs/.traj_eval_watcher.pid)"; exit 1
fi
echo $$ > logs/.traj_eval_watcher.pid

mkdir -p "$OUT_ROOT/state" "$OUT_ROOT/logs"
[[ -f $SUMMARY ]] || echo "ts,run,step,scene,n_frames,psnr,lpips,delta1,rel_aligned,nearest_src_psnr,psnr_minus_copy,beats_copy" >> "$SUMMARY"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WATCH_LOG"; }

while true; do
    for RUN in "${RUNS[@]}"; do
        RD="logs/$RUN"
        # 该 run 最新的 1000 倍数 ckpt
        best=0
        for d in "$RD"/checkpoint-step-*; do
            [[ -d $d && -f $d/model.safetensors ]] || continue
            s=${d##*-}
            (( s % 1000 == 0 )) || continue
            (( s > best )) && best=$s
        done
        (( best > 0 )) || continue
        # 从新到旧找第一个未测的（每轮每 run 最多测一个）
        for ((s = best; s >= 1000; s -= 1000)); do
            CK="$RD/checkpoint-step-$s"
            [[ -d $CK && -f $CK/model.safetensors ]] || continue
            [[ -f $OUT_ROOT/state/${RUN}_s${s}.done ]] && continue
            # ckpt 完整性：20s 内尺寸稳定且 >3GB 才开测，否则下轮再试
            sz1=$(stat -c %s "$CK/model.safetensors")
            sleep 20
            sz2=$(stat -c %s "$CK/model.safetensors")
            [[ $sz1 == $sz2 && $sz1 -gt 3000000000 ]] || continue

            LOGF="$OUT_ROOT/logs/${RUN}_s${s}.log"
            log "testing $RUN step $s on npu $NPU"
            ASCEND_RT_VISIBLE_DEVICES=$NPU "$PY" -u \
                -m examples.wanvideo.model_training.infer_trajectory \
                --run_dir "$RD" --checkpoint "$CK" --scene_id "$SCENE" \
                --shared_target_noise > "$LOGF" 2>&1

            METRICS="$RD/traj_step$s/$SCENE/per_frame_metrics.jsonl"
            if [[ -s $METRICS ]]; then
                "$PY" - "$METRICS" "$RUN" "$s" "$SCENE" "$SUMMARY" <<'PYEOF'
import datetime, json, sys
import numpy as np
fp, run, step, scene, summary = sys.argv[1:6]
rows = [json.loads(l) for l in open(fp)]
def m(k):
    v = [r[k] for r in rows if r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")
psnr, copy = m("tgt_rgb_psnr"), m("nearest_src_psnr")
beats = sum(1 for r in rows if r["tgt_rgb_psnr"] > r["nearest_src_psnr"])
ts = datetime.datetime.now().strftime("%F %T")
with open(summary, "a") as f:
    f.write(f"{ts},{run},{step},{scene},{len(rows)},{psnr:.4f},"
            f"{m('tgt_rgb_lpips'):.4f},{m('tgt_depth_delta1'):.4f},"
            f"{m('tgt_depth_abs_rel_aligned'):.4f},{copy:.4f},"
            f"{psnr - copy:+.4f},{beats}/{len(rows)}\n")
PYEOF
                touch "$OUT_ROOT/state/${RUN}_s${s}.done"
                log "done $RUN step $s -> appended $SUMMARY"
            else
                log "FAILED $RUN step $s (no metrics), will retry next poll"
            fi
            [[ $NPU == 2 ]] && NPU=3 || NPU=2
            break
        done
    done
    sleep $POLL
done
