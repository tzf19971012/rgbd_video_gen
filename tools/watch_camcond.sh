#!/usr/bin/env bash
# Wait until both camera-conditioning arms have logged their step-N eval, then
# dump the metrics next to the wan_lora baseline and re-run the conditioning
# probe on the resulting checkpoints.
#
#   bash tools/watch_camcond.sh [STEP] [MAX_WAIT_SECONDS]
#
# Probe args MUST match each arm's encoder config or load_state_dict raises on
# shape (cam_norole has no role_embedding; cam_concat has only channel_scale).
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STEP="${1:-500}"
MAX_WAIT="${2:-18000}"
OUT="logs/camcond_watch_step${STEP}.txt"

: "${DVD_DATA_ROOT:?export DVD_DATA_ROOT first}"

ARMS=(cam_concat cam_norole)
declare -A PROBE_ARGS=(
  [cam_concat]="--mode concat --use-role false --normalize-moment true"
  [cam_norole]="--mode add    --use-role false --normalize-moment false"
)

marker="[eval step ${STEP}]"
waited=0
while :; do
  ready=1
  for arm in "${ARMS[@]}"; do
    grep -qF "$marker" "logs/bmvs_${arm}/train.log" 2>/dev/null || ready=0
  done
  [[ $ready -eq 1 ]] && break
  if [[ $waited -ge $MAX_WAIT ]]; then
    echo "TIMED OUT after ${waited}s waiting for '${marker}'" | tee "$OUT"
    for arm in "${ARMS[@]}"; do
      echo "--- $arm last eval ---"
      grep -F "[eval step" "logs/bmvs_${arm}/train.log" 2>/dev/null | tail -1
    done | tee -a "$OUT"
    exit 1
  fi
  # bail out early if a job died before reaching the eval
  if ! squeue -u "$USER" -h -o "%j" | grep -q "jf_cam_"; then
    echo "NO jf_cam_* JOBS IN QUEUE -- they exited before step ${STEP}" | tee "$OUT"
    for arm in "${ARMS[@]}"; do
      echo "--- $arm tail ---"
      tail -n 15 "logs/bmvs_${arm}/train.log" 2>/dev/null
    done | tee -a "$OUT"
    exit 1
  fi
  sleep 60
  waited=$((waited + 60))
done

{
  echo "=============== eval @ step ${STEP} (frozen set [0,250,500,750]) ==============="
  echo
  echo "--- BASELINE wan_lora (role-collapsed encoder), best recent evals ---"
  grep -F "[eval step" logs/bmvs_jf_pix_wan_lora/train.log 2>/dev/null | tail -2
  echo
  for arm in "${ARMS[@]}"; do
    echo "--- ${arm} ---"
    grep -F "$marker" "logs/bmvs_${arm}/train.log"
    echo
  done

  echo "=============== key comparison ==============="
  printf '%-14s %12s %12s %12s %12s %10s\n' arm src_skill src_corr tgt_skill tgt_corr lpips
  for src in "jf_pix_wan_lora:baseline(6000)" "${ARMS[0]}:${ARMS[0]}" "${ARMS[1]}:${ARMS[1]}"; do
    d=${src%%:*}; label=${src##*:}
    if [[ $d == jf_pix_wan_lora ]]; then
      line=$(grep -F "[eval step" "logs/bmvs_${d}/train.log" | tail -1)
    else
      line=$(grep -F "$marker" "logs/bmvs_${d}/train.log")
    fi
    get() { sed -n "s/.*[[:space:]]$1=\([-0-9.]*\).*/\1/p" <<<"$line"; }
    printf '%-14s %12s %12s %12s %12s %10s\n' "$label" \
      "$(get src_depth_delta1_skill)" "$(get src_depth_struct_corr)" \
      "$(get tgt_depth_delta1_skill)" "$(get tgt_depth_struct_corr)" \
      "$(get tgt_rgb_lpips)"
  done
  echo
  echo "PASS CRITERIA: src_corr>0.2, src_skill>0, tgt_skill>0 (baseline tgt_skill=-0.56)"
  echo

  echo "=============== conditioning probe ==============="
  for arm in "${ARMS[@]}"; do
    ckpt=$(ls -d logs/bmvs_${arm}/checkpoint-step-* 2>/dev/null \
           | sed 's/.*-//' | sort -n | tail -1)
    ckpt="logs/bmvs_${arm}/checkpoint-step-${ckpt}"
    echo "--- ${arm} : ${ckpt} ---"
    if [[ -f "${ckpt}/model.safetensors" ]]; then
      PYTHONPATH=$PWD OMP_NUM_THREADS=8 .venv/bin/python tools/probe_condition_scale.py \
        --checkpoint "$ckpt" --num-scenes 4 ${PROBE_ARGS[$arm]} 2>&1 \
        | grep -E "probing arm|loaded|std \(|magnitude ratio|discriminative|CONTROL|SNR"
    else
      echo "  no model.safetensors yet (max_checkpoints=2 may have rotated it out)"
    fi
    echo
  done
} 2>&1 | tee "$OUT"

echo "written to $OUT"
