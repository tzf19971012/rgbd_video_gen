#!/usr/bin/env bash
# Poll the b-sweep every 5 minutes and append a snapshot to a log file.
# Records only -- never cancels or resubmits anything.
cd "$(git rev-parse --show-toplevel)" || exit 1
OUT=logs/b_sweep_watch.log
for _ in $(seq 1 24); do
  {
    echo "===== $(date '+%F %T') ====="
    squeue -u "$USER" -o "%.8i %.12j %.4t %.10M %R" | grep -E 'bmvs_b|JOBID'
    for a in b1 b2 b3 b4 b5 b6 b7 b8; do
      f=$(ls -t logs/bmvs_${a}_*.out 2>/dev/null | head -1)
      [ -z "$f" ] && continue
      e=${f%.out}.err
      printf "%-3s | %-52s | %s | err=%s\n" "$a" \
        "$(grep -oE 'step [0-9]+: loss=[0-9.]+.*lr=[0-9.e-]+' "$f" 2>/dev/null | tail -1)" \
        "$(grep -oE '\[eval step [0-9]+\].*' "$f" 2>/dev/null | tail -1 | cut -c1-90)" \
        "$(grep -ciE 'Traceback|Error:|CUDA out of memory' "$e" 2>/dev/null)"
    done
  } >> "$OUT" 2>&1
  sleep 300
done
