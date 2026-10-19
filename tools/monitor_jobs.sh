#!/bin/bash
# Background job monitor: checks SLURM jobs every CHECK_INTERVAL seconds.
# On failure, writes a report to REPORT_FILE for later inspection.
# Usage: bash tools/monitor_jobs.sh <job_id1> [job_id2] ...

set -uo pipefail

JOBS="${*}"
CHECK_INTERVAL=120  # seconds between checks
PROJECT_DIR="${DVD_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="${PROJECT_DIR}/logs"
REPORT_FILE="${LOG_DIR}/monitor_report.txt"

if [[ -z "${JOBS}" ]]; then
  echo "Usage: $0 <job_id1> [job_id2] ..."
  exit 1
fi

JOB_IDS=$(echo "${JOBS}" | tr ' ' ',')
echo "[$(date)] Monitor started for jobs: ${JOB_IDS}" | tee "${REPORT_FILE}"

declare -A REPORTED

while true; do
  # Get current status for all jobs
  QUEUE=$(squeue -j "${JOB_IDS}" --format="%i %T %R" --noheader 2>/dev/null)
  RUNNING_IDS=()
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    JID=$(echo "$line" | awk '{print $1}')
    STATE=$(echo "$line" | awk '{print $2}')
    REASON=$(echo "$line" | awk '{print $3}')
    RUNNING_IDS+=("$JID")
    echo "[$(date)] Job ${JID}: ${STATE} (${REASON})"
  done <<< "${QUEUE}"

  # Check each requested job
  for JID in $(echo "${JOB_IDS}" | tr ',' ' '); do
    if [[ -n "${REPORTED[$JID]:-}" ]]; then
      continue  # already reported
    fi

    # If job is not in queue anymore, it either completed or failed
    if ! echo "${QUEUE}" | grep -q "^${JID} "; then
      # Find log files
      OUT_FILE=$(ls -t "${LOG_DIR}"/*_${JID}.out 2>/dev/null | head -1)
      ERR_FILE=$(ls -t "${LOG_DIR}"/*_${JID}.err 2>/dev/null | head -1)

      # Check for FAILED indicators
      FAILED=0
      if [[ -n "${ERR_FILE}" ]]; then
        if grep -q "FAILED\|Error\|Traceback\|RuntimeError\|ValueError" "${ERR_FILE}" 2>/dev/null; then
          FAILED=1
        fi
      fi
      if [[ -n "${OUT_FILE}" ]]; then
        if grep -q "FAILED\|Error\|Traceback\|RuntimeError\|ValueError" "${OUT_FILE}" 2>/dev/null; then
          FAILED=1
        fi
      fi

      {
        echo ""
        echo "========================================"
        echo "[$(date)] Job ${JID} LEFT QUEUE"
        if [[ "${FAILED}" -eq 1 ]]; then
          echo "STATUS: FAILED"
          echo "--- Last 50 lines of stdout (${OUT_FILE}) ---"
          tail -50 "${OUT_FILE}" 2>/dev/null || echo "(no stdout)"
          echo "--- Last 80 lines of stderr (errors only) ---"
          grep -E "Error|Traceback|RuntimeError|ValueError|size mismatch|FAILED" \
            "${ERR_FILE}" 2>/dev/null | tail -80 || echo "(no errors in stderr)"
        else
          echo "STATUS: COMPLETED (or no error detected)"
          echo "--- Last 20 lines of stdout ---"
          tail -20 "${OUT_FILE}" 2>/dev/null || echo "(no stdout)"
        fi
        echo "========================================"
      } | tee -a "${REPORT_FILE}"

      REPORTED[$JID]=1
    fi
  done

  # If all jobs have been reported, exit
  TOTAL=$(echo "${JOB_IDS}" | tr ',' '\n' | wc -l)
  if [[ "${#REPORTED[@]}" -ge "${TOTAL}" ]]; then
    echo "[$(date)] All jobs finished. Monitor exiting." | tee -a "${REPORT_FILE}"
    break
  fi

  sleep "${CHECK_INTERVAL}"
done
