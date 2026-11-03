#!/usr/bin/env bash
# List the files Claude Code edited in a session, with a per-file edit count.
#
# `git status` cannot answer this: it does not distinguish who made a change,
# and it is blind to anything outside the repo (~/.claude/CLAUDE.md, memory
# files, configs elsewhere on the box).  The session transcript records every
# tool call with its arguments, so it is the authoritative source.
#
# Usage:
#   tools/session_edits.sh                 # most recent session for this repo
#   tools/session_edits.sh -n 3            # 3rd most recent
#   tools/session_edits.sh -a              # every session for this repo
#   tools/session_edits.sh <path-to.jsonl> # a specific transcript
#   tools/session_edits.sh -d              # also show the diff-relevant details
#
# CAVEAT: this reports Edit/Write/NotebookEdit tool calls only.  Files written by
# a Bash command (`sed -i`, a build step, a job writing to logs/) do not appear
# here -- no transcript field records them, because the tool call was a shell
# string, not a file path.  Cross-check with `git status` for in-repo changes.

set -euo pipefail

command -v jq >/dev/null || { echo "need jq" >&2; exit 1; }

# Claude Code stores transcripts under a slug of the project's absolute path
# with '/', '.' and '_' all collapsed to '-', so e.g.
# /home/alice/my-project becomes -home-alice-my-project.
project_dir="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
slug="$(printf '%s' "$project_dir" | sed 's/[\/._]/-/g')"
hist_dir="${HOME}/.claude/projects/${slug}"

nth=1
detail=0
all=0
transcript=""
while (( $# )); do
  case "$1" in
    -n) nth="$2"; shift 2 ;;
    -a) all=1; shift ;;
    -d) detail=1; shift ;;
    -*) echo "unknown flag $1" >&2; exit 1 ;;
    *)  transcript="$1"; shift ;;
  esac
done

if [[ -n "${transcript}" ]]; then
  files=("${transcript}")
elif (( all )); then
  mapfile -t files < <(ls -t "${hist_dir}"/*.jsonl 2>/dev/null)
else
  if [[ ! -d "${hist_dir}" ]]; then
    # The slugging rule is an implementation detail and could change; say what
    # was looked for rather than just "not found".
    echo "no transcript dir ${hist_dir} (for ${project_dir})" >&2
    echo "available:" >&2
    ls "${HOME}/.claude/projects" >&2
    exit 1
  fi
  mapfile -t all_files < <(ls -t "${hist_dir}"/*.jsonl 2>/dev/null)
  (( ${#all_files[@]} >= nth )) || { echo "only ${#all_files[@]} session(s)" >&2; exit 1; }
  files=("${all_files[$((nth - 1))]}")
fi

for f in "${files[@]}"; do
  echo "=== $(basename "$f" .jsonl)  ($(date -r "$f" '+%Y-%m-%d %H:%M'))"
  # One record per Edit/Write; group by path so a file edited five times shows
  # up once with a count, which is what makes the output scannable.
  jq -r 'select(.message.content) | .message.content[]?
         | select(.type=="tool_use")
         | select(.name=="Edit" or .name=="Write" or .name=="NotebookEdit")
         | "\(.name)\t\(.input.file_path)"' "$f" \
  | awk -F'\t' '
      { n[$2]++; if (!($2 in first)) { first[$2]=++order; kind[$2]=$1 }
        else if (kind[$2] != $1) kind[$2]="Edit+Write" }
      END { for (p in n) printf "%d\t%s\t%d\t%s\n", first[p], kind[p], n[p], p }' \
  | sort -n | cut -f2- \
  | awk -F'\t' '{printf "  %-11s %2d x  %s\n", $1, $2, $3}'

  if (( detail )); then
    echo "  --- bash commands that may have written files (not tracked above):"
    jq -r 'select(.message.content) | .message.content[]?
           | select(.type=="tool_use" and .name=="Bash") | .input.command' "$f" \
    | grep -aoE '(sed -i|tee|>>?[[:space:]]*[^[:space:]|&;]+|mv |cp |rm |mkdir )[^;|&]*' \
    | sed 's/^/    /' | sort -u | head -30 || echo "    (none)"
  fi
  echo
done
