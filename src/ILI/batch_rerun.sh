#!/usr/bin/env bash
# Step 5 — execute every generated ILI rerun script (generation + evaluation).
#
#   ./batch_rerun.sh                       # sequential, default index
#   MAXJOBS=4 ./batch_rerun.sh             # 4 reruns at once (reruns are light)
#   ./batch_rerun.sh path/to/index.txt     # custom index
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX="${1:-$HERE/rerun_scripts/index.txt}"
MAXJOBS="${MAXJOBS:-1}"

if [[ ! -f "$INDEX" ]]; then
  echo "Index not found: $INDEX (run generate_rerun_scripts.py first)" >&2
  exit 1
fi

mapfile -t SCRIPTS < <(grep -v '^[[:space:]]*$' "$INDEX")
echo "Running ${#SCRIPTS[@]} rerun scripts from $INDEX (MAXJOBS=$MAXJOBS)"

# Background subshells can't update a parent variable, so record failures in a
# shared file (append of one short line is atomic) that works in both modes.
FAILED_LIST="$(mktemp)"
trap 'rm -f "$FAILED_LIST"' EXIT
run_one() {
  local script="$1"
  echo "[$(date '+%F %T')] START  $(basename "$script")"
  if bash "$script"; then
    echo "[$(date '+%F %T')] DONE   $(basename "$script")"
  else
    local code=$?
    echo "[$(date '+%F %T')] FAIL   $(basename "$script") (exit $code)" >&2
    echo "$script" >> "$FAILED_LIST"
  fi
}

if [[ "$MAXJOBS" -le 1 ]]; then
  for s in "${SCRIPTS[@]}"; do
    run_one "$s"
  done
else
  for s in "${SCRIPTS[@]}"; do
    run_one "$s" &
    while (( $(jobs -rp | wc -l) >= MAXJOBS )); do wait -n; done
  done
  wait
fi

fail=$(wc -l < "$FAILED_LIST" | tr -d ' ')
if (( fail > 0 )); then
  echo "Failed scripts ($fail):"; cat "$FAILED_LIST"
fi
echo "All rerun scripts finished (failures=$fail)."
exit $(( fail > 0 ? 1 : 0 ))
