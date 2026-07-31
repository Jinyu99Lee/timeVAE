#!/usr/bin/env bash
# Step 3 — execute every generated ILI run script in order.
#
# Each run script internally parallelises its own HPO grid via --gpu-slots, so
# the default here runs the run scripts ONE AT A TIME. Set MAXJOBS>1 to run that
# many run scripts concurrently (only if your GPUs can hold several grids).
#
#   ./batch_run.sh                       # sequential, default index
#   MAXJOBS=2 ./batch_run.sh             # 2 subsets at once
#   ./batch_run.sh path/to/index.txt     # custom index
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX="${1:-$HERE/run_scripts/index.txt}"
MAXJOBS="${MAXJOBS:-1}"

if [[ ! -f "$INDEX" ]]; then
  echo "Index not found: $INDEX (run generate_run_scripts.py first)" >&2
  exit 1
fi

mapfile -t SCRIPTS < <(grep -v '^[[:space:]]*$' "$INDEX")
echo "Running ${#SCRIPTS[@]} run scripts from $INDEX (MAXJOBS=$MAXJOBS)"

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
echo "All run scripts finished (failures=$fail)."
exit $(( fail > 0 ? 1 : 0 ))
