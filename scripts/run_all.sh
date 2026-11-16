#!/bin/sh
# Train the networks; after each one, evaluate, plot and compile the poster.
# Usage: scripts/run_all.sh [iters] [data-mode] [minutes]
#
# Networks (pattern:name:input), most important first. The fixed and the agnostic
# networks share the same input; conventional_imec is the optional mask-less baseline.
NETWORKS="imec:fixed_imec:wb random:agnostic:wb osp:fixed_osp:wb imec:conventional_imec:mosaic"
#
# Resumable: finished networks are skipped and an interrupted one continues from
# its saved state, so the same command can be run in several sessions.
# With [minutes], training pauses at the first validation after that budget.
cd "$(dirname "$0")/.."
ITERS=${1:-40000}
MODE=${2:-cave-split}
MINUTES=${3:-}
START=$(date +%s)
mkdir -p checkpoints
rm -f checkpoints/ALL_DONE
for spec in $NETWORKS; do
  pattern=${spec%%:*}; rest=${spec#*:}; name=${rest%%:*}; input=${rest##*:}
  if [ -f "checkpoints/$name.pt" ]; then
    echo "$name: already trained, skipping"
    continue
  fi
  budget=""
  if [ -n "$MINUTES" ]; then
    left=$(( MINUTES - ( $(date +%s) - START ) / 60 ))
    if [ "$left" -le 0 ]; then echo "time budget used up; rerun to continue"; exit 3; fi
    budget="--max-minutes $left"
  fi
  # Train to a temporary name so a partial run is never mistaken for a final checkpoint.
  .venv/bin/python -m msdemosaic.train --pattern "$pattern" --input "$input" --data-mode "$MODE" \
    --iters "$ITERS" $budget --out "checkpoints/_running_$name.pt" >> "checkpoints/$name.out" 2>&1
  status=$?
  if [ "$status" -eq 3 ]; then echo "$name: paused; rerun the same command to continue"; exit 3; fi
  if [ "$status" -ne 0 ]; then echo "$name: training failed, see checkpoints/$name.out"; exit "$status"; fi
  mv "checkpoints/_running_$name.pt" "checkpoints/$name.pt"
  mv "checkpoints/_running_$name.log.csv" "checkpoints/$name.log.csv"
  echo "$name: done"
  # Refresh results and poster after every network, so there is always a usable poster.
  .venv/bin/python -m msdemosaic.evaluate_all --data-mode "$MODE" > checkpoints/evaluate.out 2>&1
  .venv/bin/python -m msdemosaic.figures --data-mode "$MODE" > checkpoints/figures.out 2>&1
  typst compile --root . Poster/poster.typ >> checkpoints/figures.out 2>&1 || true
done
echo done > checkpoints/ALL_DONE
echo "all networks trained"
