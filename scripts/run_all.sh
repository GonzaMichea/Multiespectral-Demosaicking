#!/bin/sh
# Train the three networks, then evaluate, plot and compile the poster.
# Usage: scripts/run_all.sh [iters] [data-mode]
set -e
cd "$(dirname "$0")/.."
ITERS=${1:-40000}
MODE=${2:-cave-split}
mkdir -p checkpoints
for pair in imec:fixed_imec random:agnostic osp:fixed_osp; do
  pattern=${pair%%:*}; name=${pair##*:}
  # Train to a temporary file so a partial run is never mistaken for a final checkpoint.
  .venv/bin/python -m msdemosaic.train --pattern "$pattern" --data-mode "$MODE" \
    --iters "$ITERS" --out "checkpoints/_running_$name.pt" > "checkpoints/$name.out" 2>&1
  mv "checkpoints/_running_$name.pt" "checkpoints/$name.pt"
  mv "checkpoints/_running_$name.log.csv" "checkpoints/$name.log.csv"
  # Refresh results and poster after every network, so there is always a usable poster.
  .venv/bin/python -m msdemosaic.evaluate_all --data-mode "$MODE" > checkpoints/evaluate.out 2>&1
  .venv/bin/python -m msdemosaic.figures --data-mode "$MODE" > checkpoints/figures.out 2>&1
  typst compile --root . Poster/poster.typ >> checkpoints/figures.out 2>&1 || true
done
echo done > checkpoints/ALL_DONE
