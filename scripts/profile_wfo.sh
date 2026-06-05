#!/usr/bin/env bash
# Profile a single-process WFO run with cProfile.
#
# IMPORTANT: this profiles scripts/run_walk_forward_optimization.py DIRECTLY, not
# `ggt.py research`. `run_research` always shells the WFO compute out to that worker
# script via subprocess.run (even with --no-parallel), so cProfile on `ggt.py research`
# only ever sees the parent blocked in waitpid (~99.8% of time) and zero backtest
# frames. Profiling the worker in-process is the only way cProfile sees the real work.
#
# Writes a timestamped .prof under results/profiling/ and prints a summary via
# analyze_profile.py.
#
# Usage (inside the container, or via `docker compose run --rm`):
#   scripts/profile_wfo.sh "BTC-USD,ETH-USD,SOL-USD"
#
# Env overrides:
#   SYMBOLS     comma-separated symbol list (default BTC-USD,ETH-USD,SOL-USD)
#   EXCHANGE    data/execution venue (default binanceus)
#   TAG         label appended to the output filename (default "primary")
#   START_DATE  WFO training start (default 2023-06-05)
#   END_DATE    backtest end (default today)
set -euo pipefail

SYMBOLS="${1:-${SYMBOLS:-BTC-USD,ETH-USD,SOL-USD}}"
EXCHANGE="${EXCHANGE:-binanceus}"
TAG="${TAG:-primary}"
START_DATE="${START_DATE:-2023-06-05}"
END_DATE="${END_DATE:-$(date +%F)}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/profiling

TS="$(date +%Y%m%d_%H%M%S)"
OUT="results/profiling/wfo_${TS}_${TAG}.prof"
RUN_DIR="results/profiling/run_${TS}_${TAG}"
mkdir -p "$RUN_DIR"

echo "[profile_wfo] venue=$EXCHANGE symbols=$SYMBOLS window=$START_DATE..$END_DATE -> $OUT"
EXCHANGE="$EXCHANGE" python -m cProfile -o "$OUT" \
  scripts/run_walk_forward_optimization.py \
  --symbols "$SYMBOLS" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --phase1 --no-progress \
  --run-dir "$(cd "$RUN_DIR" && pwd)" \
  --pipeline-stage research

echo "[profile_wfo] analyzing $OUT"
python scripts/analyze_profile.py "$OUT" --top 30 --markdown "${OUT%.prof}.md"
echo "[profile_wfo] done: $OUT"
