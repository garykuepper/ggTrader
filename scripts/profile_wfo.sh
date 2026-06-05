#!/usr/bin/env bash
# Profile a single-process WFO research run with cProfile.
#
# Runs `ggt.py research` under cProfile with --no-parallel (so the profiler sees the
# whole run; parallel workers are subprocesses cProfile cannot follow). Writes a
# timestamped .prof under results/profiling/ and prints a summary via analyze_profile.py.
#
# Usage (inside the container, or via `docker compose run --rm`):
#   scripts/profile_wfo.sh "BTC-USD,ETH-USD,SOL-USD"
#
# Env overrides:
#   SYMBOLS   comma-separated symbol list (default BTC-USD,ETH-USD,SOL-USD)
#   EXCHANGE  data/execution venue (default binanceus)
#   TAG       label appended to the output filename (default "primary")
set -euo pipefail

SYMBOLS="${1:-${SYMBOLS:-BTC-USD,ETH-USD,SOL-USD}}"
EXCHANGE="${EXCHANGE:-binanceus}"
TAG="${TAG:-primary}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/profiling

TS="$(date +%Y%m%d_%H%M%S)"
OUT="results/profiling/wfo_${TS}_${TAG}.prof"

echo "[profile_wfo] venue=$EXCHANGE symbols=$SYMBOLS -> $OUT"
EXCHANGE="$EXCHANGE" python -m cProfile -o "$OUT" ggt.py research \
  --symbols "$SYMBOLS" --no-parallel --min-volume 0

echo "[profile_wfo] analyzing $OUT"
python scripts/analyze_profile.py "$OUT" --top 30 --markdown "${OUT%.prof}.md"
echo "[profile_wfo] done: $OUT"
