#!/usr/bin/env bash
# Fresh-process cold-start matrix: import vectorbt + the trader import chain + first
# simulate_signals / simulate_weights, each run with an EMPTY NUMBA_CACHE_DIR ("cold") and
# then again with the cache that run just filled ("warm"). One JSON line per run.
#
# Usage: run_coldstart.sh OUT.jsonl NEW_PY OLD_PY ORIG_SRC_DIR REPO [REPS]
set -euo pipefail
OUT=$1; NEWPY=$2; OLDPY=$3; ORIG=$4; REPO=$5; REPS=${6:-2}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_ROOT="$(mktemp -d)"
: > "$OUT"
one() { # label py src mode cachedir [extra env...]
    local label=$1 py=$2 src=$3 mode=$4 cdir=$5; shift 5
    local line
    line=$(cd "$REPO" && env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$src" NUMBA_CACHE_DIR="$cdir" "$@" \
        "$py" "$HERE/coldstart_probe.py" "$mode" 2>/dev/null | tail -1)
    echo "{\"label\":\"$label\",\"load1\":\"$(cut -d' ' -f1 /proc/loadavg)\",\"r\":$line}" >> "$OUT"
}
for i in $(seq 1 "$REPS"); do
    one "new_rust_cold_$i"  "$NEWPY" "$REPO/src" rust  "$CACHE_ROOT/rust$i"
    one "new_rust_warm_$i"  "$NEWPY" "$REPO/src" rust  "$CACHE_ROOT/rust$i"
    one "new_numba_cold_$i" "$NEWPY" "$REPO/src" numba "$CACHE_ROOT/numba$i"
    one "new_numba_warm_$i" "$NEWPY" "$REPO/src" numba "$CACHE_ROOT/numba$i"
    one "old_numba_cold_$i" "$OLDPY" "$ORIG"     numba "$CACHE_ROOT/old$i"
    one "old_numba_warm_$i" "$OLDPY" "$ORIG"     numba "$CACHE_ROOT/old$i"
done
rm -rf "$CACHE_ROOT"
echo DONE >> "$OUT"
