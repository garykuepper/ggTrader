#!/usr/bin/env bash
# Cold/warm numba-cache matrix, each run in a fresh process. Output: 50_coldstart.jsonl
E=/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S3-evidence
W=/home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0
NEWPY=$W/.venv/bin/python
OLDPY=/home/flynn/ggTrader/.venv/bin/python
OUT=$E/50_coldstart.jsonl
: > "$OUT"
cd "$W"
one() { # label py mode cachedir [extra env...]
    local label=$1 py=$2 mode=$3 cdir=$4; shift 4
    local line
    line=$(env PYTHONPATH=$W/src NUMBA_CACHE_DIR="$cdir" "$@" "$py" "$E/coldstart.py" "$mode" 2>"$E/50_coldstart_$label.err" | tail -1)
    echo "{\"label\":\"$label\",\"load1\":\"$(cut -d' ' -f1 /proc/loadavg)\",\"r\":$line}" >> "$OUT"
}
R=$E/nc; rm -rf "$R"; mkdir -p "$R"
one new_auto_cold   "$NEWPY" auto  "$R/a"
one new_auto_warm   "$NEWPY" auto  "$R/a"
one new_numba_cold  "$NEWPY" numba "$R/b"
one new_numba_warm  "$NEWPY" numba "$R/b"
one new_numba_generic_cold "$NEWPY" numba "$R/c" NUMBA_CPU_NAME=generic
one new_numba_generic_warm "$NEWPY" numba "$R/c" NUMBA_CPU_NAME=generic
one old_cold        "$OLDPY" numba "$R/d"
one old_warm        "$OLDPY" numba "$R/d"
echo DONE >> "$OUT"
