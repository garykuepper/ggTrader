#!/usr/bin/env bash
E=/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S3-evidence
W=/home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0
cd "$W"
OUT=$E/51_sweep_timing.jsonl; : > "$OUT"
t() { # label py engine
    echo "load=$(cut -d' ' -f1 /proc/loadavg) $1" >> "$OUT"
    PYTHONPATH=$W/src NUMBA_CACHE_DIR=$E/nc/t_$1 "$E/with_db_env.sh" "$2" "$E/sweep_timing.py" "$E/sweep_new/pit_mask.parquet" "$3" 3 >> "$OUT" 2>> "$E/51_sweep_timing.err"
}
t old_numba  /home/flynn/ggTrader/.venv/bin/python na
t new_numba  "$W/.venv/bin/python" numba
t new_auto   "$W/.venv/bin/python" auto
t old_numba2 /home/flynn/ggTrader/.venv/bin/python na
t new_auto2  "$W/.venv/bin/python" auto
echo DONE >> "$OUT"
