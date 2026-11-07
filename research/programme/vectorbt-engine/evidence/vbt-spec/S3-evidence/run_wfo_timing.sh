#!/usr/bin/env bash
E=/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S3-evidence
W=/home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0
cd "$W"
OUT=$E/58_wfo_timing.txt; : > "$OUT"
for spec in "old /home/flynn/ggTrader/.venv/bin/python" "new_patched $W/.venv/bin/python" "old2 /home/flynn/ggTrader/.venv/bin/python" "new_patched2 $W/.venv/bin/python"; do
    set -- $spec
    echo "load=$(cut -d' ' -f1 /proc/loadavg) $1" >> "$OUT"
    PYTHONPATH=$W/src NUMBA_CACHE_DIR=$E/nc/w_$1 "$E/with_db_env.sh" "$2" "$E/e2e_wfo.py" "$E/wfo_t_$1" --n-symbols 60 2>/dev/null | grep -E '"(load_s|wfo_s|peak_rss_mb|oos_equity_last)"' >> "$OUT"
done
echo DONE >> "$OUT"
