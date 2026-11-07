#!/usr/bin/env bash
# Sequential read-only e2e runs, capped at 6G, no swap. Logs next to this script.
set -u
E=/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S3-evidence
W=/home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0
NEWPY=$W/.venv/bin/python
OLDPY=/home/flynn/ggTrader/.venv/bin/python
cd "$W"
run() { # name python script outdir args...
    local name=$1 py=$2 script=$3 outdir=$4; shift 4
    echo "== $name start $(date +%T)" >> "$E/40_e2e_all.log"
    PYTHONPATH=$W/src "$E/with_db_env.sh" systemd-run --user --scope -q --collect \
        -p MemoryMax=6G -p MemorySwapMax=0 env PYTHONPATH=$W/src \
        /usr/bin/time -v "$py" "$script" "$outdir" "$@" > "$E/$name.log" 2> "$E/$name.err"
    echo "== $name exit $? $(date +%T)" >> "$E/40_e2e_all.log"
}
run 31_e2e_old "$OLDPY" "$E/e2e_wfo.py" "$E/e2e_old" --n-symbols 60
run 32_sweep_new "$NEWPY" "$E/e2e_sweep.py" "$E/sweep_new" --n-symbols 100
run 33_sweep_old "$OLDPY" "$E/e2e_sweep.py" "$E/sweep_old" --n-symbols 100
echo ALLDONE >> "$E/40_e2e_all.log"
