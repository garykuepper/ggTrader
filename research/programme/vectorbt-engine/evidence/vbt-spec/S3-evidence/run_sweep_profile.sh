#!/usr/bin/env bash
E=/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S3-evidence
W=/home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0
cd "$W"
for spec in "old /home/flynn/ggTrader/.venv/bin/python na" "new $W/.venv/bin/python auto"; do
    set -- $spec
    PROFILE=$E/52_prof_$1.prof PYTHONPATH=$W/src NUMBA_CACHE_DIR=$E/nc/t_$1 "$E/with_db_env.sh" "$2" "$E/sweep_timing.py" "$E/sweep_new/pit_mask.parquet" "$3" 2 > "$E/52_sweep_profile_$1.txt" 2>&1
done
echo DONE > "$E/52_done"
