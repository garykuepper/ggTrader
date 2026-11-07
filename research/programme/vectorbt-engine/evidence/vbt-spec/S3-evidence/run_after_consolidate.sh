#!/usr/bin/env bash
# After the extract_* consolidation patch: (1) steady-state sweep timing on new stack, (2) re-run
# e2e_sweep on new stack and re-compare to the OLD-stack artefacts (bit-identity must hold).
E=/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S3-evidence
W=/home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0
cd "$W"
OUT=$E/55_after_consolidate.jsonl; : > "$OUT"
echo "load=$(cut -d' ' -f1 /proc/loadavg) new_auto_patched" >> "$OUT"
PYTHONPATH=$W/src NUMBA_CACHE_DIR=$E/nc/t_new_auto "$E/with_db_env.sh" "$W/.venv/bin/python" "$E/sweep_timing.py" "$E/sweep_new/pit_mask.parquet" auto 3 >> "$OUT" 2>&1
echo "load=$(cut -d' ' -f1 /proc/loadavg) new_numba_patched" >> "$OUT"
PYTHONPATH=$W/src NUMBA_CACHE_DIR=$E/nc/t_new_numba "$E/with_db_env.sh" "$W/.venv/bin/python" "$E/sweep_timing.py" "$E/sweep_new/pit_mask.parquet" numba 3 >> "$OUT" 2>&1
rm -rf "$E/sweep_new_patched"
PYTHONPATH=$W/src "$E/with_db_env.sh" "$W/.venv/bin/python" "$E/e2e_sweep.py" "$E/sweep_new_patched" --n-symbols 100 > "$E/56_sweep_new_patched.log" 2>&1
# compare patched-new vs old by pointing the compare script at a temp dir that pairs them
rm -rf "$E/cmp_patched"; mkdir -p "$E/cmp_patched"
ln -s "$E/sweep_old" "$E/cmp_patched/sweep_old"; ln -s "$E/sweep_new_patched" "$E/cmp_patched/sweep_new"
ln -s "$E/e2e_old" "$E/cmp_patched/e2e_old"; ln -s "$E/e2e_new" "$E/cmp_patched/e2e_new"
"$W/.venv/bin/python" "$E/e2e_compare.py" "$E/cmp_patched" > "$E/57_compare_patched.txt" 2>&1
echo DONE >> "$OUT"
