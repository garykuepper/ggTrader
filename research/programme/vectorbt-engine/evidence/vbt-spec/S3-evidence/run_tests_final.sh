#!/usr/bin/env bash
E=/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S3-evidence
cd /home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0
# 1) new stack, full suite, with the single event_study patch
scripts/run_tests.sh tests/ -q -p no:cacheprovider --junitxml="$E/12_tests_new_patched.xml" -rfEs > "$E/12_tests_new_patched.log" 2>&1
echo "exit $?" >> "$E/12_tests_new_patched.log"
# 2) OLD stack with FutureWarning promoted to error: pre-migration hazard finder (informational)
GGT_PYTHON=/home/flynn/ggTrader/.venv/bin/python scripts/run_tests.sh tests/ -q -p no:cacheprovider -W error::FutureWarning -W error::DeprecationWarning:ggTrader --junitxml="$E/13_tests_old_futurewarning.xml" -rfEs > "$E/13_tests_old_futurewarning.log" 2>&1
echo "exit $?" >> "$E/13_tests_old_futurewarning.log"
echo ALLDONE >> "$E/13_tests_old_futurewarning.log"
