#!/usr/bin/env bash
# Baseline: same worktree source, run with the project's CURRENT venv interpreter (unmodified).
E=/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S3-evidence
cd /home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0
export GGT_PYTHON=/home/flynn/ggTrader/.venv/bin/python
time scripts/run_tests.sh tests/ -q -p no:cacheprovider --junitxml="$E/11_tests_old.xml" -rfEs > "$E/11_tests_old.log" 2>&1
echo "exit $?" >> "$E/11_tests_old.log"
