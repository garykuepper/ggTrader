#!/usr/bin/env bash
E=/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S3-evidence
cd /home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0
scripts/run_tests.sh tests/ -q -p no:cacheprovider --junitxml="$E/14_tests_new_final.xml" -rfEs > "$E/14_tests_new_final.log" 2>&1
echo "exit $?" >> "$E/14_tests_new_final.log"
