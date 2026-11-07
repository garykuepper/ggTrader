#!/usr/bin/env bash
# `--help` smoke over every scripts/*.py (import-time breakage detector) for one interpreter.
# Usage: scripts_help_sweep.sh <label> <python>   -> 60_help_<label>.tsv  (script \t exit)
E=/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S3-evidence
W=/home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0
label=$1; py=$2
cd "$W"
: > "$E/60_help_$label.tsv"
for f in scripts/*.py; do
    if ! grep -qE "argparse|typer|click" "$f"; then printf "%s\tSKIP-no-argparse\t\n" "$f" >> "$E/60_help_$label.tsv"; continue; fi
    out=$(PYTHONPATH=$W/src timeout 90 "$py" -W ignore "$f" --help 2>&1 </dev/null); rc=$?
    last=$(echo "$out" | grep -E "Error|Exception" | tail -1 | cut -c1-200)
    printf "%s\t%s\t%s\n" "$f" "$rc" "$last" >> "$E/60_help_$label.tsv"
done
echo done >> "$E/60_help_$label.tsv"
