#!/usr/bin/env bash
# PostToolUse hook: auto-fix and format .py files after Edit/Write/MultiEdit.
# Uses the project's .venv ruff so the version matches ruff.toml expectations.
# Failures are non-blocking — Claude sees stderr but the edit stands.

set -uo pipefail

path=$(jq -r 'try .tool_input.file_path // empty')
[ -z "$path" ] && exit 0

# Only Python files inside this repo
case "$path" in
  *.py) ;;
  *) exit 0 ;;
esac

REPO=/home/flynn/ggTrader
case "$path" in
  "$REPO"/*) ;;
  *) exit 0 ;;
esac

RUFF="$REPO/.venv/bin/ruff"
[ -x "$RUFF" ] || RUFF=ruff
# Run from the repo root so ruff.toml (force-exclude) applies whatever the session cwd is.
cd "$REPO" || exit 0

"$RUFF" check --fix --quiet "$path" 2>&1 || true
"$RUFF" format --quiet "$path" 2>&1 || true
exit 0
