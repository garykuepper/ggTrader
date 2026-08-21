#!/usr/bin/env bash
# PreToolUse hook: block Edit/Write/MultiEdit on live trader state and secrets.
# Reads tool call JSON from stdin; exits 2 (with message on stderr) to block.

set -euo pipefail

path=$(jq -r 'try .tool_input.file_path // empty')
[ -z "$path" ] && exit 0

case "$path" in
  */data/live/*)
    echo "BLOCKED: $path is live trader runtime state. Corrupting it can desync the running trader from real Kraken positions. If you really need to edit, do it from a shell while the trader is stopped." >&2
    exit 2
    ;;
  */.env)
    echo "BLOCKED: $path holds Kraken/Alpaca API keys. Edit it manually in a shell, not through Claude." >&2
    exit 2
    ;;
esac

exit 0
