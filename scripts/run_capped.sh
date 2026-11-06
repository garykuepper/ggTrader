#!/usr/bin/env bash
# Run any command inside a memory-capped transient cgroup with swap disabled.
#
# Wraps investigation backtest re-runs so a runaway cannot wedge the host
# (31 GB RAM; a past incident reached ~10 GB and froze the machine, see
# scripts/run_tests.sh). MemorySwapMax=0 matters most: swap thrashing, not the
# allocation itself, is what made the box unreachable.
#
# Usage:
#   scripts/run_capped.sh <command...>
#   GGT_RUN_MEM_MAX=4G scripts/run_capped.sh python scripts/some_wfo.py
#   scripts/run_capped.sh --help
#
# Environment:
#   GGT_RUN_MEM_MAX   cgroup MemoryMax for the command (default 6G)
#
# Exit status is the command's own. Exit 137 means the cap was hit: this script
# never retries, and a retry with a higher cap must be a deliberate decision.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEM_MAX="${GGT_RUN_MEM_MAX:-6G}"

usage() {
    cat <<EOF
Usage: scripts/run_capped.sh [--help] <command...>

Run <command...> in a transient systemd scope with MemoryMax=\$GGT_RUN_MEM_MAX
(default 6G) and MemorySwapMax=0, with the repo .venv/bin first on PATH.

Environment:
  GGT_RUN_MEM_MAX   memory cap for the command (default 6G)

Exit 137 means the command was killed at the cap; the script prints a message
and does not retry.
EOF
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi

cd "$REPO"

echo "running under MemoryMax=$MEM_MAX (no swap)" >&2

set +e
systemd-run --user --scope -q --collect \
    -p MemoryMax="$MEM_MAX" \
    -p MemorySwapMax=0 \
    env "PATH=$REPO/.venv/bin:$PATH" "$@"
status=$?
set -e

# 137 = SIGKILL, which from inside this scope means the cap was hit.
if [[ $status -eq 137 ]]; then
    echo "KILLED AT MEMORY CAP ($MEM_MAX): do not retry with a higher cap; report it" >&2
fi

exit $status
