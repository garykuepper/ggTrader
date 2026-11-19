#!/usr/bin/env bash
# Run any command inside a memory-capped transient cgroup with swap disabled.
#
# Wraps investigation backtest re-runs so a runaway cannot wedge the host
# (31 GB RAM; a past incident reached ~10 GB and froze the machine, see
# scripts/run_tests.sh). MemorySwapMax=0 matters most: swap thrashing, not the
# allocation itself, is what made the box unreachable.
# The scope inherits the caller's slice (systemd >= 246); direct commands do not.
#
# Usage:
#   scripts/run_capped.sh <command...>
#   GGT_RUN_MEM_MAX=4G scripts/run_capped.sh python scripts/some_wfo.py
#   scripts/run_capped.sh --help
#
# Environment:
#   GGT_RUN_MEM_MAX      cgroup MemoryMax for the command (default 6G)
#   GGT_RUN_MAX_SECONDS  wall-clock limit for the command (default 7200)
#   GGT_HEAVY_WAIT       seconds to wait for another heavy job to finish (default 1800)
#   GGT_HEAVY_MARGIN     free memory required beyond the cap (default 4G)
#
# Only one heavy ggTrader job (this script or scripts/run_tests.sh) runs at a time: a second one
# waits on a lock in the repository's git directory, then exits 75. It also exits 75, at once,
# when free memory is below the cap plus the margin. See scripts/_heavy_job.sh.
#
# Exit status is the command's own. Exit 137 or 143 alone does not establish the cause. This
# script never retries; raising a cap or limit must be deliberate. Exit 75 means busy or short
# of memory, and the command never started.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEM_MAX="${GGT_RUN_MEM_MAX:-6G}"
MAX_SECONDS="${GGT_RUN_MAX_SECONDS:-7200}"

usage() {
    cat <<EOF
Usage: scripts/run_capped.sh [--help] <command...>

Run <command...> in a transient systemd scope with MemoryMax=\$GGT_RUN_MEM_MAX
(default 6G), MemorySwapMax=0 and a wall-clock limit of \$GGT_RUN_MAX_SECONDS
(default 7200), with the repo .venv/bin first on PATH. Only one heavy ggTrader job
runs at a time (see scripts/_heavy_job.sh).

Environment:
  GGT_RUN_MEM_MAX      memory cap for the command (default 6G)
  GGT_RUN_MAX_SECONDS  wall-clock limit for the command (default 7200)
  GGT_HEAVY_WAIT       seconds to wait for another heavy job (default 1800)
  GGT_HEAVY_MARGIN     free memory required beyond the cap (default 4G)

Exit 137 means the command was killed at the cap and exit 143 that the time limit
stopped it; the script prints a message and does not retry. Exit 75 means another
heavy job is running or free memory is short: the command never started.
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

# shellcheck source=scripts/_heavy_job.sh
source "$REPO/scripts/_heavy_job.sh"
heavy_job_enter run_capped "$MEM_MAX"

echo "running under MemoryMax=$MEM_MAX (no swap), time limit ${MAX_SECONDS}s" >&2

set +e
systemd-run --user --scope -q --collect --slice-inherit \
    -p MemoryMax="$MEM_MAX" \
    -p MemorySwapMax=0 \
    -p RuntimeMaxSec="$MAX_SECONDS" \
    env "PATH=$REPO/.venv/bin:$PATH" "$@"
status=$?
set -e

# Exit codes alone do not identify the cause.
if [[ $status -eq 137 ]]; then
    echo "Killed (exit 137); a memory cap or another cause is unverified. Do not retry blindly." >&2
fi

if [[ $status -eq 143 ]]; then
    echo "Terminated (exit 143); a time limit or another signal may be responsible." >&2
fi

exit $status
