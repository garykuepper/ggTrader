#!/usr/bin/env bash
# Run the ggTrader test suite inside a memory-capped transient cgroup.
#
# The suite's memory grows monotonically with test count rather than
# plateauing: a full `pytest tests/` run reached ~10 GB on 2026-09-16 and
# exhausted both RAM and the 8 GB swap file, wedging the whole host for
# minutes at a time until the OOM killer landed. The cap bounds a runaway
# run to this scope so it dies alone instead of taking Encom with it.
#
# MemorySwapMax=0 is the important half: swap-file thrashing, not the
# allocation itself, is what made the box unreachable.
#
# Usage:
#   scripts/run_tests.sh                    # full suite
#   scripts/run_tests.sh tests/paper -q     # any pytest args
#   GGT_TEST_MEM_MAX=4G scripts/run_tests.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEM_MAX="${GGT_TEST_MEM_MAX:-8G}"
PYTHON="$REPO/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "error: no venv interpreter at $PYTHON" >&2
    exit 1
fi

# No pytest args given: run the default suite.
if [[ $# -eq 0 ]]; then
    set -- tests/
fi

cd "$REPO"

echo "running pytest under MemoryMax=$MEM_MAX (no swap)" >&2

set +e
systemd-run --user --scope -q --collect \
    -p MemoryMax="$MEM_MAX" \
    -p MemorySwapMax=0 \
    env PYTHONPATH=src "$PYTHON" -m pytest "$@"
status=$?
set -e

# 137 = SIGKILL, which from inside this scope means the cap was hit.
if [[ $status -eq 137 ]]; then
    echo >&2
    echo "pytest was killed at the $MEM_MAX cap -- the suite outgrew its budget." >&2
    echo "Raise it deliberately with GGT_TEST_MEM_MAX, or fix the leak." >&2
fi

exit $status
