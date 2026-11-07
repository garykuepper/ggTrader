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
# Tiers (the marker expression of each lives in tests/tiers.py, the one source CI uses too):
#   scripts/run_tests.sh                 FAST CORE: the developer loop, about a minute and a half.
#   scripts/run_tests.sh --fast [ARGS]   Same as no arguments, with extra pytest ARGS.
#   scripts/run_tests.sh --full [ARGS]   EVERYTHING except `integration`: the merge, deploy and
#                                        vectorbt-upgrade gate. Includes the live golden, the
#                                        real-tape oracles, the numba engine and the study tests.
#   scripts/run_tests.sh --studies [ARGS]  Only the study tier (`-m study`). It always names `tests`
#                                        (study tests still live there) plus every existing
#                                        research/*/*/code/tests and research/*/_lib/tests folder,
#                                        explicitly: never pass the bare `research` directory to
#                                        pytest (it holds frozen test_*.py files that import deleted
#                                        modules). When such folders exist it adds
#                                        --import-mode=importlib (study test file names may repeat).
#                                        Collecting no test at all is an error (pytest exit 5).
#   scripts/run_tests.sh <pytest args>   Anything else goes to pytest unchanged and, as always,
#                                        runs every test in the paths named (default: all of
#                                        tests/ except `integration`), i.e. the FULL semantics.
#
# Examples:
#   scripts/run_tests.sh tests/paper -q     # any pytest args
#   scripts/run_tests.sh --full -q          # the gate before a merge or a deploy
#   GGT_TEST_MEM_MAX=4G scripts/run_tests.sh
#   GGT_TEST_PYTHON=/absolute/frozen/venv/bin/python scripts/run_tests.sh tests/lab -q
#   (GGT_PYTHON is accepted as a shorter alias and wins if both are set)
#   GGT_TEST_NO_CGROUP=1  skip systemd-run (CI runners; the cap is for the developer host)
#   GGT_TEST_DRY_RUN=1    print the pytest command and exit (used by tests/test_root_conftest.py)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEM_MAX="${GGT_TEST_MEM_MAX:-8G}"
PYTHON="${GGT_PYTHON:-${GGT_TEST_PYTHON:-$REPO/.venv/bin/python}}"

usage() {
    # The header comment above is the documentation; print it without the shebang.
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

# The first argument picks the tier; every other way of calling this script is passed through.
TIER=""
case "${1:-}" in
    "") TIER="fast" ;;
    --fast) TIER="fast"; shift ;;
    --full) TIER="full"; shift ;;
    --studies) TIER="studies"; shift ;;
esac

if [[ ! -x "$PYTHON" ]]; then
    echo "error: no venv interpreter at $PYTHON" >&2
    exit 1
fi

cd "$REPO"

if [[ "$TIER" == "studies" ]]; then
    # Explicit paths only. nullglob: no match means no extra path, not a literal pattern.
    shopt -s nullglob
    study_dirs=(research/*/*/code/tests research/*/_lib/tests)
    shopt -u nullglob
    if [[ ${#study_dirs[@]} -gt 0 ]]; then
        set -- --import-mode=importlib tests "${study_dirs[@]}" "$@"
    else
        set -- tests "$@"
    fi
fi

if [[ -n "$TIER" ]]; then
    # -rs names every skip: the real-tape oracles skip silently on a checkout without the
    # git-ignored tape, and a green run that skipped them must say so.
    set -- --tier "$TIER" -rs "$@"
    echo "running the $TIER test tier under MemoryMax=$MEM_MAX (no swap)" >&2
else
    echo "running pytest under MemoryMax=$MEM_MAX (no swap)" >&2
fi

if [[ "${GGT_TEST_DRY_RUN:-}" == "1" ]]; then
    echo "$PYTHON -m pytest $*"
    exit 0
fi

set +e
if [[ "${GGT_TEST_NO_CGROUP:-}" == "1" ]]; then
    env PYTHONPATH=src "$PYTHON" -m pytest "$@"
else
    systemd-run --user --scope -q --collect \
        -p MemoryMax="$MEM_MAX" \
        -p MemorySwapMax=0 \
        env PYTHONPATH=src "$PYTHON" -m pytest "$@"
fi
status=$?
set -e

# 137 = SIGKILL, which from inside this scope means the cap was hit.
if [[ $status -eq 137 ]]; then
    echo >&2
    echo "pytest was killed at the $MEM_MAX cap -- the suite outgrew its budget." >&2
    echo "Raise it deliberately with GGT_TEST_MEM_MAX, or fix the leak." >&2
fi

if [[ "$TIER" == "fast" ]]; then
    echo >&2
    echo "FAST CORE only: the slow, numba, tripwire, study, tooling and integration tests" >&2
    echo "were deselected (see the deselected count above)." >&2
    echo "Run scripts/run_tests.sh --full before any merge, deploy or vectorbt upgrade." >&2
fi

exit $status
