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
#                                        vectorbt-upgrade gate. Two pytest runs, one after the
#                                        other: (1) `pytest --tier full tests` (the live golden,
#                                        the real-tape oracles, the numba engine), then (2) the
#                                        study half below.
#   scripts/run_tests.sh --studies [ARGS]  The study tier. Two pytest runs: (1) `pytest --tier
#                                        studies tests` (exit 5, nothing collected, is accepted:
#                                        study tests are leaving tests/), then (2) the study half.
#                                        The study half is `pytest --import-mode=importlib -m "not
#                                        integration" --min-collected N <folders>` over the
#                                        research/*/*/code/tests and research/*/_lib/tests folders
#                                        that exist (tests/tiers.py: STUDY_TEST_GLOBS, and N =
#                                        STUDIES_MIN_COLLECTED, which a run below fails). importlib
#                                        mode because study test file names may repeat. It never
#                                        passes the bare `research` folder to pytest: that holds
#                                        frozen test_*.py files that import deleted modules. With no
#                                        folder yet and N = 0 the half is skipped with a note.
#                                        The exit status is non-zero if either run fails.
#   scripts/run_tests.sh <pytest args>   Anything else goes to pytest unchanged and, as always,
#                                        runs every test in the paths named (default: all of
#                                        tests/ except `integration`), i.e. the FULL semantics.
#
#   A path among the ARGS of --full or --studies (an existing file or folder, or file::test)
#   makes it a targeted run: ONE pytest run of the tier over just those paths, no study half.
#
# Examples:
#   scripts/run_tests.sh tests/paper -q     # any pytest args
#   scripts/run_tests.sh --full -q          # the gate before a merge or a deploy
#   GGT_TEST_MEM_MAX=4G scripts/run_tests.sh
#   GGT_TEST_PYTHON=/absolute/frozen/venv/bin/python scripts/run_tests.sh tests/lab -q
#   (GGT_PYTHON is accepted as a shorter alias and wins if both are set)
#   GGT_TEST_NO_CGROUP=1  skip systemd-run (CI runners; the cap is for the developer host)
#   GGT_TEST_DRY_RUN=1    print the pytest command(s) and exit (used by tests/test_root_conftest.py)
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

run_pytest() {
    # One pytest run, in the memory-capped scope unless GGT_TEST_NO_CGROUP=1. Returns its status.
    if [[ "${GGT_TEST_DRY_RUN:-}" == "1" ]]; then
        echo "$PYTHON -m pytest $*"
        return 0
    fi
    if [[ "${GGT_TEST_NO_CGROUP:-}" == "1" ]]; then
        env PYTHONPATH=src "$PYTHON" -m pytest "$@"
    else
        systemd-run --user --scope -q --collect \
            -p MemoryMax="$MEM_MAX" \
            -p MemorySwapMax=0 \
            env PYTHONPATH=src "$PYTHON" -m pytest "$@"
    fi
}

study_plan() {
    # Prints the STUDIES_MIN_COLLECTED minimum, then each existing study test folder, one per
    # line. tests/tiers.py is loaded by file path: an installed top-level `tests` package could
    # otherwise win the import.
    "$PYTHON" -c '
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("ggt_tiers", "tests/tiers.py")
tiers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiers)
print(tiers.STUDIES_MIN_COLLECTED)
for folder in tiers.study_test_dirs(Path.cwd()):
    print(folder)
'
}

# A path among the user's arguments makes this a targeted run of one pytest invocation.
# The value of an option that takes one (`-k research`, `-m slow`, `--ignore tests/x`) is not a
# target path even when a file or folder of that name exists, so it is skipped.
has_path=0
skip_value=0
for arg in "$@"; do
    if [[ $skip_value -eq 1 ]]; then
        skip_value=0
        continue
    fi
    case "$arg" in
        -k|-m|-p|-c|-o|-n|-W|-r|--tier|--deselect|--ignore|--ignore-glob|--maxfail|--durations|\
        --durations-min|--min-collected|--rootdir|--import-mode|--junitxml|--basetemp|\
        --log-level|--confcutdir|--golden-reason|--override-ini|--capture|--tb|--color)
            skip_value=1
            continue
            ;;
        -*) continue ;;
    esac
    [[ -e "${arg%%::*}" ]] && has_path=1
done

if [[ -n "$TIER" ]]; then
    echo "running the $TIER test tier under MemoryMax=$MEM_MAX (no swap)" >&2
else
    echo "running pytest under MemoryMax=$MEM_MAX (no swap)" >&2
fi

# The tests/ run (the whole run for fast, targeted and pass-through calls).
if [[ -z "$TIER" ]]; then
    first=("$@")
elif [[ "$TIER" == "fast" || $has_path -eq 1 ]]; then
    # -rs names every skip: the real-tape oracles skip silently on a checkout without the
    # git-ignored tape, and a green run that skipped them must say so.
    first=(--tier "$TIER" -rs "$@")
else
    first=(--tier "$TIER" -rs tests "$@")
fi

status=0
run_pytest "${first[@]}" || status=$?
# Nothing collected from tests/ is fine for the studies tier: the study tests are leaving it.
if [[ "$TIER" == "studies" && $has_path -eq 0 && $status -eq 5 ]]; then
    status=0
fi

if [[ ( "$TIER" == "full" || "$TIER" == "studies" ) && $has_path -eq 0 ]]; then
    mapfile -t plan < <(study_plan)
    min_collected="${plan[0]}"
    study_dirs=()
    for folder in "${plan[@]:1}"; do
        [[ -n "$folder" ]] && study_dirs+=("$folder")
    done
    if [[ ${#study_dirs[@]} -eq 0 && "$min_collected" -eq 0 ]]; then
        echo "no study test folders yet (STUDIES_MIN_COLLECTED = 0): skipping the study half" >&2
    elif [[ ${#study_dirs[@]} -eq 0 ]]; then
        echo "error: no study test folder exists but STUDIES_MIN_COLLECTED = $min_collected" >&2
        status=$(( status == 0 ? 1 : status ))
    else
        half=0
        run_pytest --import-mode=importlib -m "not integration" \
            --min-collected "$min_collected" -rs "${study_dirs[@]}" "$@" || half=$?
        # Exit 5 (nothing collected) is only acceptable while no minimum is promised.
        if [[ $half -eq 5 && "$min_collected" -eq 0 ]]; then
            half=0
        fi
        status=$(( status == 0 ? half : status ))
    fi
fi

if [[ "${GGT_TEST_DRY_RUN:-}" == "1" ]]; then
    exit "$status"
fi

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
