#!/usr/bin/env bash
# Write a deterministic snapshot of state the root-cause investigation must not touch.
#
# Covers the intraday observer worktree and its four systemd user timers, the
# live ggtrader_live container, and data/ plus branch main in the main checkout.
# Run it before and after the investigation and diff the two outputs: any
# difference means off-limits state changed. Everything here is read-only; it
# never starts, stops, enables, restarts or edits a unit, container or worktree.
#
# Volatile fields (next/last run, uptime text, git status of the observer
# worktree) are deliberately excluded so legitimate day-to-day churn does not
# raise a false alarm.
#
# Usage:
#   research/audits/root-cause-2026-11-04/code/guard_snapshot.sh [<outfile>]   write to <outfile> (default: stdout)
#   research/audits/root-cause-2026-11-04/code/guard_snapshot.sh --help
set -euo pipefail

REPO="/home/flynn/ggTrader"
OBSERVER_WORKTREE="/home/flynn/.t3/worktrees/ggTrader/research-intraday-stage0a"
CONTAINER="ggtrader_live"
TIMER_PREFIX="ggtrader-intraday-observer"
TIMER_SUFFIXES=(recover inputs decide archive)

usage() {
    cat <<EOF
Usage: research/audits/root-cause-2026-11-04/code/guard_snapshot.sh [--help] [<outfile>]

Write a deterministic, read-only snapshot of off-limits state to <outfile>
(or stdout if omitted):

  1. observer worktree HEAD and branch (no git status)
  2. enabled state and unit-file sha256 for the four observer timers
     (recover, inputs, decide, archive)
  3. ggtrader_live container name, image and state (no uptime)
  4. git status --short data/ and the hash of branch main in the main checkout

Missing items are recorded as "<what>: not found". Two runs with no real change
produce identical output.
EOF
}

section() {
    printf '== %s ==\n' "$1"
}

snapshot_worktree() {
    section "observer worktree"
    if [[ -d "$OBSERVER_WORKTREE" ]] \
        && git -C "$OBSERVER_WORKTREE" rev-parse --git-dir >/dev/null 2>&1; then
        echo "path: $OBSERVER_WORKTREE"
        echo "HEAD: $(git -C "$OBSERVER_WORKTREE" rev-parse HEAD)"
        echo "branch: $(git -C "$OBSERVER_WORKTREE" rev-parse --abbrev-ref HEAD)"
    else
        echo "observer worktree: not found"
    fi
}

snapshot_timers() {
    section "observer timers"
    local suffix name state unit_text
    for suffix in "${TIMER_SUFFIXES[@]}"; do
        name="$TIMER_PREFIX-$suffix"
        # is-enabled exits non-zero for disabled/not-found; the printed word is the datum.
        state="$(systemctl --user is-enabled "$name.timer" 2>/dev/null || true)"
        echo "$name.timer is-enabled: ${state:-unknown}"
        if unit_text="$(systemctl --user cat "$name.timer" "$name.service" 2>/dev/null)" \
            && [[ -n "$unit_text" ]]; then
            echo "$name unit sha256: $(printf '%s\n' "$unit_text" | sha256sum | cut -d' ' -f1)"
        else
            echo "$name unit: not found"
        fi
    done
}

snapshot_container() {
    section "container"
    local line=""
    if command -v docker >/dev/null 2>&1; then
        line="$(docker ps -a --filter "name=^${CONTAINER}\$" \
            --format '{{.Names}} {{.Image}} {{.State}}' 2>/dev/null || true)"
    fi
    if [[ -n "$line" ]]; then
        echo "$line"
    else
        echo "$CONTAINER: not found"
    fi
}

snapshot_main_checkout() {
    section "main checkout"
    echo "git status --short data/:"
    git -C "$REPO" status --short data/ | LC_ALL=C sort | sed 's/^/  /'
    echo "main: $(git -C "$REPO" log -1 --format=%H main)"
}

snapshot() {
    snapshot_worktree
    snapshot_timers
    snapshot_container
    snapshot_main_checkout
}

main() {
    if [[ $# -gt 1 ]]; then
        usage >&2
        exit 2
    fi
    if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
        usage
        exit 0
    fi
    if [[ $# -eq 1 ]]; then
        snapshot >"$1"
    else
        snapshot
    fi
}

main "$@"
