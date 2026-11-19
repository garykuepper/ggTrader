# shellcheck shell=bash
# Shared guard for heavy ggTrader jobs. Sourced by scripts/run_tests.sh and scripts/run_capped.sh;
# it defines functions and has no effect until heavy_job_enter is called. The caller sets REPO.
#
# Why it exists: on 2026-11-18 two parallel `pytest tests/paper/` runs each grew toward their own
# 8 GiB cap, together needed more than Encom had free, so neither was killed and the host thrashed
# until it was power-cycled. A per-run cap cannot stop two runs adding up. This guard makes sure
# only one heavy job runs at a time, and that it starts only when memory is free for it.
#
# What heavy_job_enter does (it returns 0, or exits 75 so a caller can tell "busy" from "failed"):
#   1. Does nothing when GGT_TEST_NO_CGROUP=1 (CI runners), GGT_HEAVY_LOCK_HELD=1 (a job started
#      from inside a job that already holds the lock, such as the runner scripts' own tests) or
#      GGT_TEST_DRY_RUN=1. GGT_TEST_NO_SCOPE=1 (the whole-job broker's unit, which cannot start a
#      nested scope) is NOT a bypass: it keeps every step below, cancels the GGT_TEST_NO_CGROUP=1
#      bypass, and fails closed (exit 75) where the default mode would carry on: the
#      lock must be the shared one in the git directory (or GGT_HEAVY_LOCK), never the per-checkout
#      fallback; a lock that cannot be opened refuses; an unreadable MemAvailable refuses.
#   2. Takes an exclusive flock on <git common dir>/ggtrader-heavy.lock, shared by every worktree
#      of this repository. A second job waits (GGT_HEAVY_WAIT seconds, default 1800), then exits 75.
#      The kernel releases the lock when the holder dies, however it dies.
#   3. Writes a one-line note of who holds it (<lock>.holder); a stale note is only information.
#   4. Exports GGT_HEAVY_LOCK_HELD=1 so nested jobs do not wait on their own parent.
#   5. Refuses (exit 75) unless MemAvailable is at least the job's cap plus GGT_HEAVY_MARGIN
#      (default 4G): a job that would need more than is free must not start.
#
# Environment: GGT_HEAVY_LOCK (lock path override), GGT_HEAVY_WAIT, GGT_HEAVY_MARGIN,
# GGT_MEMINFO (meminfo path override, for tests).

# size_to_kb SIZE: 512M, 4G, 64K or plain bytes -> kilobytes. Anything else (1.5G, 50%, 4g) is an
# error, because systemd reads some of those differently: callers must run it as
# `kb="$(size_to_kb "$x")" || exit 2`, since an exit inside $(...) only leaves the subshell.
size_to_kb() {
    local size="$1"
    if [[ ! "$size" =~ ^[0-9]+[KMG]?$ ]]; then
        echo "error: cannot read the size '$1' (use for example 512M or 4G)" >&2
        exit 2
    fi
    case "$size" in
        *G) echo $(( ${size%G} * 1024 * 1024 )) ;;
        *M) echo $(( ${size%M} * 1024 )) ;;
        *K) echo "${size%K}" ;;
        *) echo $(( size / 1024 )) ;;
    esac
}

# heavy_job_lock_path: where the shared lock lives.
heavy_job_lock_path() {
    if [[ -n "${GGT_HEAVY_LOCK:-}" ]]; then
        printf '%s\n' "$GGT_HEAVY_LOCK"
        return
    fi
    local common
    if common="$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
        printf '%s\n' "$common/ggtrader-heavy.lock"
    elif [[ "${GGT_TEST_NO_SCOPE:-}" == "1" ]]; then
        # A per-checkout lock is not shared with the developer's jobs: refuse rather than use it.
        return 1
    else
        printf '%s\n' "$REPO/.ggtrader-heavy.lock"
    fi
}

# heavy_job_enter LABEL MEM_MAX: see the header.
heavy_job_enter() {
    local label="$1" mem_max="$2" strict=0
    [[ "${GGT_TEST_NO_SCOPE:-}" == "1" ]] && strict=1
    # A nested run (GGT_HEAVY_LOCK_HELD=1, exported below) is inside the job that holds the lock.
    # The broker's driver starts each phase with neither GGT_HEAVY_LOCK_HELD nor GGT_TEST_DRY_RUN.
    if [[ ( "${GGT_TEST_NO_CGROUP:-}" == "1" && $strict -eq 0 ) \
        || "${GGT_HEAVY_LOCK_HELD:-}" == "1" || "${GGT_TEST_DRY_RUN:-}" == "1" ]]; then
        return 0
    fi

    # Validate before taking the lock: an unreadable size must stop the job, never switch the
    # free-memory check off.
    local cap_kb margin_kb
    cap_kb="$(size_to_kb "$mem_max")" || exit 2
    margin_kb="$(size_to_kb "${GGT_HEAVY_MARGIN:-4G}")" || exit 2

    local lock holder wait_s note
    if ! lock="$(heavy_job_lock_path)"; then
        echo "error: cannot find the shared heavy-job lock (no git directory; set GGT_HEAVY_LOCK);" \
            "not starting" >&2
        exit 75
    fi
    holder="$lock.holder"
    wait_s="${GGT_HEAVY_WAIT:-1800}"
    # Descriptor 9 stays open for the life of this process and of every child it starts: a job
    # that outlives this script keeps the lock until it ends, which is what we want.
    if [[ $strict -eq 1 ]]; then
        if ! { exec 9>>"$lock"; } 2>/dev/null; then
            echo "error: cannot open the shared heavy-job lock $lock; not starting" >&2
            exit 75
        fi
    else
        exec 9>>"$lock"
    fi
    if ! flock -n 9; then
        note="$(cat "$holder" 2>/dev/null || echo 'holder unknown')"
        echo "waiting for another heavy ggTrader job ($note)" >&2
        if ! flock -w "$wait_s" 9; then
            note="$(cat "$holder" 2>/dev/null || echo 'holder unknown')"
            echo "another heavy ggTrader job is still running after ${wait_s}s: $note" >&2
            exit 75
        fi
    fi
    printf 'pid=%s label=%s cap=%s started=%s\n' "$$" "$label" "$mem_max" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$holder"
    # shellcheck disable=SC2064
    trap "rm -f '$holder'" EXIT
    export GGT_HEAVY_LOCK_HELD=1

    local avail_kb need_kb margin
    margin="${GGT_HEAVY_MARGIN:-4G}"
    avail_kb="$(awk '/^MemAvailable:/ {print $2}' "${GGT_MEMINFO:-/proc/meminfo}" 2>/dev/null || true)"
    need_kb=$(( cap_kb + margin_kb ))
    if [[ $strict -eq 1 && ! "$avail_kb" =~ ^[0-9]+$ ]]; then
        echo "error: cannot read MemAvailable; not starting a $mem_max job" >&2
        exit 75
    fi
    if [[ -n "$avail_kb" ]] && (( avail_kb < need_kb )); then
        echo "not enough free memory for a $mem_max job: $(( avail_kb / 1024 )) MiB available," \
            "$(( need_kb / 1024 )) MiB needed (the cap plus a $margin margin); not starting" >&2
        exit 75
    fi
}
