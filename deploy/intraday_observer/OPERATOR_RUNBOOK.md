# DRAFT intraday observer operator runbook

**Not an activation package and not approved to run.** Parent evidence, final code,
exact placeholders and independent Opus review remain required. Nothing in this file has
been installed or enabled. The observer records hypothetical SPY positions only. It cannot
place orders through this package.

## Registered dispatch

All times are `America/New_York`. The scheduler wrapper must dispatch from the frozen
calendar, return success without source requests on known non-sessions, and skip stages
outside their registered windows. It must not infer a trading day from weekday alone.

| Timer | Service stage | `TimeoutStartSec` | Hard stage boundary |
|---|---|---:|---|
| 15:15 Mon-Fri | `recover` | 270 seconds | finish strictly before 15:20 |
| 15:20 Mon-Fri | `inputs` | 270 seconds | finish strictly before 15:25 |
| 15:25 Mon-Fri | `decide` | 45 seconds | finish strictly before 15:26 |
| 16:20 Mon-Fri | `archive` | 270 seconds | archive due by 16:25 |

The timeouts leave 30 seconds before the recovery/input/archive boundaries and 15 seconds
before the decision boundary. `AccuracySec=1s`, `RandomizedDelaySec=0` and
`Persistent=false` are deliberate. A missed timer is a recorded fault, not permission for
catch-up or a backdated decision. There is no `revise` timer. Optional revisions are manual,
append-only and must never delay required captures.

The four services call only:

```text
REVIEWED_PYTHON RELEASE_ROOT/scripts/intraday_observer_job.py --stage STAGE --state-dir STATE_DIR
```

The wrapper is Sonnet's integration surface. Parent must verify its holiday, early-close,
post-window and failure behavior and ensure the final code binding covers it before freezing
the manifest. These unit drafts do not establish that evidence.

## Fixed identities and required placeholders

Every `__OBSERVER_*__` token must be rendered from parent-reviewed values. There are no
defaults. Never point a service at a T3 worktree, `current` symlink or other mutable checkout.

```text
RELEASE_ROOT=/home/flynn/.local/share/ggtrader/intraday-observer/releases/REVIEWED_COMMIT
STATE_DIR=/home/flynn/.local/share/ggtrader/intraday-observer/studies/REVIEWED_STUDY_ID
```

`REVIEWED_COMMIT` is the exact full commit, `REVIEWED_STUDY_ID` is the approved study ID,
and `REVIEWED_PYTHON` is the exact existing interpreter recorded in readiness evidence.
The observer's existing manifest bindings check protocol, source, interpreter and library
versions. A mismatch is a refusal, never an auto-upgrade. This draft makes no claim of
read-only API scope or independently witnessed timestamps. Credentials authorize whatever
the existing account permits, while this observer code issues only its registered bars GETs.
All decision timestamps remain local host evidence.

## Offline render and syntax verification

These commands render into a temporary directory only. They do not create release/state
paths, load units, read credentials or start jobs.

```bash
set -eu
: "${PACKAGE_DIR:?absolute path to reviewed deploy/intraday_observer}"
: "${REVIEWED_COMMIT:?full reviewed commit required}"
: "${REVIEWED_STUDY_ID:?approved study id required}"
: "${REVIEWED_PYTHON:?exact reviewed interpreter required}"

case "$REVIEWED_COMMIT" in ''|*[!0-9a-f]*) echo "invalid commit" >&2; exit 2;; esac
[ "${#REVIEWED_COMMIT}" -eq 40 ] || { echo "full 40-character commit required" >&2; exit 2; }
case "$REVIEWED_STUDY_ID" in
  ''|*[!A-Za-z0-9._-]*|[!A-Za-z0-9]*) echo "invalid study id" >&2; exit 2;;
esac
case "$REVIEWED_PYTHON" in
  /*) ;;
  *) echo "interpreter must be absolute" >&2; exit 2;;
esac
case "$REVIEWED_PYTHON" in
  *[!A-Za-z0-9._/-]*) echo "unsafe interpreter path" >&2; exit 2;;
esac
[ -x "$REVIEWED_PYTHON" ] || { echo "interpreter not executable" >&2; exit 2; }

RELEASE_ROOT="/home/flynn/.local/share/ggtrader/intraday-observer/releases/$REVIEWED_COMMIT"
STATE_DIR="/home/flynn/.local/share/ggtrader/intraday-observer/studies/$REVIEWED_STUDY_ID"
RENDER_DIR="$(mktemp -d)"
for source in "$PACKAGE_DIR"/*.service "$PACKAGE_DIR"/*.timer; do
  name="${source##*/}"
  sed \
    -e "s|__OBSERVER_RELEASE_ROOT__|$RELEASE_ROOT|g" \
    -e "s|__OBSERVER_STATE_DIR__|$STATE_DIR|g" \
    -e "s|__OBSERVER_PYTHON__|$REVIEWED_PYTHON|g" \
    "$source" > "$RENDER_DIR/$name"
done
if grep -R '__OBSERVER_' "$RENDER_DIR"; then
  echo "unrendered placeholder" >&2
  exit 2
fi
systemd-analyze verify --user "$RENDER_DIR"/*.service "$RENDER_DIR"/*.timer
for stage in recover inputs decide archive; do
  systemd-analyze calendar "$(sed -n 's/^OnCalendar=//p' \
    "$RENDER_DIR/ggtrader-intraday-observer-$stage.timer")"
done
```

Treat a missing executable or user-manager warning during template-only validation as an
environment warning, not as a successful execution rehearsal. Before activation, rerun the
same verification with the real fixed release present and resolve every path warning.
`systemd-analyze verify` does not execute the observer and is not evidence of source access,
calendar dispatch or timing.

## Activation-only installation and initialization

**Do not run this section under Tasks 1-3.** It requires the separate owner activation
decision, final readiness evidence and the exact rendered package. Parent alone provisions
`APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` in the private user-manager environment using
the existing credential process. Services inherit that environment. Do not put credentials
in unit files, this runbook, command arguments, logs or a copied `.env`. Do not run
`systemctl --user show-environment`, because it prints secret values.

```bash
set -eu
# Reuse the reviewed variables and RENDER_DIR from above.
test -d "$RELEASE_ROOT"
test -f "$RELEASE_ROOT/scripts/intraday_observer_job.py"
test -x "$REVIEWED_PYTHON"
test "$(git -C "$RELEASE_ROOT" rev-parse HEAD)" = "$REVIEWED_COMMIT"
test -z "$(git -C "$RELEASE_ROOT" status --porcelain)"
test "$(loginctl show-user "$USER" --property=Linger --value)" = yes

# Read-only logging gate: record the existing bound. Do not change global journald here.
journalctl --disk-usage
systemd-analyze cat-config systemd/journald.conf

# Parent must supply these reviewed values. No value is guessed by this runbook.
: "${ACTIVATION_DATE:?registered New York activation date}"
: "${OFFICIAL_CALENDAR:?saved official calendar JSON}"
: "${SOURCE_DELAY_SECONDS:?verified source delay}"
: "${READINESS_FILE:?verified manifest-bound readiness JSON}"
: "${OWNER_APPROVAL:?exact owner approval record}"
PROTOCOL_FILE="$RELEASE_ROOT/docs/research/briefs/2026-10-25-intraday-prospective-observation.md"
PROTOCOL_SHA256="$(sha256sum "$PROTOCOL_FILE" | cut -d' ' -f1)"

# First persistent writes in this runbook. Stop here unless activation was approved.
install -d -m 700 "$(dirname "$STATE_DIR")"
"$REVIEWED_PYTHON" "$RELEASE_ROOT/scripts/intraday_observe.py" init \
  --state-dir "$STATE_DIR" --mode live --activation-date "$ACTIVATION_DATE" \
  --official-calendar "$OFFICIAL_CALENDAR" --protocol-file "$PROTOCOL_FILE" \
  --protocol-sha256 "$PROTOCOL_SHA256" --source-delay-seconds "$SOURCE_DELAY_SECONDS"
chmod 700 "$STATE_DIR"

UNIT_DIR="$HOME/.config/systemd/user"
install -d -m 700 "$UNIT_DIR"
install -m 600 "$RENDER_DIR"/*.service "$RENDER_DIR"/*.timer "$UNIT_DIR/"
systemctl --user daemon-reload
systemd-analyze verify --user "$UNIT_DIR"/ggtrader-intraday-observer-*.service \
  "$UNIT_DIR"/ggtrader-intraday-observer-*.timer

READINESS_SHA256="$(sha256sum "$READINESS_FILE" | cut -d' ' -f1)"
"$REVIEWED_PYTHON" "$RELEASE_ROOT/scripts/intraday_observe.py" activate \
  --state-dir "$STATE_DIR" --owner-approval "$OWNER_APPROVAL" \
  --readiness-file "$READINESS_FILE" --readiness-sha256 "$READINESS_SHA256"

systemctl --user enable --now \
  ggtrader-intraday-observer-recover.timer \
  ggtrader-intraday-observer-inputs.timer \
  ggtrader-intraday-observer-decide.timer \
  ggtrader-intraday-observer-archive.timer
systemctl --user list-timers --all 'ggtrader-intraday-observer-*'
```

A missing `Linger=yes`, unclean release, absent storage bound, unresolved readiness check or
path warning blocks activation. Do not repair an initialized manifest after a code/runtime
change. Regenerate an unactivated package if approval or activation dates move.

The frozen observer enforces the registered source bounds: at most 12 GET attempts per New
York date across all stages, at least 0.65 seconds between starts, a 30-second request
deadline and at most three attempts per page. The timers add no independent requests.
Parent's Task 1 storage measurement and Task 2 representative-state timing are activation
gates, not facts established by these files.

## Local status and reports

The services write stdout/stderr to the local user journal under identifier
`ggtrader-intraday-observer`, rate-limited to 200 messages per 60 seconds per unit. Journald
must have an existing verified storage bound. This package does not alter global retention.
Original evidence stays under the owner-only `STATE_DIR`; never prune it as log rotation.

```bash
set -eu
"$REVIEWED_PYTHON" "$RELEASE_ROOT/scripts/intraday_observe.py" verify \
  --state-dir "$STATE_DIR"
"$REVIEWED_PYTHON" "$RELEASE_ROOT/scripts/intraday_observe.py" report \
  --state-dir "$STATE_DIR"
journalctl --user -u 'ggtrader-intraday-observer-*' --since today --no-pager
systemctl --user list-timers --all 'ggtrader-intraday-observer-*'

# Optional manual local snapshot after activation. Existing dates are never overwritten.
install -d -m 700 "$STATE_DIR/reports"
REPORT_DATE="$(TZ=America/New_York date +%F)"
"$REVIEWED_PYTHON" "$RELEASE_ROOT/scripts/intraday_observe.py" report \
  --state-dir "$STATE_DIR" --out "$STATE_DIR/reports/$REPORT_DATE.json"
```

Inspect faults, missing timers, clock status and checkpoint status locally. No external
notification is configured or authorized. A journal entry or a process start is not proof of
a timely completed artifact. The JSON report and `verify` result are the local health check.

## Final recovery and disable

After slot 252, keep the timers installed through the next official session's 15:15 recovery.
That final recovery must complete strictly before its 15:20 input boundary if the final
archive lacks a canonical valuation. It cannot create a 253rd observation or extend the
endpoint. Post-window input/decision stages must remain no-ops. Do not use `revise` to cure
an unresolved primary value.

Run `verify` and save the final local report only after that recovery outcome is known. Then
disable all four timers. Disabling never deletes or rewrites the journal.

```bash
systemctl --user disable --now \
  ggtrader-intraday-observer-recover.timer \
  ggtrader-intraday-observer-inputs.timer \
  ggtrader-intraday-observer-decide.timer \
  ggtrader-intraday-observer-archive.timer
systemctl --user list-timers --all 'ggtrader-intraday-observer-*'
```

Use the same disable procedure to stop an active study early. Preserve `STATE_DIR`, the
fixed release, rendered units and all faults for later review. Do not delete state or reset
inception to obtain a clean run.
