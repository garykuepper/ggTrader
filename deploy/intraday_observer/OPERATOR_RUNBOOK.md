# Intraday observer operator runbook

**Preparation only. No live state or timers have been installed or activated.**
The package records hypothetical SPY positions, uses local timestamps and never
places orders. Parent addresses Opus findings; this does not imply a second Opus
approval. Activate only after the owner approves the concrete final package.

## Fixed package and runtime

Use the exact full commit, Python path, calendar and state path in
`docs/research/_intraday_observer_activation_proposal.json`. If the New York
activation date moves, regenerate dates before initialization. Never roll an
active inception. First observation is the third full session strictly after
activation, with 252 fixed slots and an operational checkpoint at slot 20.

The dedicated runtime is an independent copy of the existing environment with
the interpreter binary copied and the editable root-checkout hook removed.
No new dependency was installed. `frozen-requirements.txt` records every installed
name/version; `_intraday_observer_runtime_snapshot.json` records its identity.
After tests it is sealed read-only. Use its `bin/python`, not copied `pip` or
console scripts (their old shebangs can name the research environment). Never
upgrade this snapshot. System standard library/shared OS libraries remain host
dependencies; this is not a hermetic system. Package versions and the interpreter
binary must match the reviewed evidence; the observer also enforces its runtime
version binding on every stage.

## Activation-only: create the exact release

**All installation/provisioning blocks below require owner activation approval.**
The source branch need not be pushed; clone from the local checkout containing
the reviewed commit. No existing checkout is modified. Run as the owner.

```bash
set -eu
umask 077
: "${SOURCE_CHECKOUT:?local checkout containing reviewed commit}"
: "${REVIEWED_COMMIT:?full 40-character reviewed commit}"
: "${REVIEWED_PYTHON:?dedicated reviewed runtime bin/python}"
: "${REVIEWED_STUDY_ID:?approved study id}"
: "${REVIEWED_PYTHON_SHA256:?reviewed interpreter binary hash}"
case "$REVIEWED_COMMIT" in ''|*[!0-9a-f]*) exit 2;; esac
[ "${#REVIEWED_COMMIT}" -eq 40 ]
case "$REVIEWED_STUDY_ID" in ''|*[!A-Za-z0-9._-]*|[!A-Za-z0-9]*) exit 2;; esac
case "$REVIEWED_PYTHON" in /*) ;; *) exit 2;; esac
case "$REVIEWED_PYTHON" in *[!A-Za-z0-9._/-]*) exit 2;; esac
[ "$(sha256sum "$REVIEWED_PYTHON" | cut -d' ' -f1)" = "$REVIEWED_PYTHON_SHA256" ]
RELEASE_ROOT="/home/flynn/.local/share/ggtrader/intraday-observer/releases/$REVIEWED_COMMIT"
STATE_DIR="/home/flynn/.local/share/ggtrader/intraday-observer/studies/$REVIEWED_STUDY_ID"
test ! -e "$RELEASE_ROOT"
git clone --no-hardlinks --no-checkout "$SOURCE_CHECKOUT" "$RELEASE_ROOT"
git -C "$RELEASE_ROOT" checkout --detach "$REVIEWED_COMMIT"
test "$(git -C "$RELEASE_ROOT" rev-parse HEAD)" = "$REVIEWED_COMMIT"
test -z "$(git -C "$RELEASE_ROOT" status --porcelain)"
chmod -R a-w "$RELEASE_ROOT"
PACKAGE_DIR="$RELEASE_ROOT/deploy/intraday_observer"
```

An interrupted release creation is inspected and verified rather than blindly
rerunning clone or deleting directories. The fixed release contains its own unit
templates. Do not render from another checkout or mutable `current` symlink.

## Activation-only: private persistent credentials

Recover, inputs and archive use a required `EnvironmentFile=`. Decide has no
credential file. This survives user-manager restart and does not place secrets
in the shared manager environment. Parent alone provisions exactly the two
existing Alpaca variables into a new owner-only file; the root `.env` is never
copied, edited or printed. No secret appears in arguments or logs. Existing
credentials authorize the account's normal scope; only the reviewed code limits
this recorder to GETs. This is not a read-only broker API key.

```bash
set -eu
CREDENTIALS_FILE=/home/flynn/.config/ggtrader/intraday-observer.credentials
install -d -m 700 "$(dirname "$CREDENTIALS_FILE")"
# Parent runs this only after activation approval. Existing file refuses overwrite.
"$REVIEWED_PYTHON" - "$CREDENTIALS_FILE" <<'PY'
import json, os, sys
from dotenv import dotenv_values
source = dotenv_values('/home/flynn/ggTrader/.env')
names = ('APCA_API_KEY_ID', 'APCA_API_SECRET_KEY')
values = {name: source[name] for name in names}
assert all(value and not any(c in value for c in '\r\n\0') for value in values.values())
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as handle:
    for name, value in values.items():
        handle.write(name + '=' + json.dumps(value, ensure_ascii=False) + '\n')
    handle.flush()
    os.fsync(handle.fileno())
PY
test "$(stat -c %a "$CREDENTIALS_FILE")" = 600
test "$(stat -c %u "$CREDENTIALS_FILE")" = "$(id -u)"
```

Do not display the credential file, `systemctl show-environment` or command
headers. Do not import these variables into the user manager. A missing file
fails the fetching unit before execution and is visible in the daily health check.

## Render and verify units — no enabling yet

The rendering itself can be checked in a temporary directory during preparation
with the proposed exact paths; this does not install a unit. `CREDENTIALS_FILE`
is a path, not its contents. The activation-time verification must use the actual
fixed release and runtime. No randomized delay, missed-decision catch-up or
optional revision timer is permitted.

```bash
set -eu
case "$CREDENTIALS_FILE" in *[!A-Za-z0-9._/-]*) exit 2;; esac
RENDER_DIR="$(mktemp -d)"
for source in "$PACKAGE_DIR"/*.service "$PACKAGE_DIR"/*.timer; do
  name="${source##*/}"
  sed -e "s|__OBSERVER_RELEASE_ROOT__|$RELEASE_ROOT|g" \
      -e "s|__OBSERVER_STATE_DIR__|$STATE_DIR|g" \
      -e "s|__OBSERVER_PYTHON__|$REVIEWED_PYTHON|g" \
      -e "s|__OBSERVER_CREDENTIALS_FILE__|$CREDENTIALS_FILE|g" \
      "$source" > "$RENDER_DIR/$name"
done
if grep -R '__OBSERVER_' "$RENDER_DIR"; then exit 2; fi
systemd-analyze verify --user "$RENDER_DIR"/*.service "$RENDER_DIR"/*.timer
for stage in recover inputs decide archive; do
  systemd-analyze calendar "$(sed -n 's/^OnCalendar=//p' \
    "$RENDER_DIR/ggtrader-intraday-observer-$stage.timer")"
done
test "$(loginctl show-user "$USER" --property=Linger --value)" = yes
journalctl --disk-usage
systemd-analyze cat-config systemd/journald.conf
```

| Stage | America/New_York | Timeout |
|---|---:|---:|
| conditional prior archive recovery | 15:15 | 270 s |
| frozen inputs | 15:20 | 270 s |
| decision | 15:25 | 45 s |
| archive | 16:20 | 270 s |

All timers are Mon–Fri, with `AccuracySec=1s`, `RandomizedDelaySec=0`,
`Persistent=false`. The calendar-aware wrapper skips holidays/out-of-study dates,
anchor trading stages and early-close trading stages; early closes still archive.
One late timer can fault today's stage, never decide an earlier day. Timeouts
leave 30/15 seconds of margin before stage boundaries. An exceptional bootstrap
retry sequence can be killed; it remains a fault, not a reason to extend the
window. Prefer the scheduled anchor archive so bootstrap is unnecessary.

## Activation-only: initialize, then stop

This block creates a new inactive manifest. It is run once. A restart must inspect
and use the existing manifest; never rerun `init` or edit the manifest to repair
its hashes. The source calendar and preflight artifacts must already be saved.

```bash
set -eu
: "${ACTIVATION_DATE:?registered New York activation date}"
: "${OFFICIAL_CALENDAR:?saved official calendar JSON}"
: "${PREFLIGHT_FILE:?reviewed final preflight JSON}"
PROTOCOL_FILE="$RELEASE_ROOT/docs/research/briefs/2026-10-25-intraday-prospective-observation.md"
PROTOCOL_SHA256="$(sha256sum "$PROTOCOL_FILE" | cut -d' ' -f1)"
install -d -m 700 "$(dirname "$STATE_DIR")"
"$REVIEWED_PYTHON" "$RELEASE_ROOT/scripts/intraday_observe.py" init \
  --state-dir "$STATE_DIR" --mode live --activation-date "$ACTIVATION_DATE" \
  --official-calendar "$OFFICIAL_CALENDAR" --protocol-file "$PROTOCOL_FILE" \
  --protocol-sha256 "$PROTOCOL_SHA256" --source-delay-seconds 900
chmod 700 "$STATE_DIR"
sha256sum "$STATE_DIR/manifest.json"
```

**Stop here.** Parent compares the actual manifest calendar, paths, code/runtime
identities and delay with the owner-approved proposal. The printed hash now hashes
the exact on-disk bytes. No activation or timers have begun. If these checks do not
match, stop; do not rewrite the manifest or invent readiness booleans.

## Activation-only: bind evidence, then activate and enable

Only after the previous inspection, create the readiness file. Assertions below
must reflect the finalized preflight, tests and parent disposition of all review
findings. It is operator attestation, not independent authentication. A failed or
pending check must stop this block. `READINESS_FILE` is new and write-once.

```bash
set -eu
READINESS_FILE="$STATE_DIR/readiness.json"
"$REVIEWED_PYTHON" - "$STATE_DIR" "$PREFLIGHT_FILE" "$REVIEWED_COMMIT" <<'PY'
import hashlib, json, pathlib, sys
state, preflight, commit = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
raw = (state / 'manifest.json').read_bytes()
preflight_raw = preflight.read_bytes()
manifest, evidence = json.loads(raw), json.loads(preflight_raw)
assert evidence['status'] == 'preparation_verified_not_activated'
assert evidence['verification']['code_commit'] == commit
assert evidence['verification']['full_suite']['exit_code'] == 0
assert evidence['verification']['review_disposition'] == 'important_findings_addressed_by_parent'
assert evidence['source_delay_seconds'] == manifest['source_delay_seconds'] == 900
checks = {k: evidence['checks'][k] for k in
          ('sip_access', 'source_delay', 'calendar_source', 'synthetic_rehearsal')}
assert all(v is True for v in checks.values())
assert manifest['calendar']['official_snapshot_sha256'] == evidence['calendar_snapshot_sha256']
assert manifest['bindings']['code']['sha256'] == evidence['final_code_sha256']
result = {'schema': 'intraday_observer_readiness', 'status': 'verified',
          'manifest_sha256': hashlib.sha256(raw).hexdigest(),
          'source_delay_seconds': 900,
          'calendar_snapshot_sha256': manifest['calendar']['official_snapshot_sha256'],
          'checks': checks, 'preflight_sha256': hashlib.sha256(preflight_raw).hexdigest(),
          'release_commit': commit}
with (state / 'preflight.json').open('xb') as handle:
    handle.write(preflight_raw)
with (state / 'readiness.json').open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
PY
: "${OWNER_APPROVAL:?exact owner activation approval record}"
READINESS_SHA256="$(sha256sum "$READINESS_FILE" | cut -d' ' -f1)"
"$REVIEWED_PYTHON" "$RELEASE_ROOT/scripts/intraday_observe.py" activate \
  --state-dir "$STATE_DIR" --owner-approval "$OWNER_APPROVAL" \
  --readiness-file "$READINESS_FILE" --readiness-sha256 "$READINESS_SHA256"
UNIT_DIR="$HOME/.config/systemd/user"
install -d -m 700 "$UNIT_DIR"
install -m 600 "$RENDER_DIR"/*.service "$RENDER_DIR"/*.timer "$UNIT_DIR/"
systemctl --user daemon-reload
systemctl --user enable --now ggtrader-intraday-observer-{recover,inputs,decide,archive}.timer
systemctl --user list-timers --all 'ggtrader-intraday-observer-*'
```

Activation must occur on the registered New York date. If it has passed, prepare
a new future registration without observations; do not backdate activation.
Verify the first actual inputs, decision and archive, including completion times,
source references and integrity. A process start or successful unit exit alone
is not proof of a timely decision (exit zero also covers no-ops/recorded faults).

## Daily local health check and weekly calendar review

Once each trading day, after 16:30 ET, parent/operator performs:

```bash
systemctl --user --failed --no-pager
systemctl --user list-timers --all 'ggtrader-intraday-observer-*'
"$REVIEWED_PYTHON" "$RELEASE_ROOT/scripts/intraday_observe.py" verify --state-dir "$STATE_DIR"
journalctl --user -u 'ggtrader-intraday-observer-*' --since today --no-pager
```

Compare expected stages with actual completed records; note missing/late stages
and failed units. This is a manual daily obligation, not an installed monitor or
external alert. Local logs are journal-rate-limited to 200 messages/60 s/unit and
use existing journald storage policy (no global retention changes). Never prune
source evidence as logs. Avoid heavy research work between 15:10 and 16:30 ET.
Generate the 51-second year-sized report only outside that interval; save dated,
write-once local JSON with `report --out NEW_FILE`. Do not poll reports each minute.

Weekly and after an exchange closure notice, compare the official paper calendar
with upcoming effective manifest rows. Parent makes one bounded read-only calendar
GET with no redirects/retries and a 30-second deadline, reserving it against the
same New York date's twelve-request budget via `Observer.reserve_request` first.
Save original bytes, request parameters, receipt time and hash before comparison.
Do this outside collection windows; skip the check if the daily budget lacks room.
Check missing dates and changed open/close times, not weekdays alone. A discrepancy
requires owner approval of an append-only amendment, **before the affected day's
15:20 ET** and before any input, decision or archive exists for that day:

```bash
# Set SESSION and the exact approval; never amend a past observed session.
"$REVIEWED_PYTHON" "$RELEASE_ROOT/scripts/intraday_observe.py" amend-session \
  --state-dir "$STATE_DIR" --session "$SESSION" --status exchange_closed \
  --owner-approval "$OWNER_APPROVAL"
# Alternatively, for an official early close:
"$REVIEWED_PYTHON" "$RELEASE_ROOT/scripts/intraday_observe.py" amend-session \
  --state-dir "$STATE_DIR" --session "$SESSION" --status early_close --close 13:00 \
  --owner-approval "$OWNER_APPROVAL"
```

Keep original/revised official source artifacts. Slots remain counted and the
endpoint is not extended. A notice arriving too late remains an operational fault;
never retroactively amend to remove it. Unrepresentable changes to the frozen
calendar/grace require owner review, not silent changes.

## Halts, final recovery and disable

Unknown traded-day valuations or failed checkpoint require review and a future,
owner-approved `amend-resume` before its 15:20 stage. Existing faults remain.
An integrity-failed report contains `status: unresolved`, problems and identities,
with no return statistics. **Current corrupt evidence cannot be waived by resume.**
Preserve an independent copy of damaged bytes and findings; only exact previously
preserved bytes matching their original hashes can restore evidence. No re-fetch,
manifest edit, chain rewrite or invented value repairs it. If exact restoration
is impossible, preserve unresolved state and keep positions halted. After verified
restoration, record the owner's future resume amendment; do not erase the incident.
A manifest/code binding mismatch still refuses normal reporting rather than
trusting evidence with a different identity; preserve its error and files.

Run optional `revise` only outside 15:10–16:30 ET and only with remaining daily
budget. It cannot repair primary values or choose a favorable revision.
After slot 252, retain recovery through the next official session's 15:15 stage;
its first valid final valuation must finish before 15:20. It creates no slot 253.
If the final slot was unexpectedly closed, an older expired recovery can refuse;
record the refusal rather than extend grace. Save final verify/report artifacts.

```bash
systemctl --user disable --now ggtrader-intraday-observer-{recover,inputs,decide,archive}.timer
systemctl --user list-timers --all 'ggtrader-intraday-observer-*'
```

The same command stops collection early. Preserve the state, release, runtime,
credential file and rendered units; no deletion or favorable restart is implied.

## Releases cut after the study-isolation move (S8)

Everything above describes release `280d4a0`, which is the layout the running recorder uses. Its
units, runtime and credentials are unchanged and nothing here edits them. The study-isolation
slice S8 later moved the observer code in this repository: `git log --follow --
research/intraday/spy-momentum-prospective-observer/code/intraday_observer_job.py` names the
commit. A release cut from a checkout at or after that commit has a different layout:

- The wrappers are `research/intraday/spy-momentum-prospective-observer/code/intraday_observer_job.py`
  and `.../intraday_observe.py`; the observer module sits beside them and the probe library is
  `research/intraday/_lib/intraday_probe.py`. There is no `scripts/` or `src/` copy.
- The wrappers put their own folder and `research/intraday/_lib` on `sys.path` using only the
  standard library, so the release runtime needs no installed repository package.
- The `ExecStart` line of each `*.service` template still names `scripts/intraday_observer_job.py`,
  so it must be edited to the new wrapper path before such a release is installed. The templates in
  `deploy/` are left as they are because they describe `280d4a0` and the tests read them.
- The code binding (`CODE_FILES`) now lists the new paths, so a new release has a new code hash.
  Treat it as a new release, not a continuation: this needs the owner's activation approval, like
  any release. Never roll an active inception.
