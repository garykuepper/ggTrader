"""``ggt db load-sp500`` and ``ggt db membership ...``: the CLI face of the membership database."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from ggTrader.data.membership import (
    DbMembershipSource,
    FixtureMembershipSource,
    MembershipError,
    redact,
)
from ggTrader.data.membership_events import (
    EventsFileError,
    PreparedEventsLoad,
    prepare_events_load,
)
from ggTrader.data.membership_export import EXPORT_DIR, ExportError, export_universe
from ggTrader.data.membership_gate import decide_gate, max_verify_age_days
from ggTrader.data.membership_holdings import HoldingsRun, PgHoldingsStore, check_holdings
from ggTrader.data.membership_upstream import (
    CheckRun,
    PgCheckStore,
    check_upstream,
    recent_checks,
)
from ggTrader.data.membership_weekly import PgWeeklyStore, WeeklyRun, plain_text, run_weekly
from ggTrader.data.universe_algebra import HistoryValidationError
from ggTrader.data.universe_db import (
    FILE_SOURCES,
    LoadRefused,
    LoadReport,
    LoadRequest,
    apply_ddl,
    connect,
    diff_vintages,
    git_sha,
    load_vintage,
    read_vintage,
    reinstate_vintage,
    save_report_json,
    verify_vintage,
)
from ggTrader.paper.notifier import DryRunNotifier, TelegramNotifier

#: ``ggt db load-sp500`` exit codes.
EXIT_REFUSED = 2
#: ``ggt db membership check-upstream``: 0 upstream equals the latest vintage, 2 it differs
#: (reported, not loaded), 1 the download or the database failed.
EXIT_UPSTREAM_DIFFERS = 2
#: ``ggt db membership check-holdings``: 0 the list equals the SPDR SPY holdings, 2 it differs
#: (reported, nothing changed), 1 no usable download or the database failed.
EXIT_HOLDINGS_MISMATCH = 2


def _parse_instant(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise argparse.ArgumentTypeError("--retrieved-at needs a zone, e.g. ...T00:00:00Z")
    return value


def register_membership_parsers(db_subs: argparse._SubParsersAction) -> None:
    """Add ``load-sp500`` and ``membership`` to the ``ggt db`` subcommands."""
    load = db_subs.add_parser(
        "load-sp500",
        help="Load an S&P 500 membership file as the next vintage (dry run unless --commit)",
        description=(
            "Validate a date,tickers file (--from-file) or an events file (--from-events) and "
            "store the result as a new vintage in schema universe. Dry run by default: prints "
            "the diff against the latest vintage. --commit writes. An events file is applied "
            "to the latest vintage's last snapshot at its effective dates and stored as a "
            "'curated' vintage together with the events file and the saved evidence pages."
        ),
    )
    source = load.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-file", type=Path, help="CSV or CSV.GZ (upstream fetch is not built)"
    )
    source.add_argument(
        "--from-events",
        type=Path,
        help="Events file (effective_date,action,ticker,announcement_date,evidence_url,"
        "evidence_sha256,note,new_ticker): the curated path",
    )
    load.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help="--from-events: folder with the saved evidence pages (found by the sha256 each "
        "event cites); stored gzipped in universe.load_evidence",
    )
    load.add_argument("--source", choices=FILE_SOURCES, default=None, help="--from-file only")
    load.add_argument("--source-url", default=None, help="Where the bytes came from")
    load.add_argument(
        "--retrieved-at",
        type=_parse_instant,
        default=None,
        help="ISO instant with a zone, e.g. 2026-06-09T00:00:00Z (required with --from-file; "
        "--from-events defaults to now)",
    )
    load.add_argument("--upstream-commit", default=None, help="40-hex upstream git commit")
    load.add_argument("--upstream-etag", default=None)
    load.add_argument("--expect-sha256", default=None, help="sha256 of the uncompressed bytes")
    load.add_argument("--note", default=None)
    load.add_argument(
        "--parent",
        default=None,
        help="Pin the latest vintage seen in the dry run (an id, or 'none' for an empty index)",
    )
    load.add_argument("--force", action="store_true", help="Allow a refused (suspicious) load")
    load.add_argument("--reason", default=None, help="Why --force is right (stored)")
    load.add_argument("--commit", action="store_true", help="Write the vintage")
    load.add_argument("--report-json", type=Path, default=None, help="Save the diff report")

    member = db_subs.add_parser("membership", help="Inspect and verify stored membership vintages")
    subs = member.add_subparsers(dest="membership_command", required=True)
    subs.add_parser("apply-ddl", help="Create/refresh schema universe (idempotent)")
    verify = subs.add_parser(
        "verify", help="Compare a stored vintage with a file, on every snapshot date"
    )
    verify.add_argument("--load-id", type=int, required=True)
    subs.add_parser(
        "check-upstream",
        help="Fetch the public upstream list, compare it with the latest vintage, record the check",
        description=(
            "Fetch fja05680/sp500 at its newest commit, compare it with the latest vintage and "
            "append one row to universe.checks (unchanged, refused = differs, or fetch_failed). "
            "Never loads or changes a vintage. Exit 0 equal, 2 differs, 1 failure."
        ),
    )
    subs.add_parser(
        "check-holdings",
        help="Match the latest vintage against the SPDR SPY holdings file; record the check",
        description=(
            "Download the public SPDR SPY daily holdings workbook, compare its tickers with the "
            "latest vintage's members on the holdings' own as-of date and append one row to "
            "universe.checks (matched, mismatch or fetch_failed) with the raw bytes in "
            "universe.check_blobs. A match is what keeps the live gate open. Never loads or "
            "changes a vintage. Exit 0 matched, 2 mismatch, 1 failure."
        ),
    )
    weekly = subs.add_parser(
        "weekly",
        help="Weekly job: fetch and archive the sources, check the list, alert, PROPOSE changes",
        description=(
            "Fetch and archive the raw bytes of the SPDR SPY holdings, the Wikipedia change "
            "ledger and current list and the chinobing CSVs (universe.check_blobs and "
            "universe.source_snapshots); match the latest vintage against SPY (a match advances "
            "the live gate); compare the other sources (warnings only); apply the refusal "
            "conditions; and on any refusal write a PROPOSED events file under "
            "data/exports/membership_proposals/ for a human to confirm and load. It NEVER loads "
            "a vintage. Dry by default: the message is printed and logged and nothing is sent; "
            "--notify sends it through Telegram. Exit 0 clean (or warnings), 2 a refusal or a "
            "proposal, 1 the job failed."
        ),
    )
    weekly.add_argument(
        "--notify",
        action="store_true",
        help="Send the message through the real TelegramNotifier (default: dry, nothing is sent)",
    )
    weekly.add_argument(
        "--proposal-dir",
        type=Path,
        default=None,
        help="Folder for a proposed events file (default: data/exports/membership_proposals/)",
    )
    weekly.add_argument("--json", type=Path, default=None, help="Also save the run summary here")
    status = subs.add_parser(
        "status",
        help="Latest vintage, last SPY match, known-through date and the live gate verdict",
    )
    status.add_argument(
        "--asof",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="Judge the live gate at this day instead of today (UTC)",
    )
    export = subs.add_parser(
        "export",
        help="Dump schema universe (schema and data) to a timestamped file; prints path, sha256",
        description=(
            "Run pg_dump --schema=universe --format=custom into a new timestamped file and print "
            "its path, size and sha256. A read-only operation; dump files are git-ignored and "
            "never committed. Restore: CREATE EXTENSION btree_gist on the target, then "
            "pg_restore."
        ),
    )
    export.add_argument(
        "--out-dir",
        type=Path,
        default=EXPORT_DIR,
        help="Folder for the dump (default: data/exports/universe/)",
    )
    reinstate = subs.add_parser(
        "reinstate",
        help="Write a NEW vintage with the content of an older one (the rollback; dry run unless "
        "--commit)",
        description=(
            "Create a new vintage (source 'reinstate', parent = the latest vintage) whose content "
            "equals vintage N. Nothing is deleted or edited: the rows the parent has and N lacks "
            "are marked superseded and N's missing intervals are inserted again as new rows. "
            "Dry run by default; --commit writes (one transaction under the loader's advisory "
            "lock). The new vintage has never been matched against the SPY holdings, so the "
            "live gate refuses it until `check-holdings` matches."
        ),
    )
    reinstate.add_argument("--load-id", type=int, required=True, help="The vintage to reinstate")
    reinstate.add_argument("--reason", required=True, help="Why (stored in the new load's note)")
    reinstate.add_argument(
        "--parent",
        default=None,
        help="Pin the latest vintage seen in the dry run (an id); a different latest refuses",
    )
    reinstate.add_argument("--commit", action="store_true", help="Write the vintage")
    reinstate.add_argument("--report-json", type=Path, default=None, help="Save the report")
    diff = subs.add_parser("diff", help="Compare two stored vintages (read-only)")
    diff.add_argument("--from", dest="from_id", type=int, required=True, help="Older load id")
    diff.add_argument("--to", dest="to_id", type=int, required=True, help="Newer load id")
    diff.add_argument("--json", type=Path, default=None, help="Also save the full diff here")
    against = verify.add_mutually_exclusive_group()
    against.add_argument(
        "--against-file",
        type=Path,
        default=None,
        help="Constituent file to compare with (default: the vintage's own archived bytes)",
    )
    against.add_argument(
        "--against-fixture",
        default=None,
        metavar="NAME",
        help="Compare with a fixture from data/universe/membership/MANIFEST.json "
        "(its hashes are checked first), e.g. sp500_v1_2026-06-02",
    )


def _read_input(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def _parent_pin(text: str | None) -> int | str | None:
    if text is None or text == "none":
        return text
    return int(text)


def print_load_report(report: LoadReport) -> None:
    """Print the human summary of a dry run or a committed load."""
    verb = "COMMITTED" if report.committed else "DRY RUN (nothing written)"
    print(f"{verb}: index={report.index_name} no_op={report.no_op}")
    if report.load_id is not None:
        print(f"  load_id            {report.load_id}")
    print(f"  parent             {report.parent_load_id}")
    print(f"  coverage           {report.coverage_start} .. {report.coverage_end}")
    print(f"  snapshot dates     {report.n_snapshot_dates}")
    print(f"  intervals          {report.n_intervals} ({report.n_symbols} symbols)")
    print(f"  members per date   {report.min_members} .. {report.max_members}")
    print(f"  rows added/superseded  {report.rows_added} / {report.rows_superseded}")
    if report.n_evidence or report.n_symbol_changes:
        print(f"  evidence files / symbol changes  {report.n_evidence} / {report.n_symbol_changes}")
    print(f"  raw sha256         {report.raw_sha256}")
    print(f"  content sha256     {report.content_sha256}")
    if report.parent_load_id is not None:
        print(
            f"  dates only new/parent  {len(report.dates_only_in_new)} / "
            f"{len(report.dates_only_in_parent)}; changed dates {len(report.dates_changed)}"
        )
        print(f"  historical rewrites {report.historical_rewrites}")
        print(f"  tail changes        {report.tail_changes}")
        print(f"  renames             {report.renames}")
    for reason in report.refusal_reasons:
        print(f"  needs --force: {reason}")


def _print_events_plan(prepared: PreparedEventsLoad) -> None:
    """Print the dry-run table of an events load: each date's changes and member count."""
    plan = prepared.plan
    print(f"events applied to parent load {prepared.parent_load_id}")
    if plan.parent_last_day is not None:
        print(f"  {plan.parent_last_day}  parent's last snapshot: {plan.parent_members} members")
    print("  date        adds   removes   net  members")
    for change in plan.days:
        print(
            f"  {change.day}  +{len(change.adds):<3}  -{len(change.removes):<3}  "
            f"{change.net:+d}  {change.members:>5}   "
            f"+[{' '.join(change.adds)}] -[{' '.join(change.removes)}]"
        )
    for item in plan.symbol_changes:
        print(
            f"  symbol change {item.old_symbol} -> {item.new_symbol} "
            f"effective {item.effective_date}"
        )
    stored = [f"{item.name} ({item.kind})" for item in prepared.request.evidence]
    print(f"  evidence stored with the vintage: {stored}")
    for warning in prepared.warnings:
        print(f"  warning: {warning}")


def run_load_from_events(args: argparse.Namespace) -> int:
    """Run ``ggt db load-sp500 --from-events``: print the table, refuse on any problem."""
    if args.source is not None or args.upstream_commit or args.upstream_etag:
        print(
            "REFUSED: --source/--upstream-* belong to --from-file; a curated vintage has none",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    if args.force:
        print("REFUSED: --force does not apply to an events file", file=sys.stderr)
        return EXIT_REFUSED
    conn = connect()
    try:
        prepared = prepare_events_load(
            conn,
            events_path=args.from_events,
            evidence_dir=args.evidence_dir,
            parent=_parent_pin(args.parent),  # type: ignore[arg-type]
            retrieved_at=args.retrieved_at,
            commit=args.commit,
            loader_git_sha=git_sha(),
            note=args.note,
            source_url=args.source_url,
            expect_sha256=args.expect_sha256,
        )
        _print_events_plan(prepared)
        if prepared.problems:
            for problem in prepared.problems:
                print(f"REFUSED: {problem}", file=sys.stderr)
            return EXIT_REFUSED
        report = load_vintage(conn, prepared.request)
    except (LoadRefused, HistoryValidationError, EventsFileError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    finally:
        conn.close()
    print_load_report(report)
    if not report.committed:
        print("  (dry run: add --commit to write it)")
    if args.report_json is not None:
        report.save(args.report_json)
        print(f"  report saved to {args.report_json}")
    return 0


def run_load_sp500(args: argparse.Namespace) -> int:
    """Run ``ggt db load-sp500``; returns the process exit code."""
    if args.from_events is not None:
        return run_load_from_events(args)
    if args.retrieved_at is None:
        print("REFUSED: --from-file needs --retrieved-at", file=sys.stderr)
        return EXIT_REFUSED
    request = LoadRequest(
        raw=_read_input(args.from_file),
        source=args.source or "fixture-import",
        source_url=args.source_url or f"file:{args.from_file}",
        retrieved_at=args.retrieved_at,
        upstream_commit=args.upstream_commit,
        upstream_etag=args.upstream_etag,
        note=args.note,
        expect_sha256=args.expect_sha256,
        parent=_parent_pin(args.parent),  # type: ignore[arg-type]
        force=args.force,
        force_reason=args.reason,
        commit=args.commit,
        loader_git_sha=git_sha(),
    )
    conn = connect()
    try:
        report = load_vintage(conn, request)
    except (LoadRefused, HistoryValidationError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    finally:
        conn.close()
    print_load_report(report)
    if args.report_json is not None:
        report.save(args.report_json)
        print(f"  report saved to {args.report_json}")
    return 0


def run_membership(args: argparse.Namespace) -> int:
    """Run ``ggt db membership <apply-ddl|verify>``; returns the process exit code."""
    if args.membership_command == "apply-ddl":
        conn = connect()
        try:
            apply_ddl(conn)
        finally:
            conn.close()
        print("schema universe is up to date")
        return 0
    if args.membership_command == "check-upstream":
        return run_check_upstream(args)
    if args.membership_command == "check-holdings":
        return run_check_holdings(args)
    if args.membership_command == "weekly":
        return run_weekly_command(args)
    if args.membership_command == "status":
        return run_status(args)
    if args.membership_command == "diff":
        return run_diff(args)
    if args.membership_command == "export":
        return run_export(args)
    if args.membership_command == "reinstate":
        return run_reinstate(args)
    path = args.against_file
    if args.against_fixture is not None:
        try:
            path = FixtureMembershipSource(args.against_fixture).path
        except MembershipError as exc:
            print(f"fixture {args.against_fixture!r} cannot be used: {exc}", file=sys.stderr)
            return 1
    conn = connect(read_only=True)
    try:
        report = verify_vintage(conn, args.load_id, path)
    finally:
        conn.close()
    for check in report.checks:
        print(f"  [{'ok' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    print(f"verify load {report.load_id}: {'OK' if report.ok else 'FAILED'}")
    return 0 if report.ok else 1


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _print_check(run: CheckRun) -> None:
    result = run.result
    detail = result.detail
    print(f"upstream check: {result.result.upper()} (recorded outcome: {result.outcome})")
    if result.outcome == "fetch_failed":
        print(f"  reason             {detail.get('reason')}: {detail.get('error')}")
    else:
        print(f"  upstream commit    {result.upstream_commit}")
        print(f"  upstream bytes     {detail.get('upstream_bytes')}")
        print(f"  upstream raw sha   {result.raw_sha256}")
        print(
            f"  upstream coverage  {detail.get('upstream_coverage_start')} .. "
            f"{detail.get('upstream_coverage_end')} "
            f"({detail.get('upstream_n_snapshot_dates')} dates)"
        )
        if "compared_load_id" in detail:
            print(
                f"  compared with      load {detail['compared_load_id']} "
                f"(coverage ends {detail['latest_coverage_end']})"
            )
    if result.result == "differs":
        print(
            f"  dates only upstream/latest  {len(detail.get('dates_only_in_upstream', []))} / "
            f"{len(detail.get('dates_only_in_latest', []))}; "
            f"changed {len(detail.get('dates_changed', []))}"
        )
        print(
            f"  historical rewrites {len(detail.get('historical_rewrites', []))}; "
            f"tail changes {len(detail.get('tail_changes', []))}; "
            f"renames {len(detail.get('renames', []))}"
        )
        for reason in detail.get("would_need_force", []):
            print(f"  a load would need --force: {reason}")
        print("  nothing was loaded or changed; this informational check does NOT reset the gate")
    if run.check_id is not None:
        print(f"  recorded as universe.checks row {run.check_id}")
    if run.record_error:
        print(f"  NOT RECORDED: {run.record_error}", file=sys.stderr)


def run_check_upstream(args: argparse.Namespace) -> int:
    """Run ``ggt db membership check-upstream``; returns the process exit code."""
    store = PgCheckStore()
    try:
        run = check_upstream(store)
    except Exception as exc:  # noqa: BLE001 - the latest vintage could not be read
        first = (str(exc).strip().splitlines() or [""])[0]
        print(f"cannot compare: {type(exc).__name__}: {redact(first, None)}", file=sys.stderr)
        return 1
    _print_check(run)
    if run.record_error:
        return 1
    return run.result.exit_code


def _print_holdings(run: HoldingsRun) -> None:
    result = run.result
    detail = result.detail
    print(f"holdings check: {result.result.upper()} (recorded outcome: {result.outcome})")
    if result.outcome == "fetch_failed":
        print(f"  reason             {detail.get('reason')}: {detail.get('error')}")
        if result.raw is not None:
            print(f"  raw bytes archived {detail.get('raw_bytes')} (sha256 {result.raw_sha256})")
    else:
        print(f"  holdings as of     {detail.get('holdings_as_of')}")
        print(
            f"  holdings tickers   {detail.get('n_holdings')} (excluded {detail.get('n_excluded')})"
        )
        print(f"  raw sha256         {result.raw_sha256} ({detail.get('raw_bytes')} bytes)")
        if "compared_load_id" in detail:
            print(
                f"  compared with      load {detail['compared_load_id']}, snapshot "
                f"{detail.get('compared_snapshot_date')} ({detail.get('n_list')} tickers; "
                f"coverage ends {detail['vintage_coverage_end']})"
            )
    if result.result == "mismatch":
        print(f"  only in our list      {detail.get('only_in_list')}")
        print(f"  only in the holdings  {detail.get('only_in_holdings')}")
        print("  nothing was loaded or changed; the live gate was NOT advanced")
    if result.result == "matched":
        print(f"  the list is attested through {detail.get('known_through')}")
    if run.check_id is not None:
        print(f"  recorded as universe.checks row {run.check_id}")
    if run.record_error:
        print(f"  NOT RECORDED: {run.record_error}", file=sys.stderr)


def run_check_holdings(args: argparse.Namespace) -> int:
    """Run ``ggt db membership check-holdings``; returns the process exit code."""
    store = PgHoldingsStore()
    try:
        run = check_holdings(store)
    except Exception as exc:  # noqa: BLE001 - the latest vintage could not be read
        first = (str(exc).strip().splitlines() or [""])[0]
        print(f"cannot compare: {type(exc).__name__}: {redact(first, None)}", file=sys.stderr)
        return 1
    _print_holdings(run)
    if run.record_error:
        return 1
    return run.result.exit_code


def _print_weekly(run: WeeklyRun) -> None:
    print(f"weekly membership check: {run.status.upper()} (exit {run.exit_code})")
    if run.spy is not None:
        spy = run.spy
        print(
            f"  SPY holdings       {spy.outcome.upper()} as of {spy.holdings_as_of} vs "
            f"load {run.load_id} (attested floor {run.floor})"
        )
    for name, row in run.sources.items():
        changed = row.last_change.isoformat() if row.last_change else "unknown"
        size = "-" if row.size_bytes is None else f"{row.size_bytes} bytes"
        sha = (row.raw_sha256 or "-")[:12]
        print(f"  source {name:<21} {row.status:<6} last change {changed:<10} {size:>14} sha {sha}")
    for finding in run.findings:
        print(f"  [{finding.severity}] {finding.code}: {finding.text}")
    if run.proposal is not None:
        where = run.proposal_path or "(not written)"
        print(f"  proposal: {len(run.proposal.rows)} rows -> {where}")
        for row in run.proposal.rows:
            when = row.effective.isoformat() if row.effective else "unknown"
            flag = "CONFIRM" if row.needs_confirmation else "agreed"
            cells = f"{row.action:<6} {row.ticker} {row.new_ticker} [{row.date_source}]"
            print(f"    {when} {cells} {flag}")
    if run.check_id is not None:
        print(f"  recorded as universe.checks row {run.check_id}; nothing was loaded")
    if run.record_error:
        print(f"  NOT RECORDED: {run.record_error}", file=sys.stderr)


def run_weekly_command(args: argparse.Namespace) -> int:
    """Run ``ggt db membership weekly``; the real Telegram notifier exists only with --notify."""
    notifier = TelegramNotifier() if args.notify else DryRunNotifier(None)
    try:
        run = run_weekly(PgWeeklyStore(), notifier, proposal_dir=args.proposal_dir)
    except Exception as exc:  # noqa: BLE001 - never echo a connection string
        first = (str(exc).strip().splitlines() or [""])[0]
        print(f"weekly check failed: {type(exc).__name__}: {redact(first, None)}", file=sys.stderr)
        return 1
    _print_weekly(run)
    if args.notify:
        print(f"  Telegram: {'sent' if run.delivered else 'NOT SENT (see TELEGRAM_* settings)'}")
        if not run.delivered:
            return 1
    else:
        print("  dry run: nothing was sent. The message would be:")
        print("    " + plain_text(run.message).replace("\n", "\n    "))
    if args.json is not None:
        summary = {
            "status": run.status,
            "message": run.message,
            "check_id": run.check_id,
            "findings": [vars(f) for f in run.findings],
        }
        args.json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return run.exit_code


def run_status(args: argparse.Namespace) -> int:
    """Run ``ggt db membership status``: the latest vintage and the live gate's verdict."""
    asof = args.asof or _today_utc()
    try:
        source = DbMembershipSource("latest")
    except MembershipError as exc:
        print(f"membership unavailable: {exc}", file=sys.stderr)
        print("the live gate would REFUSE (membership_unavailable)")
        return 1
    v = source.vintage()
    print(f"index {v.index_name}, latest load {v.load_id}")
    print(f"  content sha256     {v.content_sha256}")
    print(f"  raw sha256         {v.raw_sha256}")
    print(f"  coverage           {v.coverage_start} .. {v.coverage_end}")
    print(f"  retrieved at       {v.retrieved_at.isoformat()}")
    if v.matched_at is None:
        print("  last matched check none: never matched against the SPDR SPY holdings")
        print("  known through      none (run `ggt db membership check-holdings`)")
    else:
        print(f"  last matched check {v.matched_at.isoformat()}")
        print(f"  known through      {v.known_through}")
    try:
        checks = recent_checks(v.index_name, 3)
    except Exception as exc:  # noqa: BLE001 - informational only
        checks = []
        print(f"  (checks unreadable: {type(exc).__name__})")
    for row in checks:
        print(
            f"  check {row['check_id']:>3}  {row['checked_at'].isoformat()}  "
            f"{row['outcome']:<12} {(row['detail'] or {}).get('result', '')}"
        )
    try:
        limit = max_verify_age_days()
    except MembershipError as exc:
        print(f"gate configuration error: {exc}", file=sys.stderr)
        return 1
    decision = decide_gate(v, asof, max_verify_age=limit)
    print(f"as of {asof}:")
    if decision.age_since_match_days is None:
        print(f"  match age          no matched check (limit {limit}, refuse above)")
    else:
        print(
            f"  match age          {decision.age_since_match_days} days (limit {limit}, "
            f"refuse above)"
        )
    base = "known through" if decision.known_through is not None else "last change (never matched)"
    print(
        f"  lag to {base}  {decision.lag_days} days (warn from {decision.lag_warn_days}, "
        f"refuse from {decision.lag_refuse_days})"
    )
    print(f"  live gate would    {decision.level.upper()}")
    for reason in decision.reasons:
        print(f"    - {reason}")
    return 0


def run_export(args: argparse.Namespace) -> int:
    """Run ``ggt db membership export``; prints the dump's path, size and sha256."""
    try:
        result = export_universe(out_dir=args.out_dir)
    except ExportError as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - e.g. no database configuration; never echo URLs
        first = (str(exc).strip().splitlines() or [""])[0]
        print(f"export failed: {type(exc).__name__}: {redact(first, None)}", file=sys.stderr)
        return 1
    print(f"exported schema universe to {result.path}")
    print(f"  size    {result.size_bytes} bytes")
    print(f"  sha256  {result.sha256}")
    return 0


def run_reinstate(args: argparse.Namespace) -> int:
    """Run ``ggt db membership reinstate --load-id N --reason ...``; dry run unless ``--commit``."""
    conn = connect()
    try:
        report = reinstate_vintage(
            conn,
            load_id=args.load_id,
            reason=args.reason,
            commit=args.commit,
            parent=_parent_pin(args.parent),  # type: ignore[arg-type]
            loader_git_sha=git_sha(),
        )
    except (LoadRefused, HistoryValidationError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    finally:
        conn.close()
    print(f"reinstating load {args.load_id} as a new vintage")
    print_load_report(report)
    if not report.committed:
        print("  (dry run: add --commit to write it)")
    if args.report_json is not None:
        report.save(args.report_json)
        print(f"  report saved to {args.report_json}")
    return 0


def run_diff(args: argparse.Namespace) -> int:
    """Run ``ggt db membership diff --from A --to B``; read-only."""
    conn = connect(read_only=True)
    try:
        with conn.cursor() as cur:
            try:
                older = read_vintage(cur, args.from_id)
                newer = read_vintage(cur, args.to_id)
            except LookupError as exc:
                print(f"{exc}", file=sys.stderr)
                return 1
    finally:
        conn.close()
    report = diff_vintages(older, newer)
    print(f"load {report['from_load_id']} -> {report['to_load_id']}: ", end="")
    print("same content" if report["same_content"] else "content differs")
    print(f"  coverage      {report['coverage_from']} -> {report['coverage_to']}")
    print(
        f"  dates only in from/to {len(report['dates_only_in_from'])} / "
        f"{len(report['dates_only_in_to'])}; changed dates {len(report['dates_changed'])}"
    )
    print(
        f"  intervals only in from/to {report['intervals_only_in_from']} / "
        f"{report['intervals_only_in_to']}"
    )
    print(
        f"  historical rewrites {len(report['historical_rewrites'])}; "
        f"tail changes {len(report['tail_changes'])}; renames {len(report['renames'])}"
    )
    for reason in report["would_need_force"]:
        print(f"  as a successor it would need --force: {reason}")
    if args.json is not None:
        save_report_json(report, args.json)
        print(f"  full diff saved to {args.json}")
    return 0
