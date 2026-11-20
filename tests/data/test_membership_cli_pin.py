"""Pins every ``ggt db membership`` subcommand's exit code and first error line (D7a, Task 5).

Written against the unchanged ``cmd_membership.py`` before the repeated connect and refusal blocks
were merged; it must keep passing byte for byte afterwards. A systemd unit reads the exit codes
(0 ok, 1 failure, 2 refused or differs), so a change here is a behavior change, not a refactor.
Every collaborator is a fake: no database, no network, no file outside ``tmp_path``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ggTrader.cli import cmd_membership
from ggTrader.cli.cmd_db import register_db_parser
from ggTrader.data.membership import MembershipError
from ggTrader.data.membership_events import EventsFileError
from ggTrader.data.membership_export import ExportError
from ggTrader.data.membership_weekly import ProposalNotFound
from ggTrader.data.universe_algebra import HistoryValidationError
from ggTrader.data.universe_db import LoadRefused, LoadReport

SUBCOMMANDS = {
    "apply-ddl",
    "verify",
    "check-upstream",
    "check-holdings",
    "weekly",
    "proposal",
    "status",
    "diff",
    "export",
    "reinstate",
}


class FakeConn:
    """A connection that records how it was opened and whether it was closed."""

    def __init__(self, read_only: bool) -> None:
        self.read_only = read_only
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def cursor(self) -> FakeConn:
        return self

    def __enter__(self) -> FakeConn:
        return self

    def __exit__(self, *_: object) -> None:
        return None


@dataclass
class Env:
    """The fakes of one case: every connection opened, and the calls made on the helpers."""

    conns: list[FakeConn] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ggt")
    register_db_parser(parser.add_subparsers(dest="command"))
    return parser


def _parse(*argv: str) -> argparse.Namespace:
    return _parser().parse_args(["db", *argv])


def _report(**kw: Any) -> LoadReport:
    base = dict(
        committed=False, no_op=False, index_name="sp500", load_id=None, parent_load_id=2,
        raw_sha256="a" * 64, content_sha256="b" * 64, coverage_start=date(1996, 1, 2),
        coverage_end=date(2026, 6, 2), parent_coverage=(date(1996, 1, 2), date(2026, 8, 18)),
        n_snapshot_dates=2712, n_intervals=1255, n_symbols=1202, min_members=487,
        max_members=507, rows_added=17, rows_superseded=24,
    )  # fmt: skip
    return LoadReport(**{**base, **kw})


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Env:
    """Replace the database door and the git lookup; no case reaches a real database."""
    state = Env()

    def connect(read_only: bool = False) -> FakeConn:
        conn = FakeConn(read_only)
        state.conns.append(conn)
        return conn

    monkeypatch.setattr(cmd_membership, "connect", connect)
    monkeypatch.setattr(cmd_membership, "git_sha", lambda: "abc")
    return state


def _raise(exc: BaseException):
    def fail(*_args: object, **_kwargs: object) -> None:
        raise exc

    return fail


# Each case: id -> (argv, setup(monkeypatch, env, tmp_path), exit code, first stderr line,
# first stdout line, connection modes opened). ``None`` for a line means that stream is empty.
# The values below were recorded from the unchanged code on 2026-11-20.
def _apply_ddl_ok(mp, env, tmp_path):
    mp.setattr(cmd_membership, "apply_ddl", lambda conn: env.calls.append("apply_ddl"))


def _verify_unknown_fixture(mp, env, tmp_path):
    mp.setattr(cmd_membership, "verify_vintage", _raise(AssertionError("never reached")))


def _verify_failed(mp, env, tmp_path):
    check = SimpleNamespace(ok=False, name="dates", detail="2 differ")
    report = SimpleNamespace(checks=[check], load_id=1, ok=False)
    mp.setattr(cmd_membership, "verify_vintage", lambda conn, load_id, path: report)


def _verify_ok(mp, env, tmp_path):
    check = SimpleNamespace(ok=True, name="dates", detail="all equal")
    report = SimpleNamespace(checks=[check], load_id=1, ok=True)
    mp.setattr(cmd_membership, "verify_vintage", lambda conn, load_id, path: report)


def _check_upstream_cannot_compare(mp, env, tmp_path):
    mp.setattr(cmd_membership, "PgCheckStore", lambda: object())
    mp.setattr(cmd_membership, "check_upstream", _raise(RuntimeError("boom\nsecond line")))


def _check_holdings_cannot_compare(mp, env, tmp_path):
    mp.setattr(cmd_membership, "PgHoldingsStore", lambda: object())
    mp.setattr(cmd_membership, "check_holdings", _raise(RuntimeError("boom\nsecond line")))


def _weekly_failed(mp, env, tmp_path):
    mp.setattr(cmd_membership, "PgWeeklyStore", lambda: object())
    mp.setattr(cmd_membership, "run_weekly", _raise(RuntimeError("boom\nsecond line")))


def _proposal_not_found(mp, env, tmp_path):
    mp.setattr(cmd_membership, "read_stored_proposal", _raise(ProposalNotFound("check 7")))


def _proposal_unreadable(mp, env, tmp_path):
    mp.setattr(cmd_membership, "read_stored_proposal", _raise(RuntimeError("boom\nsecond line")))


def _proposal_not_written(mp, env, tmp_path):
    (tmp_path / "out.csv").write_bytes(b"edited by a human")
    mp.setattr(cmd_membership, "read_stored_proposal", lambda check_id: b"data")


def _proposal_written(mp, env, tmp_path):
    mp.setattr(cmd_membership, "read_stored_proposal", lambda check_id: b"data")


def _status_unavailable(mp, env, tmp_path):
    mp.setattr(cmd_membership, "DbMembershipSource", _raise(MembershipError("no vintage")))


def _diff_missing(mp, env, tmp_path):
    def fake_read(cur, load_id, index_name="sp500"):
        raise LookupError(f"no load {load_id} for index 'sp500'")

    mp.setattr(cmd_membership, "read_vintage", fake_read)


def _export_failed(mp, env, tmp_path):
    mp.setattr(cmd_membership, "export_universe", _raise(ExportError("pg_dump exited 1")))


def _export_crashed(mp, env, tmp_path):
    mp.setattr(cmd_membership, "export_universe", _raise(RuntimeError("boom\nsecond line")))


def _export_ok(mp, env, tmp_path):
    result = SimpleNamespace(path=Path("/x/dump"), size_bytes=5, sha256="c" * 8)
    mp.setattr(cmd_membership, "export_universe", lambda out_dir: result)


def _reinstate_refused(mp, env, tmp_path):
    mp.setattr(cmd_membership, "reinstate_vintage", _raise(LoadRefused(["stale", "again"])))


def _reinstate_invalid(mp, env, tmp_path):
    mp.setattr(cmd_membership, "reinstate_vintage", _raise(HistoryValidationError("bad file")))


def _reinstate_dry(mp, env, tmp_path):
    mp.setattr(cmd_membership, "reinstate_vintage", lambda conn, **kw: _report())


def _load_refused(mp, env, tmp_path):
    mp.setattr(cmd_membership, "load_vintage", _raise(LoadRefused(["stale", "again"])))
    mp.setattr(cmd_membership, "_read_input", lambda path: b"bytes")


def _load_invalid(mp, env, tmp_path):
    mp.setattr(cmd_membership, "load_vintage", _raise(HistoryValidationError("bad file")))
    mp.setattr(cmd_membership, "_read_input", lambda path: b"bytes")


def _load_dry(mp, env, tmp_path):
    mp.setattr(cmd_membership, "load_vintage", lambda conn, request: _report())
    mp.setattr(cmd_membership, "_read_input", lambda path: b"bytes")


def _nothing(mp, env, tmp_path):
    return None


def _events_problems(mp, env, tmp_path):
    prepared = SimpleNamespace(
        plan=SimpleNamespace(parent_last_day=None, days=[], symbol_changes=[]),
        parent_load_id=2,
        request=SimpleNamespace(evidence=[]),
        warnings=[],
        problems=["first problem", "second problem"],
    )
    mp.setattr(cmd_membership, "prepare_events_load", lambda conn, **kw: prepared)


def _events_file_error(mp, env, tmp_path):
    mp.setattr(cmd_membership, "prepare_events_load", _raise(EventsFileError(["bad row", "x"])))


# id: (argv builder, setup, code, first stderr line, first stdout line, connection modes)
CASES = {
    "apply-ddl ok": (
        ["membership", "apply-ddl"],
        _apply_ddl_ok,
        0,
        None,
        "schema universe is up to date",
        [False],
    ),
    "verify unknown fixture": (
        ["membership", "verify", "--load-id", "1", "--against-fixture", "nope"],
        _verify_unknown_fixture,
        1,
        "<fixture>",
        None,
        [],
    ),
    "verify failed": (
        ["membership", "verify", "--load-id", "1"],
        _verify_failed,
        1,
        None,
        "  [FAIL] dates: 2 differ",
        [True],
    ),
    "verify ok": (
        ["membership", "verify", "--load-id", "1"],
        _verify_ok,
        0,
        None,
        "  [ok] dates: all equal",
        [True],
    ),
    "check-upstream cannot compare": (
        ["membership", "check-upstream"],
        _check_upstream_cannot_compare,
        1,
        "cannot compare: RuntimeError: boom",
        None,
        [],
    ),
    "check-holdings cannot compare": (
        ["membership", "check-holdings"],
        _check_holdings_cannot_compare,
        1,
        "cannot compare: RuntimeError: boom",
        None,
        [],
    ),
    "weekly failed": (
        ["membership", "weekly"],
        _weekly_failed,
        1,
        "weekly check failed: RuntimeError: boom",
        None,
        [],
    ),
    "proposal not found": (
        ["membership", "proposal", "--id", "7", "--out", "<tmp>/out.csv"],
        _proposal_not_found,
        1,
        "no proposal: check 7",
        None,
        [],
    ),
    "proposal unreadable": (
        ["membership", "proposal", "--id", "7", "--out", "<tmp>/out.csv"],
        _proposal_unreadable,
        1,
        "cannot read the proposal: RuntimeError: boom",
        None,
        [],
    ),
    "proposal not written": (
        ["membership", "proposal", "--id", "7", "--out", "<tmp>/out.csv"],
        _proposal_not_written,
        1,
        "<cannot write>",
        None,
        [],
    ),
    "proposal written": (
        ["membership", "proposal", "--id", "7", "--out", "<tmp>/out.csv"],
        _proposal_written,
        0,
        None,
        "<wrote>",
        [],
    ),
    "status unavailable": (
        ["membership", "status"],
        _status_unavailable,
        1,
        "membership unavailable: no vintage",
        "the live gate would REFUSE (membership_unavailable)",
        [],
    ),
    "diff missing load": (
        ["membership", "diff", "--from", "1", "--to", "2"],
        _diff_missing,
        1,
        "no load 1 for index 'sp500'",
        None,
        [True],
    ),
    "export failed": (
        ["membership", "export"],
        _export_failed,
        1,
        "export failed: pg_dump exited 1",
        None,
        [],
    ),
    "export crashed": (
        ["membership", "export"],
        _export_crashed,
        1,
        "export failed: RuntimeError: boom",
        None,
        [],
    ),
    "export ok": (
        ["membership", "export"],
        _export_ok,
        0,
        None,
        "exported schema universe to /x/dump",
        [],
    ),
    "reinstate refused": (
        ["membership", "reinstate", "--load-id", "1", "--reason", "undo"],
        _reinstate_refused,
        2,
        "REFUSED: stale; again",
        None,
        [False],
    ),
    "reinstate invalid": (
        ["membership", "reinstate", "--load-id", "1", "--reason", "undo"],
        _reinstate_invalid,
        2,
        "REFUSED: bad file",
        None,
        [False],
    ),
    "reinstate dry run": (
        ["membership", "reinstate", "--load-id", "1", "--reason", "undo"],
        _reinstate_dry,
        0,
        None,
        "reinstating load 1 as a new vintage",
        [False],
    ),
    "load-sp500 file without retrieved-at": (
        ["load-sp500", "--from-file", "f.csv"],
        _nothing,
        2,
        "REFUSED: --from-file needs --retrieved-at",
        None,
        [],
    ),
    "load-sp500 file refused": (
        ["load-sp500", "--from-file", "f.csv", "--retrieved-at", "2026-06-09T00:00:00Z"],
        _load_refused,
        2,
        "REFUSED: stale; again",
        None,
        [False],
    ),
    "load-sp500 file invalid": (
        ["load-sp500", "--from-file", "f.csv", "--retrieved-at", "2026-06-09T00:00:00Z"],
        _load_invalid,
        2,
        "REFUSED: bad file",
        None,
        [False],
    ),
    "load-sp500 file dry run": (
        ["load-sp500", "--from-file", "f.csv", "--retrieved-at", "2026-06-09T00:00:00Z"],
        _load_dry,
        0,
        None,
        "DRY RUN (nothing written): index=sp500 no_op=False",
        [False],
    ),
    "load-sp500 events with upstream option": (
        ["load-sp500", "--from-events", "e.csv", "--source", "fixture-import"],
        _nothing,
        2,
        "REFUSED: --source/--upstream-* belong to --from-file; a curated vintage has none",
        None,
        [],
    ),
    "load-sp500 events with force": (
        ["load-sp500", "--from-events", "e.csv", "--force"],
        _nothing,
        2,
        "REFUSED: --force does not apply to an events file",
        None,
        [],
    ),
    "load-sp500 events with problems": (
        ["load-sp500", "--from-events", "e.csv"],
        _events_problems,
        2,
        "REFUSED: first problem",
        "events applied to parent load 2",
        [False],
    ),
    "load-sp500 events file error": (
        ["load-sp500", "--from-events", "e.csv"],
        _events_file_error,
        2,
        "REFUSED: bad row; x",
        None,
        [False],
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_every_subcommand_keeps_its_exit_code_messages_and_connection_mode(
    name: str,
    env: Env,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv, setup, code, first_err, first_out, modes = CASES[name]
    setup(monkeypatch, env, tmp_path)
    args = _parse(*[a.replace("<tmp>", str(tmp_path)) for a in argv])
    run = (
        cmd_membership.run_membership
        if hasattr(args, "membership_command")
        else cmd_membership.run_load_sp500
    )
    assert run(args) == code
    captured = capsys.readouterr()
    err_lines = captured.err.splitlines()
    out_lines = captured.out.splitlines()
    if first_err is None:
        assert captured.err == ""
    elif first_err == "<fixture>":
        assert err_lines[0].startswith("fixture 'nope' cannot be used: ")
    elif first_err == "<cannot write>":
        assert err_lines[0].startswith(f"cannot write {tmp_path / 'out.csv'}: ")
    else:
        assert err_lines[0] == first_err
    if first_out is None:
        assert captured.out == ""
    elif first_out == "<wrote>":
        assert out_lines[0] == f"wrote 4 bytes of check 7's proposal to {tmp_path / 'out.csv'}"
    else:
        assert out_lines[0] == first_out
    assert [c.read_only for c in env.conns] == modes
    assert all(c.closed for c in env.conns)


def test_the_table_covers_every_membership_subcommand() -> None:
    top = next(a for a in _parser()._actions if isinstance(a, argparse._SubParsersAction))
    db = next(a for a in top.choices["db"]._actions if isinstance(a, argparse._SubParsersAction))
    member = db.choices["membership"]
    registered = set(
        next(a for a in member._actions if isinstance(a, argparse._SubParsersAction)).choices
    )
    assert registered == SUBCOMMANDS
    covered = {c[0][1] for c in CASES.values() if c[0][0] == "membership"}
    # weekly and status success paths are pinned by their own tests below and in the weekly file.
    assert covered | {"weekly"} == SUBCOMMANDS


def test_apply_ddl_runs_on_a_fresh_read_write_connection_and_closes_it(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[FakeConn] = []
    monkeypatch.setattr(cmd_membership, "apply_ddl", lambda conn: seen.append(conn))
    assert cmd_membership.run_membership(_parse("membership", "apply-ddl")) == 0
    assert capsys.readouterr().out == "schema universe is up to date\n"
    assert seen == env.conns and len(seen) == 1 and not seen[0].read_only and seen[0].closed


def test_apply_ddl_closes_the_connection_when_the_ddl_fails(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cmd_membership, "apply_ddl", _raise(RuntimeError("ddl failed")))
    with pytest.raises(RuntimeError, match="ddl failed"):
        cmd_membership.run_membership(_parse("membership", "apply-ddl"))
    assert len(env.conns) == 1 and env.conns[0].closed


def test_weekly_with_notify_returns_1_when_the_message_was_not_delivered_even_if_refused(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = SimpleNamespace(
        status="refuse", exit_code=2, spy=None, sources={}, findings=[], proposal=None,
        proposal_path=None, check_id=None, record_error=None, delivered=False, message="m",
    )  # fmt: skip
    monkeypatch.setattr(cmd_membership, "PgWeeklyStore", lambda: object())
    monkeypatch.setattr(cmd_membership, "TelegramNotifier", lambda: object())
    monkeypatch.setattr(cmd_membership, "run_weekly", lambda *a, **kw: run)
    assert cmd_membership.run_membership(_parse("membership", "weekly", "--notify")) == 1
    assert "Telegram: NOT SENT" in capsys.readouterr().out
    run.delivered = True
    assert cmd_membership.run_membership(_parse("membership", "weekly", "--notify")) == 2


def test_the_status_command_exit_codes_for_a_gate_configuration_error(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    vintage = SimpleNamespace(
        index_name="sp500", load_id=3, content_sha256="c", raw_sha256="r",
        coverage_start=date(2020, 1, 1), coverage_end=date(2026, 1, 1),
        retrieved_at=datetime(2026, 1, 2, tzinfo=timezone.utc), matched_at=None,
        known_through=None, latest_check=None,
    )  # fmt: skip
    monkeypatch.setattr(
        cmd_membership, "DbMembershipSource", lambda spec: SimpleNamespace(vintage=lambda: vintage)
    )
    monkeypatch.setattr(cmd_membership, "recent_checks", lambda index, n: [])
    monkeypatch.setattr(cmd_membership, "max_verify_age_days", _raise(MembershipError("bad env")))
    assert cmd_membership.run_membership(_parse("membership", "status")) == 1
    assert capsys.readouterr().err == "gate configuration error: bad env\n"
