"""Fast, offline tests of ``ggt db membership export`` and the argument handling of ``reinstate``.

``pg_dump`` is replaced by a fake runner that writes the file it is told to write; no database,
no process and no network is touched. The real dump and the reinstate transaction are covered in
``test_membership_reinstate_integration.py`` against a scratch database.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from ggTrader.cli import cmd_membership
from ggTrader.cli.cmd_db import register_db_parser
from ggTrader.data import membership_export as me
from ggTrader.data.membership_export import (
    EXPORT_DIR,
    ExportError,
    dump_command,
    dump_filename,
    export_universe,
    file_sha256,
)
from ggTrader.data.universe_db import LoadRefused, LoadReport, reinstate_vintage

NOW = datetime(2026, 11, 8, 21, 44, 6, tzinfo=timezone.utc)
SECRET = "p@ss/w0rd:x"
URL = "postgresql://ggt_user:p%40ss%2Fw0rd%3Ax@db.example.test:5433/ggtrader"
PAYLOAD = b"PGDMP-fake-custom-archive" * 100


class FakeRunner:
    """A ``subprocess.run`` stand-in: records the call and writes the dump file it was given."""

    def __init__(self, *, returncode: int = 0, stderr: str = "", payload: bytes | None = PAYLOAD):
        self.returncode, self.stderr, self.payload = returncode, stderr, payload
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        target = next(a.split("=", 1)[1] for a in argv if a.startswith("--file="))
        if self.payload is not None:
            Path(target).write_bytes(self.payload)
        return subprocess.CompletedProcess(argv, self.returncode, "", self.stderr)


# --------------------------------------------------------------------------- command line


def test_the_file_name_sorts_by_time_and_needs_an_aware_instant():
    assert dump_filename(NOW) == "universe_20261108T214406Z.dump"
    local = datetime(2026, 11, 8, 13, 44, 6, tzinfo=timezone(timedelta(hours=-8)))
    assert dump_filename(local) == "universe_20261108T214406Z.dump"
    with pytest.raises(ValueError, match="timezone-aware"):
        dump_filename(datetime(2026, 11, 8, 21, 44, 6))


def test_the_dump_command_dumps_only_schema_universe_and_keeps_the_password_off_the_line():
    argv, env = dump_command(URL, Path("/x/out.partial"))
    assert argv[0] == "pg_dump"
    assert "--schema=universe" in argv and "--format=custom" in argv
    assert "--file=/x/out.partial" in argv
    assert "--host=db.example.test" in argv and "--port=5433" in argv
    assert "--username=ggt_user" in argv and "--dbname=ggtrader" in argv
    assert env == {"PGPASSWORD": SECRET}
    joined = " ".join(argv)
    assert SECRET not in joined and "p%40ss" not in joined and "@" not in joined
    assert not any(a.startswith(("--table", "-t", "--exclude")) for a in argv)


def test_sslmode_travels_in_the_environment_and_a_missing_database_is_refused():
    _, env = dump_command(URL + "?sslmode=require", Path("o"))
    assert env["PGSSLMODE"] == "require"
    with pytest.raises(ExportError, match="no database"):
        dump_command("postgresql://u:p@h:5432/", Path("o"))
    argv, env = dump_command("postgresql:///ggtrader", Path("o"))
    assert env == {} and not any(a.startswith(("--host", "--port", "--username")) for a in argv)


# --------------------------------------------------------------------------- the export


def test_a_successful_export_returns_path_size_and_sha256(tmp_path):
    runner = FakeRunner()
    result = export_universe(URL, out_dir=tmp_path / "universe", now=NOW, runner=runner)
    assert result.path == tmp_path / "universe" / "universe_20261108T214406Z.dump"
    assert result.path.read_bytes() == PAYLOAD
    assert result.size_bytes == len(PAYLOAD)
    assert result.sha256 == hashlib.sha256(PAYLOAD).hexdigest() == file_sha256(result.path)
    assert not list((tmp_path / "universe").glob("*.partial"))  # renamed, not left behind
    ((argv, kwargs),) = runner.calls
    assert kwargs["env"]["PGPASSWORD"] == SECRET and SECRET not in " ".join(argv)
    assert kwargs["check"] is False and kwargs["timeout"] == me.DUMP_TIMEOUT_SECONDS


def test_an_existing_file_is_never_overwritten(tmp_path):
    export_universe(URL, out_dir=tmp_path, now=NOW, runner=FakeRunner())
    runner = FakeRunner(payload=b"other")
    with pytest.raises(ExportError, match="already exists"):
        export_universe(URL, out_dir=tmp_path, now=NOW, runner=runner)
    assert runner.calls == []
    assert (tmp_path / "universe_20261108T214406Z.dump").read_bytes() == PAYLOAD


def test_a_failed_dump_leaves_nothing_and_its_message_hides_the_credentials(tmp_path):
    stderr = (
        f"pg_dump: error: connection to server at db.example.test failed: password {SECRET} "
        f"for user ggt_user rejected, url {URL}\n"
    )
    runner = FakeRunner(returncode=1, stderr=stderr)
    with pytest.raises(ExportError, match="exit 1") as caught:
        export_universe(URL, out_dir=tmp_path, now=NOW, runner=runner)
    text = str(caught.value)
    assert SECRET not in text and "ggt_user" not in text and "db.example.test" not in text
    assert "p%40ss" not in text and "postgresql://" not in text
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("payload", [None, b""])
def test_success_without_data_is_an_error_not_an_empty_backup(tmp_path, payload):
    with pytest.raises(ExportError, match="wrote no data"):
        export_universe(URL, out_dir=tmp_path, now=NOW, runner=FakeRunner(payload=payload))
    assert list(tmp_path.iterdir()) == []


def test_a_missing_pg_dump_and_a_timeout_are_clear_errors(tmp_path):
    def missing(*_a, **_k):
        raise FileNotFoundError("pg_dump")

    def slow(argv, **_k):
        next_file = next(a.split("=", 1)[1] for a in argv if a.startswith("--file="))
        Path(next_file).write_bytes(b"half")
        raise subprocess.TimeoutExpired(argv, 1)

    with pytest.raises(ExportError, match="not installed"):
        export_universe(URL, out_dir=tmp_path, now=NOW, runner=missing)
    with pytest.raises(ExportError, match="did not finish"):
        export_universe(URL, out_dir=tmp_path, now=NOW, runner=slow)
    assert list(tmp_path.iterdir()) == []


def test_the_default_folder_is_git_ignored_and_under_data_exports():
    assert EXPORT_DIR.parts[-3:] == ("data", "exports", "universe")


# --------------------------------------------------------------------------- the CLI


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ggt")
    register_db_parser(parser.add_subparsers(dest="command"))
    return parser


@pytest.mark.parametrize(
    "argv",
    [["db", "membership", "export", "--help"], ["db", "membership", "reinstate", "--help"]],
)
def test_the_new_commands_answer_help(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _parser().parse_args(argv)
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out and ("--out-dir" in out or "--load-id" in out)


def test_export_arguments_default_to_the_ignored_folder():
    args = _parser().parse_args(["db", "membership", "export"])
    assert args.membership_command == "export" and args.out_dir == EXPORT_DIR
    args = _parser().parse_args(["db", "membership", "export", "--out-dir", "/tmp/x"])
    assert args.out_dir == Path("/tmp/x")


def test_export_command_prints_path_size_and_sha256_and_no_url(monkeypatch, capsys, tmp_path):
    dump = tmp_path / "universe_x.dump"
    dump.write_bytes(PAYLOAD)
    seen = {}

    def fake(out_dir):
        seen["out_dir"] = out_dir
        return me.ExportResult(dump, len(PAYLOAD), hashlib.sha256(PAYLOAD).hexdigest())

    monkeypatch.setattr(cmd_membership, "export_universe", lambda out_dir: fake(out_dir))
    args = _parser().parse_args(["db", "membership", "export", "--out-dir", str(tmp_path)])
    assert cmd_membership.run_membership(args) == 0
    out = capsys.readouterr().out
    assert str(dump) in out and f"{len(PAYLOAD)} bytes" in out
    assert hashlib.sha256(PAYLOAD).hexdigest() in out and seen["out_dir"] == tmp_path


@pytest.mark.parametrize(
    "error", [ExportError("pg_dump failed with exit 1: boom"), ValueError("no database config")]
)
def test_export_command_failure_is_exit_1_on_stderr(monkeypatch, capsys, error):
    def boom(out_dir):
        raise error

    monkeypatch.setattr(cmd_membership, "export_universe", boom)
    args = _parser().parse_args(["db", "membership", "export"])
    assert cmd_membership.run_membership(args) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "export failed" in captured.err


# --------------------------------------------------------------------------- reinstate arguments


def test_reinstate_needs_a_load_id_and_a_reason():
    parser = _parser()
    for argv in (
        ["db", "membership", "reinstate"],
        ["db", "membership", "reinstate", "--load-id", "1"],
        ["db", "membership", "reinstate", "--reason", "x"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
    args = parser.parse_args(
        ["db", "membership", "reinstate", "--load-id", "1", "--reason", "undo v2", "--parent", "2"]
    )
    assert (args.load_id, args.reason, args.parent, args.commit) == (1, "undo v2", "2", False)


@pytest.mark.parametrize("reason", ["", "   "])
def test_an_empty_reason_is_refused_before_the_database_is_touched(reason):
    with pytest.raises(LoadRefused, match="--reason is required"):
        reinstate_vintage(None, load_id=1, reason=reason)  # type: ignore[arg-type]


def test_a_naive_clock_is_refused_before_the_database_is_touched():
    with pytest.raises(LoadRefused, match="timezone-aware"):
        reinstate_vintage(None, load_id=1, reason="x", now=datetime(2026, 11, 8))  # type: ignore[arg-type]


class _FakeConn:
    closed = False

    def close(self) -> None:
        self.closed = True


def _report(**kw) -> LoadReport:
    base = dict(
        committed=False, no_op=False, index_name="sp500", load_id=None, parent_load_id=2,
        raw_sha256="a" * 64, content_sha256="b" * 64, coverage_start=date(1996, 1, 2),
        coverage_end=date(2026, 6, 2), parent_coverage=(date(1996, 1, 2), date(2026, 8, 18)),
        n_snapshot_dates=2712, n_intervals=1255, n_symbols=1202, min_members=487,
        max_members=507, rows_added=17, rows_superseded=24,
    )  # fmt: skip
    return LoadReport(**{**base, **kw})


def test_the_reinstate_command_is_a_dry_run_until_commit(monkeypatch, capsys):
    calls = []
    conn = _FakeConn()
    monkeypatch.setattr(cmd_membership, "connect", lambda: conn)
    monkeypatch.setattr(cmd_membership, "git_sha", lambda: "abc")

    def fake(c, **kwargs):
        calls.append(kwargs)
        return _report(committed=kwargs["commit"], load_id=3 if kwargs["commit"] else None)

    monkeypatch.setattr(cmd_membership, "reinstate_vintage", fake)
    parse = _parser().parse_args
    dry = parse(["db", "membership", "reinstate", "--load-id", "1", "--reason", "undo v2"])
    assert cmd_membership.run_membership(dry) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "add --commit" in out
    assert calls[0]["commit"] is False and calls[0]["load_id"] == 1 and conn.closed
    live = parse(
        ["db", "membership", "reinstate", "--load-id", "1", "--reason", "undo v2", "--commit",
         "--parent", "2"]
    )  # fmt: skip
    assert cmd_membership.run_membership(live) == 0
    assert "load_id            3" in capsys.readouterr().out
    assert calls[1]["commit"] is True and calls[1]["parent"] == 2


def test_a_refused_reinstate_is_exit_2_with_the_reason_on_stderr(monkeypatch, capsys):
    monkeypatch.setattr(cmd_membership, "connect", lambda: _FakeConn())
    monkeypatch.setattr(cmd_membership, "git_sha", lambda: "abc")

    def refuse(c, **kwargs):
        raise LoadRefused(["the latest vintage 2 already has the content of 1"])

    monkeypatch.setattr(cmd_membership, "reinstate_vintage", refuse)
    args = _parser().parse_args(
        ["db", "membership", "reinstate", "--load-id", "1", "--reason", "undo"]
    )
    assert cmd_membership.run_membership(args) == cmd_membership.EXIT_REFUSED == 2
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err and captured.out == ""
