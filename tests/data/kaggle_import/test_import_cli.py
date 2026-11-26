"""Import CLI safety checks without connecting to a database."""

import argparse

import pytest

from ggTrader.cli.cmd_db import register_db_parser, run_db


@pytest.mark.parametrize("dataset", ["crsp", "mcgill"])
def test_backup_choice_required(dataset):
    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "db",
                "import",
                dataset,
                "--file",
                "x",
                "--sums",
                "x",
                "--db-url-env",
                "TEST_IMPORT_URL",
            ]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize("dataset", ["crsp", "mcgill"])
def test_no_backup_refused_for_ggtrader(dataset, monkeypatch):
    monkeypatch.setenv("TEST_IMPORT_URL", "postgresql://localhost/ggtrader")
    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "db",
            "import",
            dataset,
            "--file",
            "x",
            "--sums",
            "x",
            "--db-url-env",
            "TEST_IMPORT_URL",
            "--no-backup-check",
            "--yes",
        ]
    )
    with pytest.raises(SystemExit, match="Backup check"):
        run_db(args)


def test_target_resolution_and_dry_run(monkeypatch, capsys):
    from ggTrader.cli import cmd_db

    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    base = ["db", "import", "apply-ddl"]
    for flags in ([], ["--db-url-env", "TEST_IMPORT_URL", "--use-configured-db"]):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(base + flags)
        assert exc.value.code == 2
    args = parser.parse_args(base + ["--db-url-env", "TEST_IMPORT_URL"])
    monkeypatch.delenv("TEST_IMPORT_URL", raising=False)
    with pytest.raises(SystemExit, match="TEST_IMPORT_URL"):
        cmd_db.resolve_import_url(args)
    monkeypatch.setenv("TEST_IMPORT_URL", "postgresql://localhost/scratch")
    assert cmd_db.resolve_import_url(args) == "postgresql://localhost/scratch"
    run_db(args)
    assert "Would apply" in capsys.readouterr().out
    monkeypatch.setattr(
        cmd_db, "get_db_connection_string", lambda: "postgresql://localhost/configured"
    )
    args = parser.parse_args(base + ["--use-configured-db"])
    assert cmd_db.resolve_import_url(args) == "postgresql://localhost/configured"


def test_release_cli_parsing_and_requires_confirmation(monkeypatch, capsys):
    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    monkeypatch.setenv("TEST_IMPORT_URL", "postgresql://localhost/ggtrader")
    for command in (
        ["build", "--crsp-bars-load", "1", "--links", "x.csv"],
        ["withdraw", "--release-id", "1"],
    ):
        args = parser.parse_args(
            ["db", "import", "release", *command, "--db-url-env", "TEST_IMPORT_URL"]
        )
        run_db(args)
        assert "--yes" in capsys.readouterr().out
    assert (
        parser.parse_args(
            ["db", "import", "release", "list", "--db-url-env", "TEST_IMPORT_URL"]
        ).release_command
        == "list"
    )


def test_approved_link_settings():
    from datetime import date

    from ggTrader.data.kaggle_import.links import APPROVED_LINK_SETTINGS

    s = APPROVED_LINK_SETTINGS
    assert (s.min_sessions, s.min_sessions_short, s.agreement_bar, s.margin) == (
        250,
        40,
        0.999,
        0.0005,
    )
    assert (s.window_start, s.window_end, s.hold_first_days) == (
        date(2005, 1, 1),
        date(2024, 12, 31),
        60,
    )


def test_release_commands_have_help(capsys):
    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    for command in ("build", "withdraw", "list"):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["db", "import", "release", command, "--help"])
        assert exc.value.code == 0
        assert "--db-url-env" in capsys.readouterr().out


SEC_COMMANDS = ("sec-cache", "sec-index", "sec-headers", "sec-labels")


def sec_source(command, path):
    if command == "sec-labels":
        return ["--study", "early", "--file", str(path / "labels.csv")]
    return ["--folder", str(path)]


@pytest.mark.parametrize("command", SEC_COMMANDS)
def test_sec_commands_require_explicit_target(command, tmp_path):
    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["db", "import", command, *sec_source(command, tmp_path), "--backup-dir", "unused"]
        )


@pytest.mark.parametrize("command", SEC_COMMANDS)
def test_sec_commands_require_backup_dir(command, tmp_path):
    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    base = ["db", "import", command, *sec_source(command, tmp_path), "--use-configured-db"]
    for flags in ([], ["--no-backup-check"], ["--backup-dir", "unused", "--no-backup-check"]):
        with pytest.raises(SystemExit):
            parser.parse_args(base + flags)


@pytest.mark.parametrize("command", SEC_COMMANDS)
def test_sec_commands_without_yes_write_nothing(command, tmp_path, monkeypatch, capsys):
    from ggTrader.data import universe_db

    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    monkeypatch.setenv("TEST_IMPORT_URL", "postgresql://localhost/scratch")
    (tmp_path / "labels.csv").write_text("permno,last_trading_day,label\n", encoding="utf-8")
    monkeypatch.setattr(universe_db, "connect", lambda *_: pytest.fail("opened database"))
    try:
        args = parser.parse_args(
            [
                "db",
                "import",
                command,
                *sec_source(command, tmp_path),
                "--backup-dir",
                str(tmp_path / "missing"),
                "--db-url-env",
                "TEST_IMPORT_URL",
            ]
        )
    except SystemExit:
        pytest.fail(f"{command} must be available")
    run_db(args)
    assert "pass --yes" in capsys.readouterr().out


def test_sec_labels_requires_study_and_file():
    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    base = ["db", "import", "sec-labels", "--backup-dir", "x", "--use-configured-db"]
    for flags in ([], ["--study", "early"], ["--file", "x"]):
        with pytest.raises(SystemExit):
            parser.parse_args(base + flags)


@pytest.mark.parametrize("command", SEC_COMMANDS)
def test_sec_confirmed_dispatch(command, tmp_path, monkeypatch, capsys):
    from ggTrader.cli import sec_import_cmds as sec

    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    monkeypatch.setenv("TEST_IMPORT_URL", "postgresql://localhost/scratch")
    (tmp_path / "labels.csv").write_text("permno,last_trading_day,label\n", encoding="utf-8")
    calls = []

    class FakeConnection:
        def close(self):
            calls.append("closed")

    monkeypatch.setattr(sec, "connect", lambda _: FakeConnection())
    monkeypatch.setattr(sec, "assert_connected_database", lambda conn, name: calls.append(name))

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return 42

    monkeypatch.setattr(
        sec, command.replace("sec-", "load_").replace("load_labels", "load_labels"), load
    )
    flags = ["--only-quarter", "2005Q2"] if command == "sec-index" else []
    args = parser.parse_args(
        [
            "db",
            "import",
            command,
            *sec_source(command, tmp_path),
            "--backup-dir",
            str(tmp_path),
            "--db-url-env",
            "TEST_IMPORT_URL",
            "--yes",
            *flags,
        ]
    )
    run_db(args)
    assert calls[0] == "scratch"
    assert calls[1][1]["backup_dir"] == tmp_path
    assert calls[1][1]["db_free_path"].as_posix() == "/"
    if command == "sec-index":
        assert calls[1][1]["only_quarters"] == ["2005Q2"]
    if command == "sec-labels":
        assert calls[1][1]["release_id"] == 1
    assert calls[-1] == "closed"
    assert "Load 42" in capsys.readouterr().out


def test_sec_cache_corruption_refuses_dry_run(tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    monkeypatch.setenv("TEST_IMPORT_URL", "postgresql://localhost/scratch")
    (tmp_path / ("a" * 64 + ".body")).write_text("incomplete", encoding="utf-8")
    args = parser.parse_args(
        [
            "db",
            "import",
            "sec-cache",
            "--folder",
            str(tmp_path),
            "--backup-dir",
            "unused",
            "--db-url-env",
            "TEST_IMPORT_URL",
        ]
    )
    with pytest.raises(SystemExit, match="SEC import failed"):
        run_db(args)


def test_sec_labels_reports_unmatched_rows(tmp_path, monkeypatch, capsys):
    from datetime import date
    from types import SimpleNamespace

    from ggTrader.cli import sec_import_cmds as sec

    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    monkeypatch.setenv("TEST_IMPORT_URL", "postgresql://localhost/scratch")
    (tmp_path / "labels.csv").write_text("permno,last_trading_day,label\n", encoding="utf-8")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(sec, "connect", lambda _: conn)
    monkeypatch.setattr(sec, "assert_connected_database", lambda *_: None)

    def reject(*args, **kwargs):
        raise sec.LabelMismatch(
            [SimpleNamespace(permno=123, last_date=date(2020, 1, 1), label="x")]
        )

    monkeypatch.setattr(sec, "load_labels", reject)
    args = parser.parse_args(
        [
            "db",
            "import",
            "sec-labels",
            *sec_source("sec-labels", tmp_path),
            "--backup-dir",
            "x",
            "--db-url-env",
            "TEST_IMPORT_URL",
            "--yes",
        ]
    )
    with pytest.raises(SystemExit, match="do not match"):
        run_db(args)
    assert "123" in capsys.readouterr().out
