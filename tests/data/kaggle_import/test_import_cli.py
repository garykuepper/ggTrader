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
    monkeypatch.setenv("TEST_IMPORT_URL", "secret-value")
    assert cmd_db.resolve_import_url(args) == "secret-value"
    run_db(args)
    assert "Would apply" in capsys.readouterr().out
    monkeypatch.setattr(cmd_db, "get_db_connection_string", lambda: "configured")
    args = parser.parse_args(base + ["--use-configured-db"])
    assert cmd_db.resolve_import_url(args) == "configured"
