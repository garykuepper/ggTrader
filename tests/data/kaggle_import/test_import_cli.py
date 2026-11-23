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
                "--db-url",
                "postgresql://localhost/scratch",
            ]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize("dataset", ["crsp", "mcgill"])
def test_no_backup_refused_for_ggtrader(dataset):
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
            "--db-url",
            "postgresql://localhost/ggtrader",
            "--no-backup-check",
            "--yes",
        ]
    )
    with pytest.raises(SystemExit, match="Backup check"):
        run_db(args)
