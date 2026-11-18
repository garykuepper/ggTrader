"""Tests for explicit database wipe confirmation flags."""

import argparse
from unittest.mock import MagicMock, Mock

import pytest

from ggTrader.cli.cmd_db import register_db_parser, run_db


def parse(*args: str) -> argparse.Namespace:
    """Parse database command arguments."""
    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["db", *args])


@pytest.mark.parametrize(
    ("command", "flag", "description"),
    [
        ("truncate", "--yes-wipe-ohlcv", "all rows of ohlcv"),
        ("clean", "--yes", "timestamp < '2010-01-01'"),
    ],
)
def test_without_flag_exits_2_and_creates_no_engine(
    command: str,
    flag: str,
    description: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """A refusal requires no engine or input and describes the deletion."""
    engine = Mock()
    monkeypatch.setattr("ggTrader.cli.cmd_db.create_db_engine_or_exit", engine)
    monkeypatch.setattr("builtins.input", Mock(side_effect=AssertionError("input read")))
    stdin = Mock()
    stdin.read.side_effect = AssertionError("stdin read")
    stdin.readline.side_effect = AssertionError("stdin read")
    monkeypatch.setattr("sys.stdin", stdin)
    with pytest.raises(SystemExit) as exc:
        run_db(parse(command))
    assert exc.value.code == 2
    engine.assert_not_called()
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    assert flag in output and description in output
    if command == "clean":
        assert "symbol ~ '_[0-9]+$'" in output


@pytest.mark.parametrize(
    ("command", "flag", "statements"),
    [
        ("truncate", "--yes-wipe-ohlcv", ["TRUNCATE TABLE ohlcv;"]),
        (
            "clean",
            "--yes",
            [
                "DELETE FROM ohlcv WHERE timestamp < '2010-01-01';",
                "DELETE FROM ohlcv WHERE symbol ~ '_[0-9]+$';",
            ],
        ),
    ],
)
def test_with_flag_runs_sql(
    command: str, flag: str, statements: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirmed commands execute the existing SQL."""
    engine = MagicMock()
    monkeypatch.setattr("ggTrader.cli.cmd_db.create_db_engine_or_exit", Mock(return_value=engine))
    run_db(parse(command, flag))
    execute = engine.begin.return_value.__enter__.return_value.execute
    assert [str(call.args[0]) for call in execute.call_args_list] == statements


@pytest.mark.parametrize(
    ("command", "flag"), [("truncate", "--yes-wipe-ohlcv"), ("clean", "--yes")]
)
def test_help_mentions_flags(command: str, flag: str, capsys: pytest.CaptureFixture) -> None:
    """Each command advertises its confirmation flag."""
    with pytest.raises(SystemExit) as exc:
        parse(command, "--help")
    assert exc.value.code == 0
    assert flag in capsys.readouterr().out
