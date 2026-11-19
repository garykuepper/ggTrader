"""Check that key CLI commands retain their help interfaces."""

import pytest
from typer.testing import CliRunner

from ggTrader.cli.main import app


@pytest.mark.parametrize(
    ("args", "name"),
    [
        (["--help"], "ggt"),
        (["paper", "--help"], "paper"),
        (["lab", "--help"], "lab"),
        (["db", "--help"], "db"),
        (["db", "truncate", "--help"], "truncate"),
        (["db", "clean", "--help"], "clean"),
    ],
)
def test_selected_ggt_commands_print_help(args: list[str], name: str) -> None:
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert name in result.output
