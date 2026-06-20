"""Test that the paper CLI command is registered and callable."""

from __future__ import annotations

from unittest.mock import patch


class TestPaperCLI:
    def test_paper_command_registered(self):
        from typer.testing import CliRunner

        from ggTrader.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert "paper" in result.output

    @patch("ggTrader.paper.trader.run_paper_trading")
    def test_paper_command_calls_run(self, mock_run):
        mock_run.return_value = {"buys": ["AAPL"], "sells": [], "errors": []}

        from typer.testing import CliRunner

        from ggTrader.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["paper"])
        assert result.exit_code == 0
        mock_run.assert_called_once()
