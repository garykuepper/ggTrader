"""Test that the paper CLI command is registered and callable."""

from __future__ import annotations

from unittest.mock import patch

#: What `ggt paper` must pass: the live S&P 500 list is named at the entry point. Slice M4 moved
#: it from the frozen vintage-1 fixture to the latest database vintage, behind the live gate.
LIVE = "db"


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


class TestPaperCliFlags:
    """`ggt paper` is a read-only dry run unless --live; --live is unchanged."""

    @staticmethod
    def _invoke(args: list[str]):
        from typer.testing import CliRunner

        from ggTrader.cli.main import app

        with patch("ggTrader.paper.trader.run_paper_trading") as run:
            run.return_value = {"buys": [], "sells": [], "errors": [], "untradable": ["AVB"]}
            result = CliRunner().invoke(app, ["paper", *args])
        return result, run

    def test_default_is_a_read_only_dry_run(self):
        result, run = self._invoke([])
        assert result.exit_code == 0
        run.assert_called_once_with(dry_run=True, read_only=True, notify=False, membership=LIVE)
        assert "Untradable (skipped): ['AVB']" in result.output
        assert "Planned buys:" in result.output
        assert "Planned sells:" in result.output

    def test_dry_run_writes_restores_the_old_dry_run(self):
        result, run = self._invoke(["--dry-run-writes"])
        assert result.exit_code == 0
        run.assert_called_once_with(dry_run=True, read_only=False, notify=False, membership=LIVE)

    def test_notify_is_forwarded(self):
        _, run = self._invoke(["--notify"])
        run.assert_called_once_with(dry_run=True, read_only=True, notify=True, membership=LIVE)

    def test_live_leaves_read_only_unset(self):
        result, run = self._invoke(["--live"])
        assert result.exit_code == 0
        run.assert_called_once_with(dry_run=False, read_only=None, notify=False, membership=LIVE)
        assert "Submitted buys:" in result.output
        assert "Submitted sells:" in result.output

    def test_the_live_membership_is_the_latest_database_vintage_never_a_fixture(self):
        from ggTrader.paper.trader import LIVE_MEMBERSHIP_SPEC

        assert LIVE_MEMBERSHIP_SPEC == LIVE == "db"
        assert not LIVE_MEMBERSHIP_SPEC.startswith("fixture:")

    def test_live_with_dry_run_flags_is_refused(self):
        for flag in ("--dry-run-writes", "--notify"):
            result, run = self._invoke(["--live", flag])
            assert result.exit_code != 0
            run.assert_not_called()
