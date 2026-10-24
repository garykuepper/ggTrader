"""Input and NYSE session checks for the read-only A23 command."""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from scripts import a23_rebalance_check as cli


@pytest.mark.parametrize(
    "row",
    [
        "SPY,1,40,2024-01-01,000",
        "SPY,1,40",
        "QQQ,1,40,2024-01-01",
        "SPY,0,40,2024-01-01",
        "SPY,1,0,2024-01-01",
        "SPY,1,40,not-a-date",
        "SPY,1,40,2027-01-01",
    ],
)
def test_read_lots_rejects_bad_rows(tmp_path: Path, row: str) -> None:
    path = tmp_path / "lots.csv"
    path.write_text("symbol,shares,cost_basis_total,acquired_date\n" + row + "\n")
    with pytest.raises(ValueError, match="invalid lots CSV row 2"):
        cli.read_lots(path, date(2026, 9, 30))


@pytest.mark.parametrize("row", ["SPY,40,2024-03-28,000", "SPY,40", "QQQ,40,2024-03-28"])
def test_read_prices_rejects_bad_rows(tmp_path: Path, row: str) -> None:
    path = tmp_path / "prices.csv"
    path.write_text("symbol,close,date\n" + row + "\nSSO,20,2024-03-28\n")
    with pytest.raises(ValueError, match="invalid prices CSV row 2"):
        cli.read_prices(path, date(2024, 3, 30))


@pytest.mark.parametrize(
    ("asof", "session", "next_session"),
    [
        (date(2024, 3, 29), date(2024, 3, 28), date(2024, 4, 1)),
        (date(2024, 3, 30), date(2024, 3, 28), date(2024, 4, 1)),
        (date(2026, 5, 31), date(2026, 5, 29), date(2026, 6, 1)),
        (date(2026, 10, 31), date(2026, 10, 30), date(2026, 11, 2)),
    ],
)
def test_calendar_month_ends_use_last_nyse_session(
    monkeypatch: pytest.MonkeyPatch, asof: date, session: date, next_session: date
) -> None:
    class FutureDate(date):
        @classmethod
        def today(cls) -> date:
            return date(2026, 11, 3)

    monkeypatch.setattr(cli, "date", FutureDate)
    assert cli.session_dates(asof)[:3] == (True, session, next_session)


def test_future_asof_rejected() -> None:
    with pytest.raises(ValueError, match="future"):
        cli.session_dates(date(2099, 1, 1))


def test_month_end_before_close_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Today(date):
        @classmethod
        def today(cls) -> date:
            return date(2026, 10, 30)

    class BeforeClose(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            return datetime(2026, 10, 30, 18, tzinfo=timezone.utc)

    monkeypatch.setattr(cli, "date", Today)
    monkeypatch.setattr(cli, "datetime", BeforeClose)
    with pytest.raises(ValueError, match="month-end close not yet final"):
        cli.session_dates(date(2026, 10, 30))


def test_cli_exits_nonzero_before_month_end_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Today(date):
        @classmethod
        def today(cls) -> date:
            return date(2026, 10, 30)

    class BeforeClose(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            return datetime(2026, 10, 30, 18, tzinfo=timezone.utc)

    lots = tmp_path / "lots.csv"
    lots.write_text("symbol,shares,cost_basis_total,acquired_date\nSPY,8,80,2023-01-01\n")
    monkeypatch.setattr(cli, "date", Today)
    monkeypatch.setattr(cli, "datetime", BeforeClose)
    monkeypatch.setattr(sys, "argv", ["a23_rebalance_check.py", "--lots", str(lots)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code != 0
    assert "no action: month-end close not yet final" in capsys.readouterr().err


def test_mismatched_price_dates_fail_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lots = tmp_path / "lots.csv"
    lots.write_text(
        "symbol,shares,cost_basis_total,acquired_date\nSPY,8,80,2023-01-01\nSSO,2,20,2023-01-01\n"
    )
    prices = tmp_path / "prices.csv"
    prices.write_text("symbol,close,date\nSPY,10,2024-03-28\nSSO,10,2024-03-27\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "a23_rebalance_check.py",
            "--lots",
            str(lots),
            "--prices",
            str(prices),
            "--asof",
            "2024-03-30",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code != 0
    assert "price dates" in capsys.readouterr().err


def test_yfinance_stale_close_fails_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lots = tmp_path / "lots.csv"
    lots.write_text(
        "symbol,shares,cost_basis_total,acquired_date\nSPY,8,80,2023-01-01\nSSO,2,20,2023-01-01\n"
    )
    monkeypatch.setattr(
        cli,
        "fetch_prices",
        lambda _asof: ({"SPY": 10, "SSO": 10}, {"SPY": "2024-03-27", "SSO": "2024-03-27"}),
    )
    monkeypatch.setattr(
        sys, "argv", ["a23_rebalance_check.py", "--lots", str(lots), "--asof", "2024-03-30"]
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code != 0
    assert "price dates" in capsys.readouterr().err


def test_first_session_pre_close_evaluates_prior_month_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """10:00 ET on 2026-10-01 must evaluate the completed 2026-09-30 month-end."""

    class Today(date):
        @classmethod
        def today(cls) -> date:
            return date(2026, 10, 1)

    class Morning(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            return datetime(2026, 10, 1, 14, tzinfo=timezone.utc)

    monkeypatch.setattr(cli, "date", Today)
    monkeypatch.setattr(cli, "datetime", Morning)
    month_end, effective, next_session = cli.session_dates(date(2026, 10, 1))
    assert (month_end, effective, next_session) == (True, date(2026, 9, 30), date(2026, 10, 1))


def test_money_display_never_negative_zero() -> None:
    assert f"{cli._money(-0.000825):,.2f}" == "0.00"
    assert f"{cli._money(-0.01):,.2f}" == "-0.01"
