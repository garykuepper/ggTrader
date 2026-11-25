"""Tests for per-stock dead-stock exit prices and payout-table coverage."""

from __future__ import annotations

import ft_replay
import numpy as np
import pandas as pd
import payout_prices
import pytest

from ggTrader.lab.dead_stocks import SIM_PRICE_FLOOR

pytestmark = pytest.mark.study


def _sample_close_and_spans() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small 6-stock, 10-day close frame and spans with dying and living stocks."""
    dates = pd.bdate_range("2020-01-02", periods=10)
    close = pd.DataFrame(
        {
            "PERMNO-10": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "PERMNO-20": [80.0, 81.0, 82.0, 83.0, 84.0, 85.0, np.nan, np.nan, np.nan, np.nan],
            "PERMNO-5~2": [50.0, 52.0, 54.0, 56.0, 58.0, 60.0, np.nan, np.nan, np.nan, np.nan],
            "PERMNO-5~3": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            "PERMNO-7~1": [40.0, 42.0, 44.0, 46.0, 50.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            "ALIVE": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
        },
        index=dates,
    )
    spans = pd.DataFrame(
        [
            {
                "symbol": "PERMNO-10",
                "last_date": dates[4],
                "has_successor": False,
                "total_loss": False,
            },
            {
                "symbol": "PERMNO-20",
                "last_date": dates[5],
                "has_successor": False,
                "total_loss": False,
            },
            {
                "symbol": "PERMNO-5~2",
                "last_date": dates[5],
                "has_successor": False,
                "total_loss": False,
            },
            {
                "symbol": "PERMNO-7~1",
                "last_date": dates[4],
                "has_successor": True,
                "total_loss": False,
            },
        ]
    )
    return close, spans


def _sample_rows() -> list[dict[str, str]]:
    """Build synthetic payout-table rows covering every confidence and loss case."""
    return [
        {"symbol": "PERMNO-10", "suggested_loss": "0.0", "match_confidence": "confirmed"},
        {"symbol": "PERMNO-20", "suggested_loss": "0.0", "match_confidence": "name_only"},
        {"symbol": "PERMNO-30", "suggested_loss": "", "match_confidence": "confirmed"},
        {"symbol": "PERMNO-40", "suggested_loss": "", "match_confidence": "name_only"},
        {"symbol": "PERMNO-50", "suggested_loss": "", "match_confidence": ""},
    ]


def test_exit_prices_pins_only_the_last_bar_of_a_pinned_dying_stock() -> None:
    close, spans = _sample_close_and_spans()
    result = payout_prices.exit_prices(close, spans, reading=1.0, pinned={"PERMNO-10": 0.25})
    # Earlier bars of PERMNO-10 stay at their original closes; only the last bar is pinned.
    assert result["PERMNO-10"].iloc[:4].tolist() == [100.0, 101.0, 102.0, 103.0]
    assert result["PERMNO-10"].iloc[4] == pytest.approx(78.0)  # 104.0 * (1 - 0.25)
    assert result["PERMNO-10"].iloc[5:].isna().all()
    # Living stock stays untouched.
    pd.testing.assert_series_equal(result["ALIVE"], close["ALIVE"])


def test_exit_prices_unpinned_stock_follows_reading() -> None:
    close, spans = _sample_close_and_spans()
    half = payout_prices.exit_prices(close, spans, reading=0.5, pinned={"PERMNO-10": 0.0})
    full_loss = payout_prices.exit_prices(close, spans, reading=1.0, pinned={"PERMNO-10": 0.0})
    no_loss = payout_prices.exit_prices(close, spans, reading=0.0, pinned={"PERMNO-10": 0.0})
    assert half["PERMNO-20"].iloc[5] == pytest.approx(42.5)  # 85.0 * 0.5
    assert full_loss["PERMNO-20"].iloc[5] == pytest.approx(85.0 * SIM_PRICE_FLOOR)
    assert no_loss["PERMNO-20"].iloc[5] == pytest.approx(85.0)
    # Pinned stock keeps its pinned price regardless of the reading.
    assert half["PERMNO-10"].iloc[4] == pytest.approx(104.0)
    assert full_loss["PERMNO-10"].iloc[4] == pytest.approx(104.0)


def test_later_spell_column_maps_to_base_symbol() -> None:
    close, spans = _sample_close_and_spans()
    assert payout_prices.base_symbol("PERMNO-5~2") == "PERMNO-5"
    assert payout_prices.base_symbol("AAPL~10") == "AAPL"
    assert payout_prices.base_symbol("PERMNO-5") == "PERMNO-5"
    assert payout_prices.base_symbol("BRK-B") == "BRK-B"
    result = payout_prices.exit_prices(close, spans, reading=1.0, pinned={"PERMNO-5": 0.2})
    # PERMNO-5~2 dies on index 5 at 60.0 and is pinned to 0.2 -> 48.0.
    assert result["PERMNO-5~2"].iloc[5] == pytest.approx(48.0)
    assert result["PERMNO-5~2"].iloc[:5].tolist() == [50.0, 52.0, 54.0, 56.0, 58.0]
    # PERMNO-5~3 shares the base symbol but stays alive, so every bar is untouched.
    pd.testing.assert_series_equal(result["PERMNO-5~3"], close["PERMNO-5~3"])


def test_successor_spell_is_not_written_down_even_if_pinned() -> None:
    close, spans = _sample_close_and_spans()
    result = payout_prices.exit_prices(close, spans, reading=1.0, pinned={"PERMNO-7": 0.4})
    # PERMNO-7~1 has has_successor=True, so exit_close leaves its last close (50.0) intact.
    assert result["PERMNO-7~1"].iloc[4] == pytest.approx(50.0)


def test_pinned_losses_confirmed_excludes_name_only() -> None:
    rows = _sample_rows()
    assert payout_prices.pinned_losses(rows, "confirmed") == {"PERMNO-10": 0.0}


def test_pinned_losses_all_includes_name_only_but_not_blank_loss() -> None:
    rows = _sample_rows()
    assert payout_prices.pinned_losses(rows, "all") == {"PERMNO-10": 0.0, "PERMNO-20": 0.0}


def test_pinned_losses_none_is_empty_and_unknown_set_raises() -> None:
    rows = _sample_rows()
    assert payout_prices.pinned_losses(rows, "none") == {}
    assert payout_prices.ANCHOR_SET == "none"
    assert payout_prices.EVIDENCE_SETS == ("confirmed", "all")
    assert payout_prices.READINGS == (1.0, 0.5, 0.0)
    with pytest.raises(ValueError, match="evidence"):
        payout_prices.pinned_losses(rows, "unknown_set")


def test_conflicting_duplicate_symbol_raises() -> None:
    rows = [
        {"symbol": "PERMNO-10", "suggested_loss": "0.0", "match_confidence": "confirmed"},
        {"symbol": "PERMNO-10", "suggested_loss": "0.5", "match_confidence": "confirmed"},
    ]
    with pytest.raises(ValueError, match="PERMNO-10"):
        payout_prices.pinned_losses(rows, "confirmed")
    with pytest.raises(ValueError, match="PERMNO-10"):
        payout_prices.pinned_losses(rows, "all")


def test_same_loss_twice_is_accepted() -> None:
    rows = [
        {"symbol": "PERMNO-10", "suggested_loss": "0.25", "match_confidence": "confirmed"},
        {"symbol": "PERMNO-10", "suggested_loss": "0.25", "match_confidence": "confirmed"},
        {"symbol": "PERMNO-10", "suggested_loss": "", "match_confidence": "confirmed"},
    ]
    assert payout_prices.pinned_losses(rows, "confirmed") == {"PERMNO-10": 0.25}


@pytest.mark.parametrize("bad_loss", ["-0.01", "1.01", "nan", "inf", "not_a_number"])
def test_loss_outside_zero_one_raises(bad_loss: str) -> None:
    rows = [{"symbol": "PERMNO-10", "suggested_loss": bad_loss, "match_confidence": "confirmed"}]
    with pytest.raises(ValueError):
        payout_prices.pinned_losses(rows, "confirmed")


def test_empty_pin_set_reproduces_scalar_exit_close() -> None:
    close, spans = _sample_close_and_spans()
    close_before = close.copy(deep=True)
    spans_before = spans.copy(deep=True)
    for reading in payout_prices.READINGS:
        got = payout_prices.exit_prices(close, spans, reading, {})
        want = ft_replay.exit_close(close, spans, reading)
        pd.testing.assert_frame_equal(got, want)
    pd.testing.assert_frame_equal(close, close_before)
    pd.testing.assert_frame_equal(spans, spans_before)


def test_one_exit_close_per_distinct_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    close, spans = _sample_close_and_spans()
    seen_losses: list[float] = []
    real_exit_close = ft_replay.exit_close

    def counting_exit_close(
        frame: pd.DataFrame, span_frame: pd.DataFrame, loss: float
    ) -> pd.DataFrame:
        seen_losses.append(loss)
        return real_exit_close(frame, span_frame, loss)

    monkeypatch.setattr(ft_replay, "exit_close", counting_exit_close)
    pinned = {"PERMNO-10": 0.0, "PERMNO-20": 0.0, "PERMNO-5": 0.3}
    got = payout_prices.exit_prices(close, spans, reading=0.5, pinned=pinned)
    # Distinct losses are reading 0.5 plus pinned values 0.0 and 0.3: 3 calls total.
    assert len(seen_losses) == 3
    assert set(seen_losses) == {0.5, 0.0, 0.3}
    assert got["PERMNO-10"].iloc[4] == pytest.approx(104.0)
    assert got["PERMNO-20"].iloc[5] == pytest.approx(85.0)
    assert got["PERMNO-5~2"].iloc[5] == pytest.approx(42.0)  # 60.0 * (1 - 0.3)


def test_held_dead_symbols_strips_suffix_and_skips_successors() -> None:
    day = pd.Timestamp("2020-01-08")
    trades = [
        ft_replay.Trade(day, "ALIVE", 10.0, 20.0, ft_replay.REBALANCE_REASON),
        ft_replay.Trade(day, "PERMNO-5~1", -5.0, 30.0, ft_replay.DEAD_EXIT_REASON),
        ft_replay.Trade(day, "PERMNO-5~2", -5.0, 30.0, ft_replay.DEAD_EXIT_REASON),
        ft_replay.Trade(day, "PERMNO-5~3", -2.0, 30.0, ft_replay.DEAD_EXIT_REASON),
        ft_replay.Trade(day, "AAPL", -3.0, 90.0, ft_replay.DEAD_EXIT_REASON),
        ft_replay.Trade(day, "MSFT", -4.0, 50.0, ft_replay.LIQUIDATION_REASON),
        ft_replay.Trade(day, "IBM", -1.0, 40.0, ft_replay.TAX_SALE_REASON),
    ]
    successors = frozenset({"PERMNO-5~1"})
    # PERMNO-5~1 is a successor (skipped); ~2 and ~3 collapse to PERMNO-5; non-dead trades ignored.
    assert payout_prices.held_dead_symbols(trades, successors) == ["AAPL", "PERMNO-5"]


def test_coverage_of_lists_missing_as_subset_of_unpinned() -> None:
    held = ["PERMNO-1", "PERMNO-2", "PERMNO-3", "PERMNO-4"]
    pinned = {"PERMNO-1": 0.0, "PERMNO-99": 0.0}
    table_symbols = ["PERMNO-1", "PERMNO-2", "PERMNO-99"]
    cov = payout_prices.coverage_of(held, pinned, table_symbols)
    assert cov == {
        "held": 4,
        "pinned": 1,
        "unpinned": 3,
        "unpinned_names": ["PERMNO-2", "PERMNO-3", "PERMNO-4"],
        "missing_from_table": ["PERMNO-3", "PERMNO-4"],
        "pinned_names": ["PERMNO-1"],
    }
    assert set(cov["missing_from_table"]).issubset(set(cov["unpinned_names"]))
