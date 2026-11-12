"""Tests for the fail-closed gates: input bars, signal counts and the late-run cutoff."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from ggTrader.data.bars import Bars
from ggTrader.paper.gates import (
    PT,
    SanityConfig,
    SignalGateError,
    late_run_reason,
    vet_bars,
    vet_signal_counts,
)
from ggTrader.paper.live_bars import bars_from_legacy_frame

TODAY = date(2026, 6, 19)
SMALL = SanityConfig(min_bars=50, min_universe=20, min_loaded_share=0.9, min_last_bar_coverage=0.95)


def _frame(
    n_sym: int = 40, n_days: int = 80, end: str = "2026-06-18", nan_last: int = 0
) -> pd.DataFrame:
    """A live-loader style `(symbol, field)` frame of random walks."""
    dates = pd.bdate_range(end=end, periods=n_days, tz="UTC")
    rng = np.random.default_rng(1)
    cols = {}
    for i in range(n_sym):
        price = 100 * np.exp(rng.normal(0, 0.01, n_days).cumsum())
        sub = pd.DataFrame(
            {"open": price, "high": price, "low": price, "close": price, "volume": 1e6},
            index=dates,
        )
        if i < nan_last:
            sub.iloc[-1] = np.nan
        cols[f"S{i:02d}"] = sub
    df = pd.concat(cols, axis=1)
    df.columns.names = ["symbol", "field"]
    return df


def _bars(frame: pd.DataFrame) -> Bars:
    return bars_from_legacy_frame(frame, sorted(set(frame.columns.get_level_values(0))))


class TestVetBars:
    def test_healthy_bars_pass_untouched(self):
        verdict = vet_bars(_bars(_frame()), TODAY, SMALL)
        assert verdict.trimmed_partial_bar is False
        assert verdict.last_bar_coverage == 1.0
        assert len(verdict.bars.close) == 80

    def test_full_coverage_today_remains_unchanged(self):
        verdict = vet_bars(_bars(_frame(end="2026-06-19")), TODAY, SMALL)
        assert verdict.bars.close.index[-1].date() == TODAY
        assert not verdict.trimmed_partial_bar

    @pytest.mark.parametrize("nan_last", [0, 39])
    def test_future_bar_is_rejected_even_when_coverage_is_thin(self, nan_last):
        bars = _bars(_frame(end="2026-06-22", nan_last=nan_last))
        with pytest.raises(SignalGateError, match="future"):
            vet_bars(bars, TODAY, SMALL)

    def test_empty_frame_fails(self):
        with pytest.raises(SignalGateError, match="empty_frame"):
            vet_bars(bars_from_legacy_frame(pd.DataFrame(), ["A"]), TODAY, SMALL)

    def test_duplicate_days_fail(self):
        frame = _frame()
        frame = pd.concat([frame, frame.iloc[[-1]]])
        with pytest.raises(SignalGateError, match="duplicate_days"):
            vet_bars(_bars(frame), TODAY, SMALL)

    def test_weekend_bar_fails(self):
        frame = _frame(end="2026-06-18")
        sunday = frame.iloc[[-1]].copy()
        sunday.index = pd.DatetimeIndex([pd.Timestamp("2026-06-21", tz="UTC")])
        with pytest.raises(SignalGateError, match="weekend_bars"):
            vet_bars(_bars(pd.concat([frame, sunday])), TODAY, SMALL)

    def test_small_universe_fails(self):
        with pytest.raises(SignalGateError, match="small_universe"):
            vet_bars(_bars(_frame(n_sym=10)), TODAY, SMALL)

    def test_loaded_share_below_floor_fails(self):
        bars = _bars(_frame(n_sym=40))
        asked = Bars(**{**bars.__dict__, "requested": tuple(f"S{i:02d}" for i in range(60))})
        with pytest.raises(SignalGateError, match="small_universe"):
            vet_bars(asked, TODAY, SMALL)

    def test_short_history_fails(self):
        with pytest.raises(SignalGateError, match="short_history"):
            vet_bars(_bars(_frame(n_days=30)), TODAY, SMALL)

    def test_thin_final_bar_of_an_older_day_is_a_bad_tape(self):
        """Coverage 20/40 on a bar that is NOT today: a vendor/tape problem, so stop."""
        frame = _frame(end="2026-06-17", nan_last=20)
        with pytest.raises(SignalGateError, match="last_bar_coverage"):
            vet_bars(_bars(frame), TODAY, SMALL)

    def test_thin_final_bar_of_today_is_trimmed_not_fatal(self):
        """The 1-of-646 case: today's bar has not been published for most symbols yet. The
        previous bar is complete, so signals use it and the run records the trim."""
        frame = _frame(end="2026-06-19", nan_last=39)
        verdict = vet_bars(_bars(frame), date(2026, 6, 19), SMALL)
        assert verdict.trimmed_partial_bar is True
        assert verdict.dropped_bar == "2026-06-19"
        assert verdict.bars.close.index[-1].date() == date(2026, 6, 18)
        assert verdict.last_bar_coverage == 1.0
        assert len(verdict.bars.close) == 79

    def test_two_thin_bars_in_a_row_are_a_bad_tape(self):
        frame = _frame(end="2026-06-19", nan_last=39)
        frame.iloc[-2, : 39 * 5] = np.nan  # yesterday is thin too
        with pytest.raises(SignalGateError, match="last_bar_coverage"):
            vet_bars(_bars(frame), date(2026, 6, 19), SMALL)

    def test_a_few_missing_closes_below_the_tolerance_pass(self):
        """1 of 40 missing is 97.5%, above the 95% floor: no trim, no stop."""
        verdict = vet_bars(_bars(_frame(nan_last=1)), TODAY, SMALL)
        assert verdict.trimmed_partial_bar is False
        assert verdict.last_bar_coverage == pytest.approx(0.975)

    def test_production_bounds_reject_a_small_panel(self):
        with pytest.raises(SignalGateError, match="small_universe"):
            vet_bars(_bars(_frame()), TODAY, SanityConfig())


class TestSignalCounts:
    def test_normal_counts_pass(self):
        vet_signal_counts(37, 127)  # the golden maxima

    @pytest.mark.parametrize("buys,sells", [(101, 0), (0, 251), (503, 503)])
    def test_absurd_counts_fail(self, buys, sells):
        with pytest.raises(SignalGateError, match="absurd_signal_count"):
            vet_signal_counts(buys, sells)


class TestLateRun:
    @staticmethod
    def _pt(h: int, m: int) -> datetime:
        return datetime(2026, 6, 19, h, m, tzinfo=PT)

    def test_on_time_at_the_cron_minute(self):
        assert late_run_reason(self._pt(12, 46)) is None

    def test_exactly_at_the_cutoff_is_still_on_time(self):
        assert late_run_reason(self._pt(12, 55)) is None

    def test_after_the_cutoff_is_late(self):
        reason = late_run_reason(self._pt(12, 56))
        assert reason is not None and "12:55" in reason

    def test_evening_run_is_late(self):
        assert late_run_reason(self._pt(21, 30)) is not None

    def test_timezone_is_converted(self):
        utc = datetime(2026, 6, 19, 20, 0, tzinfo=ZoneInfo("UTC"))  # 13:00 PDT
        assert late_run_reason(utc) is not None
        on_time = datetime(2026, 6, 19, 19, 46, tzinfo=ZoneInfo("UTC"))  # 12:46 PDT
        assert late_run_reason(on_time) is None

    def test_half_day_close_pulls_the_cutoff_earlier(self):
        close = "2026-06-19 10:00:00-07:00"
        # cutoff = close - 5 min = 09:55 PT
        assert late_run_reason(self._pt(9, 56), close) is not None
        assert late_run_reason(self._pt(9, 54), close) is None

    def test_unparsable_next_close_is_ignored(self):
        assert late_run_reason(self._pt(12, 46), "not a time") is None
