"""Tests for the fomc_drift strategy (candidate A7: pre-FOMC long-Treasury
drift -- long TLT/IEF/EDV the trading day before a scheduled FOMC
announcement, exit at/after the announcement)."""

from __future__ import annotations

import pandas as pd

from ggTrader.lab.strategies.fomc_drift import FOMC_TREASURY_UNIVERSE, FomcDriftStrategy
from ggTrader.lab.strategy import LabConfig, SignalTargets


def _make_ohlcv(
    tickers: list[str], n: int = 60, start: str = "2020-01-01", bar_time: str | None = None
) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, tz="UTC")
    if bar_time is not None:
        # Real OHLCV bars in this lab carry a non-midnight time-of-day
        # (e.g. "16:00:00+00:00", the yfinance loader's close-time
        # convention) -- tests must cover that shape, not just midnight
        # bars, or a date-vs-timestamp mismatch bug goes unnoticed.
        idx = idx + pd.Timedelta(bar_time)
    frames = {}
    for t in tickers:
        base = pd.Series(100.0, index=idx)
        frames[(t, "close")] = base
        frames[(t, "open")] = base
    df = pd.DataFrame(frames)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestFomcDriftToTargets:
    def test_enters_one_bar_before_and_exits_on_the_event_bar(self):
        data = _make_ohlcv(FOMC_TREASURY_UNIVERSE, n=30, start="2020-01-01")
        idx = data.index
        event_date = idx[10]  # a real bar in the index

        strat = FomcDriftStrategy(
            LabConfig(min_history_bars=1),
            hold_days=0,
            entry_offset_days=1,
            _events_loader=lambda: [event_date],
        )
        plans = {idx[0]: [{"symbol": s, "weight": 0.0} for s in FOMC_TREASURY_UNIVERSE]}
        targets = strat.to_targets(plans, data)

        assert isinstance(targets, SignalTargets)
        entry_bar = idx[10 - 1]
        exit_bar = idx[10]
        assert bool(targets.entries.loc[entry_bar].all())
        assert bool(targets.exits.loc[exit_bar].all())
        # No other bar should show an entry or exit.
        assert targets.entries.drop(index=entry_bar).sum().sum() == 0
        assert targets.exits.drop(index=exit_bar).sum().sum() == 0

    def test_matches_events_against_bars_with_a_non_midnight_time_of_day(self):
        """Regression: real OHLCV bars carry a '16:00:00+00:00' time-of-day
        (not midnight), but FOMC event dates from fomc_calendar.py are
        plain midnight-normalized Timestamps. An exact-timestamp match
        between the two silently matched nothing -- must compare on date
        alone."""
        data = _make_ohlcv(FOMC_TREASURY_UNIVERSE, n=30, start="2020-01-01", bar_time="16:00:00")
        idx = data.index
        event_date = idx[10].normalize()  # midnight, like real FOMC calendar data

        strat = FomcDriftStrategy(
            LabConfig(min_history_bars=1),
            hold_days=0,
            entry_offset_days=1,
            _events_loader=lambda: [event_date],
        )
        plans = {idx[0]: [{"symbol": s, "weight": 0.0} for s in FOMC_TREASURY_UNIVERSE]}
        targets = strat.to_targets(plans, data)

        assert bool(targets.entries.loc[idx[10 - 1]].all())
        assert bool(targets.exits.loc[idx[10]].all())

    def test_hold_days_shifts_the_exit_bar_forward(self):
        data = _make_ohlcv(FOMC_TREASURY_UNIVERSE, n=30, start="2020-01-01")
        idx = data.index
        event_date = idx[10]

        strat = FomcDriftStrategy(
            LabConfig(min_history_bars=1),
            hold_days=2,
            entry_offset_days=1,
            _events_loader=lambda: [event_date],
        )
        plans = {idx[0]: [{"symbol": s, "weight": 0.0} for s in FOMC_TREASURY_UNIVERSE]}
        targets = strat.to_targets(plans, data)
        exit_bar = idx[10 + 2]
        assert bool(targets.exits.loc[exit_bar].all())

    def test_event_near_the_start_of_the_window_is_skipped_not_crashed(self):
        data = _make_ohlcv(FOMC_TREASURY_UNIVERSE, n=30, start="2020-01-01")
        idx = data.index
        strat = FomcDriftStrategy(
            LabConfig(min_history_bars=1),
            entry_offset_days=1,
            _events_loader=lambda: [idx[0]],  # no room for entry_offset_days before this
        )
        plans = {idx[0]: [{"symbol": s, "weight": 0.0} for s in FOMC_TREASURY_UNIVERSE]}
        targets = strat.to_targets(plans, data)
        assert targets.entries.sum().sum() == 0

    def test_event_near_the_end_of_the_window_is_skipped_not_crashed(self):
        data = _make_ohlcv(FOMC_TREASURY_UNIVERSE, n=30, start="2020-01-01")
        idx = data.index
        strat = FomcDriftStrategy(
            LabConfig(min_history_bars=1),
            hold_days=5,
            _events_loader=lambda: [idx[-1]],  # no room for hold_days after this
        )
        plans = {idx[0]: [{"symbol": s, "weight": 0.0} for s in FOMC_TREASURY_UNIVERSE]}
        targets = strat.to_targets(plans, data)
        assert targets.exits.sum().sum() == 0

    def test_sweep_params_present(self):
        params = FomcDriftStrategy.sweep_params()
        assert "hold_days" in params
        assert isinstance(params["hold_days"], list) and len(params["hold_days"]) >= 2


class TestFomcDriftSweepSignals:
    def test_produces_one_signaltargets_per_combo_matching_its_own_params(self):
        from ggTrader.lab.sweep import combo_name

        data = _make_ohlcv(FOMC_TREASURY_UNIVERSE, n=30, start="2020-01-01")
        idx = data.index
        event_date = idx[10]
        strat = FomcDriftStrategy(
            LabConfig(min_history_bars=1), _events_loader=lambda: [event_date]
        )
        combos = [
            {"entry_offset_days": 1, "hold_days": 0},
            {"entry_offset_days": 2, "hold_days": 1},
        ]
        result = strat.sweep_signals(combos, FOMC_TREASURY_UNIVERSE, data)

        name0 = combo_name("fomc_drift", combos[0])
        name1 = combo_name("fomc_drift", combos[1])
        assert set(result) == {name0, name1}

        assert bool(result[name0].entries.loc[idx[10 - 1]].all())
        assert bool(result[name0].exits.loc[idx[10]].all())

        assert bool(result[name1].entries.loc[idx[10 - 2]].all())
        assert bool(result[name1].exits.loc[idx[10 + 1]].all())


class TestFomcDriftSelect:
    def test_returns_a_plan_covering_the_fixed_universe(self):
        data = _make_ohlcv(FOMC_TREASURY_UNIVERSE, n=30, start="2020-01-01")
        strat = FomcDriftStrategy(LabConfig(min_history_bars=1), _events_loader=lambda: [])
        plan = strat.select(data.index[-1], data, eligible=[])
        assert {s["symbol"] for s in plan} == set(FOMC_TREASURY_UNIVERSE)
