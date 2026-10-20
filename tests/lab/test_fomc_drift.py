"""Tests for the fomc_drift strategy (candidate A7: pre-FOMC long-Treasury
drift -- long TLT/IEF/EDV the trading day before a scheduled FOMC
announcement, exit at/after the announcement)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.event_study import in_position_mask, split_event_returns
from ggTrader.lab.strategies.fomc_drift import (
    FOMC_TREASURY_UNIVERSE,
    FomcDriftStrategy,
    _event_signals,
)
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


def test_event_signal_session_semantics():
    idx = pd.bdate_range("2024-01-01", periods=10, tz="UTC") + pd.Timedelta("16:00:00")
    event = idx[5].normalize()
    signals = _event_signals(idx, ["TLT"], [event], entry_offset_days=1, hold_days=0)
    assert bool(signals.entries.loc[idx[4], "TLT"])
    assert bool(signals.exits.loc[idx[5], "TLT"])

    close = pd.DataFrame({"TLT": [100.0] * 10}, index=idx)
    returns = close.pct_change()
    mask = in_position_mask(signals.entries, signals.exits)
    event_returns, _ = split_event_returns(returns, mask)
    # in_position_mask covers exactly the day-5 return
    assert list(event_returns.index.get_level_values(0)) == [idx[5]]


def test_fomc_drift_rerun_excludes_warmup_and_includes_last_bar(monkeypatch):
    """Scientific correctness regression (Finding 1):
    run_fomc_drift_rerun loads warmup days for WFO but must evaluate the event study
    strictly within registered calendar bounds [2012-01-01, 2026-04-30] inclusive.
    An extreme warmup event in 2011 must be excluded, and the 2026-04-30 16:00 UTC
    session must be included.
    """
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import audit_reruns_wfo

    # Construct bars with 16:00:00 time of day (standard close time)
    # Warmup dates in 2011 (10 business days):
    warmup_dates = pd.bdate_range("2011-06-01", periods=10, tz="UTC") + pd.Timedelta("16:00:00")
    # 2012 eval dates (15 business days):
    eval_dates_2012 = pd.bdate_range("2012-01-02", periods=15, tz="UTC") + pd.Timedelta("16:00:00")
    # 2026 eval dates (15 business days ending on 2026-04-30):
    eval_dates_2026 = pd.bdate_range(end="2026-04-30", periods=15, tz="UTC") + pd.Timedelta(
        "16:00:00"
    )
    all_dates = warmup_dates.append(eval_dates_2012).append(eval_dates_2026)

    # Prices:
    # Warmup event on warmup_dates[5] (2011-06-08): +900% jump
    # Eval event 1 on eval_dates_2012[5] (2012-01-09): +1% jump
    # Eval event 2 on eval_dates_2026[-1] (2026-04-30): +2% jump
    tlt_close = pd.Series(100.0, index=all_dates)
    tlt_close.loc[warmup_dates[5]] = 1000.0
    tlt_close.loc[eval_dates_2012[5]] = 101.0
    tlt_close.loc[eval_dates_2026[-1]] = 102.0

    spy_close = pd.Series(100.0, index=all_dates)

    df = pd.DataFrame(
        {
            ("TLT", "close"): tlt_close,
            ("TLT", "open"): tlt_close,
            ("SPY", "close"): spy_close,
            ("SPY", "open"): spy_close,
        },
        index=all_dates,
    )

    events = [
        warmup_dates[5].normalize(),  # warmup event (2011)
        eval_dates_2012[5].normalize(),  # 2012 eval event
        eval_dates_2026[-1].normalize(),  # final session eval event (2026-04-30)
    ]

    mock_wfo = MagicMock()
    mock_wfo.oos_equity = pd.Series(
        [100.0, 101.0], index=pd.date_range("2012-01-01", periods=2, tz="UTC")
    )
    mock_wfo.fold_results = [
        {
            "fold_num": 1,
            "train_start": "2012-01-01",
            "train_end": "2013-01-01",
            "test_start": "2013-01-01",
            "test_end": "2013-04-01",
            "winner_combo": "c1",
            "winner_params": {},
            "ndh_passed": True,
            "dsr_passed": True,
            "gates_passed": True,
            "wfe": 0.60,
            "oos_sharpe": 1.0,
            "used_anchor": False,
        }
    ]
    mock_wfo.table = "stub_table"

    monkeypatch.setattr(
        audit_reruns_wfo, "equity_universe_between", lambda *args, **kwargs: ["TLT"]
    )
    monkeypatch.setattr(audit_reruns_wfo, "load_ohlcv", lambda *args, **kwargs: df)
    monkeypatch.setattr(audit_reruns_wfo, "run_wfo", lambda *args, **kwargs: mock_wfo)
    monkeypatch.setattr(audit_reruns_wfo.FomcDriftStrategy, "_load_events", lambda self: events)

    res = audit_reruns_wfo.run_fomc_drift_rerun()
    assert res["ok"] is True

    # Check primary pair entry_1_hold_0
    primary_gross = res["primary_event_study"]["gross"]
    # If warmup event (2011) were included, n_event would be 3 and mean_event_return would be > 1.0.
    # Since warmup is excluded and 2026-04-30 is included: exactly 2 events.
    assert primary_gross["n_event"] == 2
    # The mean event return should be the average of +1% and +2% (= 0.015), NOT affected by +900%
    assert primary_gross["mean_event_return"] == pytest.approx(0.015, abs=1e-4)
    assert res["primary_event_study"]["net"]["mean_event_return"] == pytest.approx(0.014)
    assert res["wfo"]["aggregate_wfe"] == pytest.approx(0.6)
    assert len(res["context_pairs"]) == 5

    # Undefined fold efficiency must not disappear from an otherwise passing mean.
    mock_wfo.fold_results.append({**mock_wfo.fold_results[0], "wfe": float("nan")})
    undefined = audit_reruns_wfo.run_fomc_drift_rerun()
    assert undefined["wfo"]["aggregate_wfe"] is None
    assert undefined["passes_bar"] is False


def test_main_checkpoint_force_preserves_unselected_strategy(monkeypatch, tmp_path):
    """Data loss regression (Finding 2):
    --force on a selected strategy must NOT discard already completed unselected results.
    """
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import audit_reruns_wfo

    checkpoint_file = tmp_path / "checkpoint.json"
    initial_data = {
        "xs_momentum": {
            "strategy": "xs_momentum",
            "ok": True,
            "verdict": "NO-GO stands",
            "val": 123,
        }
    }
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(initial_data, f)

    fomc_mock = {
        "strategy": "fomc_drift",
        "ok": True,
        "verdict": "NO-GO stands",
    }
    monkeypatch.setattr(audit_reruns_wfo, "run_fomc_drift_rerun", lambda: fomc_mock)

    ret = audit_reruns_wfo.main(
        [
            "--strategies",
            "fomc_drift",
            "--force",
            "--output",
            str(checkpoint_file),
        ]
    )
    assert ret == 0

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        saved = json.load(f)

    # xs_momentum must remain unchanged
    assert "xs_momentum" in saved
    assert saved["xs_momentum"]["ok"] is True
    assert saved["xs_momentum"]["val"] == 123
    assert "fomc_drift" in saved
    assert saved["fomc_drift"]["ok"] is True


def test_main_retries_failed_checkpoint_and_returns_nonzero(monkeypatch, tmp_path):
    """Resume and exit code regression (Finding 3):
    A checkpoint with ok=False must be automatically retried on next invocation (not skipped).
    If any requested strategy fails, main must return nonzero while continuing other strategies.
    """
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import audit_reruns_wfo

    checkpoint_file = tmp_path / "checkpoint.json"
    initial_data = {
        "xs_momentum": {
            "strategy": "xs_momentum",
            "ok": False,
            "verdict": "ERROR",
        }
    }
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(initial_data, f)

    calls = []

    def mock_momentum(strat):
        calls.append(strat)
        return {"strategy": strat, "ok": True, "verdict": "NO-GO stands"}

    monkeypatch.setattr(audit_reruns_wfo, "run_momentum_rerun", mock_momentum)

    # Invocation without --force should retry xs_momentum because it previously failed
    ret = audit_reruns_wfo.main(
        [
            "--strategies",
            "xs_momentum",
            "--output",
            str(checkpoint_file),
        ]
    )
    assert ret == 0
    assert calls == ["xs_momentum"]
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["xs_momentum"]["ok"] is True

    # Now verify that when a strategy fails, main() continues other strategies but returns nonzero
    calls.clear()

    def mock_momentum_with_failure(strat):
        if strat == "xs_momentum":
            raise RuntimeError("xs failed")
        return {"strategy": strat, "ok": True, "verdict": "NO-GO stands"}

    monkeypatch.setattr(audit_reruns_wfo, "run_momentum_rerun", mock_momentum_with_failure)
    ret_fail = audit_reruns_wfo.main(
        [
            "--strategies",
            "xs_momentum",
            "dual_momentum",
            "--force",
            "--output",
            str(checkpoint_file),
        ]
    )
    assert ret_fail != 0
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["xs_momentum"]["ok"] is False
    assert saved["dual_momentum"]["ok"] is True


def test_main_fails_clearly_without_overwriting_bad_checkpoint(monkeypatch, tmp_path):
    """Corrupt checkpoint regression (Finding 4):
    If the existing checkpoint is unreadable/corrupted JSON, fail clearly without overwriting.
    """
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import audit_reruns_wfo

    checkpoint_file = tmp_path / "bad_checkpoint.json"
    corrupt_content = "NOT_VALID_JSON{abc:123"
    checkpoint_file.write_text(corrupt_content, encoding="utf-8")

    rerun_mock = MagicMock()
    monkeypatch.setattr(audit_reruns_wfo, "run_momentum_rerun", rerun_mock)

    ret = audit_reruns_wfo.main(
        [
            "--strategies",
            "xs_momentum",
            "--output",
            str(checkpoint_file),
        ]
    )
    assert ret != 0
    # Must NOT have called strategy runner when checkpoint was corrupt
    rerun_mock.assert_not_called()

    # Ensure file content was not overwritten or destroyed
    assert checkpoint_file.read_text(encoding="utf-8") == corrupt_content


def test_atomic_write_json_strict_serialization_no_nan(tmp_path):
    """Raw artifact validity regression (Finding 5):
    json.dump must use allow_nan=False and serialize non-finite floats as null (no invented 0).
    """
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import audit_reruns_wfo

    out_file = tmp_path / "strict.json"
    data = {
        "float_nan": float("nan"),
        "float_inf": float("inf"),
        "float_neginf": float("-inf"),
        "np_nan": np.nan,
        "valid_float": 1.5,
        "valid_int": np.int64(42),
        "valid_bool": np.bool_(True),
        "aggregate_wfe": None,
    }

    audit_reruns_wfo._atomic_write_json(out_file, data)

    raw_text = out_file.read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text

    loaded = json.loads(raw_text)
    assert loaded["float_nan"] is None
    assert loaded["float_inf"] is None
    assert loaded["float_neginf"] is None
    assert loaded["np_nan"] is None
    assert loaded["valid_float"] == 1.5
    assert loaded["valid_int"] == 42
    assert loaded["valid_bool"] is True
    assert loaded["aggregate_wfe"] is None
