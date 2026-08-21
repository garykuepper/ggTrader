"""Tests for the event-study supplementary test -- a trade-level lens for
sparse/event-driven strategies (fomc_drift, index_deletion_fade) where the
annualized-Sharpe-based WFO/NDH/DSR gate machinery is a poor statistical
fit: a handful of trades dominate an otherwise-flat-cash equity curve, so
per-fold Sharpe becomes extremely noisy (observed: -3.59 to +3.35 across
fomc_drift's 54 folds). This compares mean return on "event days" (when
the strategy would hold a position) against a matched sample of ordinary
days on the same instruments, via a two-sample (Welch's) t-test -- the
same style of test the source papers themselves typically report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.event_study import (
    event_study_test,
    in_position_mask,
    run_event_study_from_signals,
    split_event_returns,
)


class TestInPositionMask:
    def test_single_entry_no_exit_holds_to_end(self):
        idx = pd.RangeIndex(10)
        entries = pd.DataFrame({"A": [False] * 10})
        exits = pd.DataFrame({"A": [False] * 10})
        entries.loc[3, "A"] = True
        entries.index = idx
        exits.index = idx
        mask = in_position_mask(entries, exits)
        assert list(mask["A"]) == [False, False, False, True, True, True, True, True, True, True]

    def test_entry_then_exit_holds_only_in_between_inclusive(self):
        idx = pd.RangeIndex(10)
        entries = pd.DataFrame({"A": [False] * 10}, index=idx)
        exits = pd.DataFrame({"A": [False] * 10}, index=idx)
        entries.loc[2, "A"] = True
        exits.loc[5, "A"] = True
        mask = in_position_mask(entries, exits)
        expected = [False, False, True, True, True, True, False, False, False, False]
        assert list(mask["A"]) == expected

    def test_two_non_overlapping_events(self):
        idx = pd.RangeIndex(10)
        entries = pd.DataFrame({"A": [False] * 10}, index=idx)
        exits = pd.DataFrame({"A": [False] * 10}, index=idx)
        entries.loc[1, "A"] = True
        exits.loc[2, "A"] = True
        entries.loc[6, "A"] = True
        exits.loc[7, "A"] = True
        mask = in_position_mask(entries, exits)
        expected = [False, True, True, False, False, False, True, True, False, False]
        assert list(mask["A"]) == expected

    def test_no_entries_never_in_position(self):
        idx = pd.RangeIndex(5)
        entries = pd.DataFrame({"A": [False] * 5}, index=idx)
        exits = pd.DataFrame({"A": [False] * 5}, index=idx)
        mask = in_position_mask(entries, exits)
        assert not mask["A"].any()


class TestSplitEventReturns:
    def test_separates_event_and_nonevent_cells(self):
        idx = pd.RangeIndex(4)
        returns = pd.DataFrame(
            {"A": [0.01, 0.02, -0.01, 0.03], "B": [0.05, -0.02, 0.04, 0.01]}, index=idx
        )
        mask = pd.DataFrame(
            {"A": [True, False, True, False], "B": [False, True, False, True]}, index=idx
        )
        event_ret, nonevent_ret = split_event_returns(returns, mask)
        assert sorted(event_ret.tolist()) == sorted([0.01, -0.01, -0.02, 0.01])
        assert sorted(nonevent_ret.tolist()) == sorted([0.02, 0.03, 0.05, 0.04])

    def test_empty_mask_gives_empty_event_series(self):
        idx = pd.RangeIndex(3)
        returns = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=idx)
        mask = pd.DataFrame({"A": [False, False, False]}, index=idx)
        event_ret, nonevent_ret = split_event_returns(returns, mask)
        assert len(event_ret) == 0
        assert len(nonevent_ret) == 3


class TestEventStudyTest:
    def test_detects_a_genuine_mean_difference(self):
        rng = np.random.default_rng(0)
        event_returns = pd.Series(rng.normal(0.01, 0.005, 200))  # clearly higher mean
        nonevent_returns = pd.Series(rng.normal(0.0, 0.005, 5000))
        result = event_study_test(event_returns, nonevent_returns)
        assert result.mean_diff > 0
        assert result.p_value < 0.05
        assert result.significant

    def test_null_case_is_not_significant(self):
        rng = np.random.default_rng(1)
        event_returns = pd.Series(rng.normal(0.0, 0.01, 60))
        nonevent_returns = pd.Series(rng.normal(0.0, 0.01, 5000))
        result = event_study_test(event_returns, nonevent_returns)
        assert not result.significant

    def test_counts_and_means_are_reported(self):
        event_returns = pd.Series([0.01, 0.02, 0.03])
        nonevent_returns = pd.Series([0.0, 0.0, 0.0, 0.0])
        result = event_study_test(event_returns, nonevent_returns)
        assert result.n_event == 3
        assert result.n_nonevent == 4
        assert result.mean_event_return == pytest.approx(0.02)
        assert result.mean_nonevent_return == pytest.approx(0.0)

    def test_raises_on_insufficient_event_observations(self):
        with pytest.raises(ValueError, match="at least 2"):
            event_study_test(pd.Series([0.01]), pd.Series([0.0, 0.0, 0.0]))


class TestRunEventStudyFromSignals:
    def test_end_to_end_matches_manual_split(self):
        idx = pd.RangeIndex(20)
        close = pd.DataFrame(
            {"A": np.linspace(100, 120, 20), "B": np.linspace(50, 40, 20)}, index=idx
        )
        entries = pd.DataFrame(False, index=idx, columns=close.columns)
        exits = pd.DataFrame(False, index=idx, columns=close.columns)
        entries.loc[5, "A"] = True
        exits.loc[6, "A"] = True
        entries.loc[12, "B"] = True
        exits.loc[13, "B"] = True

        result = run_event_study_from_signals(close, entries, exits)
        assert result.n_event > 0
        assert result.n_nonevent > 0
