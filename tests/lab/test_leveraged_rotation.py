"""Tests for the leveraged/inverse index rotation strategy."""

from __future__ import annotations

import pandas as pd


def _idx(n, start="2020-01-31", freq="ME"):
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


class TestComputeBreadth:
    def test_fraction_of_true_columns(self):
        from ggTrader.lab.strategies.leveraged_rotation import compute_breadth

        entries = pd.DataFrame(
            {
                "A": [True, False, True],
                "B": [True, False, False],
                "C": [False, False, False],
                "D": [True, False, False],
            },
            index=_idx(3),
        )
        breadth = compute_breadth(entries)
        assert list(breadth) == [0.75, 0.0, 0.25]

    def test_empty_columns_returns_empty_series(self):
        from ggTrader.lab.strategies.leveraged_rotation import compute_breadth

        entries = pd.DataFrame(index=_idx(3))
        breadth = compute_breadth(entries)
        assert len(breadth) == 3
        assert breadth.isna().all() or (breadth == 0.0).all()


class TestRotatePositions:
    def test_first_state_takes_effect_immediately(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        breadth = pd.Series([0.7], index=_idx(1))
        states = rotate_positions(
            breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=3
        )
        assert states.iloc[0] == "long"

    def test_min_hold_one_flips_immediately(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        breadth = pd.Series([0.7, 0.3, 0.7], index=_idx(3))
        states = rotate_positions(
            breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=1
        )
        assert list(states) == ["long", "inverse", "long"]

    def test_min_hold_three_requires_confirmation(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        # Starts long, dips to inverse-territory for only 2 straight readings,
        # then bounces back to long-territory -- the flip to inverse should
        # never actually take effect (never held 3 consecutive readings).
        breadth = pd.Series([0.7, 0.3, 0.3, 0.7, 0.7], index=_idx(5))
        states = rotate_positions(
            breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=3
        )
        assert list(states) == ["long", "long", "long", "long", "long"]

    def test_min_hold_three_flips_after_three_consecutive(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        breadth = pd.Series([0.7, 0.3, 0.3, 0.3, 0.3], index=_idx(5))
        states = rotate_positions(
            breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=3
        )
        assert list(states) == ["long", "long", "long", "inverse", "inverse"]

    def test_between_thresholds_is_cash(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        breadth = pd.Series([0.5], index=_idx(1))
        states = rotate_positions(
            breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=1
        )
        assert states.iloc[0] == "cash"
