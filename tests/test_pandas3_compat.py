"""Pin the pandas-2 behaviours that the pandas-3 / vectorbt 1.2 stack would silently change.

Every expected value here is a literal or an independent numpy/pandas reference that gives the
same answer on pandas 2.3.3 and pandas 3.x, so the module passes on both stacks and fails if a
call site regresses to a bare `pct_change()` or a bare `stack()`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.event_study import run_event_study_from_signals, split_event_returns
from ggTrader.paper.overlay import compute_weights_and_scale
from ggTrader.utils.returns import padded_pct_change

REPO = Path(__file__).resolve().parents[1]
NAN = np.nan


def _idx(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _ohlcv(close: pd.DataFrame) -> pd.DataFrame:
    """Multi-level (symbol, field) OHLCV frame; NaN closes stay NaN in every field."""
    frames = {
        s: pd.DataFrame(
            {
                "open": close[s],
                "high": close[s] * 1.001,
                "low": close[s] * 0.999,
                "close": close[s],
                "volume": np.where(close[s].isna(), NAN, 1e6),
            }
        )
        for s in close.columns
    }
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


class TestPaddedPctChange:
    """pandas 2 forward-filled non-leading NaN before differencing; pandas 3 does not."""

    def test_series_interior_and_trailing_gaps_give_zero_returns(self) -> None:
        s = pd.Series([NAN, 100.0, 110.0, NAN, 121.0, NAN, NAN])
        got = padded_pct_change(s).to_numpy()
        expected = np.array([NAN, NAN, 110.0 / 100.0 - 1.0, 0.0, 121.0 / 110.0 - 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(got, expected)

    def test_frame_columns_are_independent(self) -> None:
        df = pd.DataFrame({"A": [1.0, 2.0, NAN, 3.0], "B": [NAN, NAN, 5.0, 10.0]})
        got = padded_pct_change(df)
        np.testing.assert_array_equal(got["A"].to_numpy(), [NAN, 1.0, 0.0, 0.5])
        np.testing.assert_array_equal(got["B"].to_numpy(), [NAN, NAN, NAN, 1.0])

    def test_gap_free_input_equals_plain_pct_change(self) -> None:
        s = pd.Series(np.linspace(10.0, 20.0, 30))
        np.testing.assert_array_equal(
            padded_pct_change(s).to_numpy(), s.pct_change(fill_method=None).to_numpy()
        )

    def test_does_not_mutate_input(self) -> None:
        s = pd.Series([1.0, NAN, 2.0])
        padded_pct_change(s)
        assert s.isna().tolist() == [False, True, False]


class TestNoBarePctChangeCalls:
    """Static guard: a bare `.pct_change()` means different numbers on pandas 2 and 3."""

    @staticmethod
    def _bare_calls(root: Path) -> list[str]:
        bad: list[str] = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "pct_change"
                    and not any(k.arg == "fill_method" for k in node.keywords)
                ):
                    bad.append(f"{path.relative_to(REPO)}:{node.lineno}")
        return bad

    def test_src_has_no_bare_pct_change(self) -> None:
        assert self._bare_calls(REPO / "src") == []

    def test_scripts_have_no_bare_pct_change(self) -> None:
        assert self._bare_calls(REPO / "scripts") == []


class TestEventStudyStack:
    """pandas 3 `stack()` keeps NaN rows; split_event_returns must still drop them."""

    def test_nan_cells_are_not_counted_as_event_or_nonevent_returns(self) -> None:
        idx = pd.RangeIndex(4)
        returns = pd.DataFrame(
            {"A": [0.01, NAN, 0.03, 0.04], "B": [NAN, 0.02, NAN, 0.05]}, index=idx
        )
        mask = pd.DataFrame({"A": [True, True, False, False], "B": [False, True, False, True]})
        event, nonevent = split_event_returns(returns, mask)
        assert sorted(event.tolist()) == [0.01, 0.02, 0.05]
        assert sorted(nonevent.tolist()) == [0.03, 0.04]

    def test_event_study_uses_padded_returns_for_price_gaps(self) -> None:
        # A one-day price gap in the non-event region: the old default padded it to a 0.0 return
        # that stays in the non-event sample; bare pandas 3 turns it (and the next return) to NaN.
        n = 20
        px = 100.0 + np.arange(n, dtype=float)
        px[3] = NAN
        close = pd.DataFrame({"A": px})
        entries = pd.DataFrame({"A": [False] * n})
        exits = pd.DataFrame({"A": [False] * n})
        entries.loc[8, "A"] = True
        exits.loc[14, "A"] = True
        result = run_event_study_from_signals(close, entries, exits)
        # event bars 9..14 (6); non-event returns on bars 1..8 and 15..19 (8 + 5), gap included.
        assert result.n_event == 6
        assert result.n_nonevent == 13


class TestLivePathOverlay:
    """paper/overlay.py:88 -- gap-free sleeve curves keep their old weights and scale."""

    @staticmethod
    def _curves() -> dict[str, pd.Series]:
        rng = np.random.default_rng(5)
        idx = _idx(120)
        return {
            "sp500": pd.Series(10000 * np.exp(rng.normal(0.0004, 0.010, 120).cumsum()), index=idx),
            "nasdaq100": pd.Series(
                10000 * np.exp(rng.normal(0.0004, 0.020, 120).cumsum()), index=idx
            ),
        }

    def test_weights_and_scale_match_independent_reference(self) -> None:
        curves = self._curves()
        weights, scale = compute_weights_and_scale(curves, target_vol=0.068, window=60)

        rets = {k: v.to_numpy()[1:] / v.to_numpy()[:-1] - 1.0 for k, v in curves.items()}
        vol = {k: np.std(r[-60:], ddof=1) * np.sqrt(252) for k, r in rets.items()}
        inv = {k: 1.0 / x for k, x in vol.items()}
        want_w = {k: x / sum(inv.values()) for k, x in inv.items()}
        for k in curves:
            assert weights[k] == pytest.approx(want_w[k], rel=1e-12)
        blended = sum(rets[k] * want_w[k] for k in curves)
        want_scale = float(np.clip(0.068 / (np.std(blended[-60:], ddof=1) * np.sqrt(252)), 0, 1.0))
        assert scale == pytest.approx(want_scale, rel=1e-12)
        assert np.isfinite(scale)
