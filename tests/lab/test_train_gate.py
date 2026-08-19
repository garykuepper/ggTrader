"""Equivalence test for train_gate._build_dataset (Phase C item 16).

train_gate.py is FALSIFIED/DISABLED (see its module docstring) but kept
correctness-preserving. The nested per-date-per-symbol `.loc` scalar lookup
over `entries` was replaced with `np.nonzero` over `entries.to_numpy()`; this
test pins output-identical behavior against a reference implementation that
mirrors the original nested-loop code exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ggTrader.lab.train_gate import _build_dataset
from ggTrader.paper.feature_gate import extract_features


def _reference_build_dataset(ohlcv: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the pre-vectorization nested for-date/for-symbol .loc loop."""
    FORWARD_DAYS = 5
    records: list[dict] = []
    sym_cols = list(entries.columns)
    available_syms = set(ohlcv.columns.get_level_values(0))
    close_frames = {s: ohlcv[s]["close"].dropna() for s in sym_cols if s in available_syms}
    volume_frames: dict = {}
    high_frames: dict = {}
    low_frames: dict = {}
    for s in sym_cols:
        if s not in available_syms:
            continue
        v = ohlcv[s].get("volume")
        volume_frames[s] = (
            v.dropna()
            if v is not None and not v.empty
            else pd.Series(1.0, index=close_frames[s].index)
        )
        h = ohlcv[s].get("high")
        high_frames[s] = h.dropna() if h is not None and not h.empty else None
        lo = ohlcv[s].get("low")
        low_frames[s] = lo.dropna() if lo is not None and not lo.empty else None

    for bar_date in entries.index:
        for symbol in sym_cols:
            if not entries.loc[bar_date, symbol]:
                continue
            if symbol not in close_frames:
                continue

            close = close_frames[symbol]
            volume = volume_frames[symbol]

            if bar_date not in close.index:
                continue
            bar_idx = close.index.get_loc(bar_date)
            if bar_idx < 20:
                continue

            future_idx = bar_idx + FORWARD_DAYS
            if future_idx >= len(close):
                continue
            fwd_ret = close.iloc[future_idx] / close.iloc[bar_idx] - 1.0
            label = 1 if fwd_ret > 0 else 0

            high = high_frames.get(symbol)
            low = low_frames.get(symbol)
            feats = extract_features(
                close.iloc[: bar_idx + 1],
                volume.iloc[: bar_idx + 1],
                bar_date,
                high=high.iloc[: bar_idx + 1] if high is not None else None,
                low=low.iloc[: bar_idx + 1] if low is not None else None,
            )
            feats["label"] = label
            feats["date"] = bar_date
            feats["symbol"] = symbol
            records.append(feats)

    return pd.DataFrame(records)


def _synthetic_ohlcv_and_entries(n_days=120, symbols=("AAA", "BBB", "CCC"), seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days, tz="UTC")
    frames = {}
    for s in symbols:
        close = 100.0 * (1.0 + pd.Series(rng.normal(0.0005, 0.01, n_days), index=idx)).cumprod()
        frames[s] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, n_days).astype(float),
            },
            index=idx,
        )
    ohlcv = pd.concat(frames, axis=1)
    ohlcv.columns = ohlcv.columns.set_names(["symbol", "field"])

    entries = pd.DataFrame(False, index=idx, columns=list(symbols))
    # Sparse True entries, deliberately including bars too early for warmup
    # (bar_idx < 20) and bars too close to the end (no forward label).
    true_positions = rng.choice(n_days, size=25, replace=False)
    for pos in true_positions:
        sym = symbols[pos % len(symbols)]
        entries.iloc[pos, entries.columns.get_loc(sym)] = True
    return ohlcv, entries


class TestBuildDatasetEquivalence:
    def test_matches_reference_nested_loop_output(self):
        ohlcv, entries = _synthetic_ohlcv_and_entries()
        vectorized = _build_dataset(ohlcv, entries)
        reference = _reference_build_dataset(ohlcv, entries)

        assert len(vectorized) == len(reference)
        assert len(vectorized) > 0, "test fixture produced zero rows -- widen it"
        pd.testing.assert_frame_equal(
            vectorized.reset_index(drop=True), reference.reset_index(drop=True)
        )

    def test_empty_entries_produces_empty_dataset(self):
        ohlcv, entries = _synthetic_ohlcv_and_entries()
        entries.loc[:, :] = False
        out = _build_dataset(ohlcv, entries)
        assert len(out) == 0

    def test_row_order_is_date_major(self):
        """np.nonzero on a row-major array must preserve the original
        date-major, symbol-minor iteration order (downstream code assumes
        it, e.g. sort_values('date') is a stable no-op for ties)."""
        ohlcv, entries = _synthetic_ohlcv_and_entries()
        out = _build_dataset(ohlcv, entries)
        dates = out["date"].tolist()
        # Non-decreasing: every row for an earlier bar_date must precede
        # every row for a later bar_date (matches nested date-outer loop).
        assert dates == sorted(dates)
