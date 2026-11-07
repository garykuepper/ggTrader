"""Synthetic bars and comparison helpers for the engine-contract strategy tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ggTrader.data.bars import Bars

BURN_IN = 200


def make_bars(
    seed: int = 11,
    n_bars: int = 700,
    n_syms: int = 10,
    *,
    holes: bool = False,
    symbols: list[str] | None = None,
) -> Bars:
    """Mean-reverting log-normal bars (UTC business days) with optional holes and late listing."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-03-01", periods=n_bars, tz="UTC")
    names = symbols or [f"S{i:02d}" for i in range(n_syms)]
    n = len(names)
    noise = rng.normal(0.0003, 0.02, (n_bars, n))
    wave = 0.012 * np.sin(np.arange(n_bars)[:, None] / (6.0 + np.arange(n)[None, :]))
    close = 100.0 * np.exp(np.cumsum(noise + np.diff(wave, axis=0, prepend=0.0), axis=0))
    spread = rng.uniform(0.0, 0.01, close.shape)
    high, low = close * (1 + spread), close * (1 - spread)
    open_ = close * (1 + rng.normal(0, 0.004, close.shape))
    volume = np.round(rng.lognormal(12.0, 0.5, close.shape))
    if holes:
        close[100:104, 1] = np.nan  # interior hole
        close[:150, 2] = np.nan  # late listing
        close[500:, 3] = np.nan  # delisting
        mask = np.isnan(close)
        high, low, open_, volume = (np.where(mask, np.nan, a) for a in (high, low, open_, volume))
    frames = {
        name: pd.DataFrame(arr, index=index, columns=names)
        for name, arr in (
            ("close", close),
            ("open", open_),
            ("high", high),
            ("low", low),
            ("volume", volume),
        )
    }
    last = frames["close"].apply(lambda s: s.last_valid_index())
    return Bars(
        requested=tuple(names),
        missing_symbols=(),
        last_valid_bar=last,
        provenance={"source": "synthetic"},
        **frames,
    )


def settled(close: pd.DataFrame, burn_in: int = BURN_IN) -> np.ndarray:
    """True on bars at least ``burn_in`` bars after each symbol's own first valid close."""
    first = close.notna().to_numpy().argmax(axis=0)
    return np.arange(len(close))[:, None] >= (first + burn_in)[None, :]


def diff_after_burn_in(new: np.ndarray, old: np.ndarray, close: pd.DataFrame) -> int:
    """Number of differing boolean cells after each symbol's burn-in."""
    return int(((np.asarray(new, bool) != np.asarray(old, bool)) & settled(close)).sum())


# --- frozen legacy outputs (captured before the legacy classes were deleted, P5c) -------------

GOLDEN_DIR = Path(__file__).parent / "golden"
_LEGACY_SIGNALS = None
_LEGACY_GRIDS = None


def legacy_signals():
    """The frozen ``legacy_signals.npz`` (see ``p5c-evidence/capture_legacy_signals.py``)."""
    global _LEGACY_SIGNALS
    if _LEGACY_SIGNALS is None:
        _LEGACY_SIGNALS = np.load(GOLDEN_DIR / "legacy_signals.npz")
    return _LEGACY_SIGNALS


def frozen_cells(prefix: str, name: str) -> np.ndarray:
    """A frozen legacy entry/exit frame as a boolean (bars x symbols) array."""
    data = legacy_signals()
    shape = tuple(int(x) for x in data[f"{prefix}|{name}|shape"])
    flat = np.unpackbits(data[f"{prefix}|{name}"])[: int(np.prod(shape))]
    return flat.reshape(shape).astype(bool)


def frozen_sizes(prefix: str) -> np.ndarray:
    """Frozen legacy per-bar conviction sizes of one ``strategy|combo`` key."""
    return legacy_signals()[f"{prefix}|sizes"]


def legacy_grids() -> dict:
    """The frozen legacy ``build_grid`` combinations and constructor attributes per strategy."""
    global _LEGACY_GRIDS
    if _LEGACY_GRIDS is None:
        _LEGACY_GRIDS = json.loads((GOLDEN_DIR / "legacy_grids.json").read_text())
    return _LEGACY_GRIDS


def frozen_frame(key: str) -> pd.DataFrame:
    """A frozen legacy weight-target frame (``legacy_weights.npz``), UTC microsecond index."""
    data = np.load(GOLDEN_DIR / "legacy_weights.npz")
    index = pd.DatetimeIndex(pd.to_datetime(data[f"{key}|index"], unit="us", utc=True))
    return pd.DataFrame(
        data[f"{key}|values"], index=index, columns=[str(c) for c in data[f"{key}|columns"]]
    )


def frozen_array(key: str) -> np.ndarray:
    """A frozen legacy array of ``legacy_weights.npz`` (for example an equity matrix)."""
    return np.load(GOLDEN_DIR / "legacy_weights.npz")[key]
