"""Voter arrays for the signal strategies, built only from engine presets and crossovers.

Each voter is computed once per distinct parameter tuple over the whole ``dates x symbols``
panel and returned as boolean ``(entry, exit)`` arrays for :mod:`ggTrader.lab.strategies.vote`.
The standalone voter strategies and the ensemble share these functions (single source of truth).

Definitions are the legacy ones, configured per spec OD1(a): Wilder RSI (``2n-1`` span), EMA
and MACD with ``ewm=True``, Bollinger ``ddof=1``, crossovers from vectorbt (strict
inequalities, so an exact float tie can differ from the legacy ``prev <= b`` form). The MACD
divergence rule, its warm-up mask and the volume spike are ggTrader rules (a few lines each).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ggTrader.data.bars import Bars
from ggTrader.engine import indicators as ind
from ggTrader.engine import signals as sig
from ggTrader.lab.strategies.vote import BoolPair, VoterArrays

Params = tuple[Hashable, ...]


def _unique(values: Iterable[Hashable]) -> list[Hashable]:
    """Distinct values in first-seen order."""
    return list(dict.fromkeys(values))


def _blocks(frame: pd.DataFrame, n_blocks: int, symbols: Sequence[str]) -> list[pd.DataFrame]:
    """Split a ``dates x (blocks * symbols)`` indicator output into per-parameter frames.

    vectorbt stacks one block of ``symbols`` per parameter value, in parameter order, and
    rebuilds the output frame on every attribute access, so callers keep the result in a local.
    """
    n = len(symbols)
    if frame.shape[1] != n_blocks * n:
        raise ValueError(f"expected {n_blocks * n} columns, got {frame.shape[1]}")
    values = frame.to_numpy()
    return [
        pd.DataFrame(values[:, k * n : (k + 1) * n], index=frame.index, columns=list(symbols))
        for k in range(n_blocks)
    ]


def _arr(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(frame.to_numpy(), dtype=bool)


def _one_or_list(values: Sequence[Hashable]) -> Hashable | list[Hashable]:
    return values[0] if len(values) == 1 else list(values)


def bb_bands(
    close: pd.DataFrame, pairs: Sequence[Params]
) -> dict[Params, tuple[np.ndarray, np.ndarray]]:
    """Lower and middle Bollinger band arrays per ``(bb_period, bb_std)`` (ddof=1 preset)."""
    symbols = list(close.columns)
    out: dict[Params, tuple[np.ndarray, np.ndarray]] = {}
    for period in _unique(p[0] for p in pairs):
        stds = _unique(p[1] for p in pairs if p[0] == period)
        bb = ind.bbands(close, int(period), _one_or_list([float(s) for s in stds]))
        lower, middle = bb.lower, bb.middle  # locals: vectorbt rebuilds the frames per access
        for std, low, mid in zip(
            stds,
            _blocks(lower, len(stds), symbols),
            _blocks(middle, len(stds), symbols),
            strict=True,
        ):
            out[(period, std)] = (low.to_numpy(), mid.to_numpy())
    return out


def bb_pairs(close: pd.DataFrame, pairs: Sequence[Params]) -> dict[Params, BoolPair]:
    """Bollinger voter: entry = close crosses below the lower band, exit = above the middle.

    Args:
        close: ``dates x symbols`` closes.
        pairs: distinct ``(bb_period, bb_std)`` tuples.
    """
    cols = list(close.columns)
    out: dict[Params, BoolPair] = {}
    for key, (low, mid) in bb_bands(close, pairs).items():
        lower = pd.DataFrame(low, index=close.index, columns=cols)
        middle = pd.DataFrame(mid, index=close.index, columns=cols)
        out[key] = (_arr(sig.cross_below(close, lower)), _arr(sig.cross_above(close, middle)))
    return out


def rsi_triples(close: pd.DataFrame, triples: Sequence[Params]) -> dict[Params, BoolPair]:
    """RSI voter: entry = RSI crosses below ``oversold``, exit = above ``exit_level``.

    Args:
        triples: distinct ``(rsi_period, rsi_oversold, rsi_exit)`` tuples.
    """
    symbols = list(close.columns)
    periods = _unique(t[0] for t in triples)
    rsi = ind.wilder_rsi(close, _one_or_list([int(p) for p in periods])).rsi
    frames = dict(zip(periods, _blocks(rsi, len(periods), symbols), strict=True))
    entry_cache: dict[Params, np.ndarray] = {}
    exit_cache: dict[Params, np.ndarray] = {}
    out: dict[Params, BoolPair] = {}
    for period, oversold, exit_level in triples:
        if (period, oversold) not in entry_cache:
            entry_cache[(period, oversold)] = _arr(sig.cross_below(frames[period], float(oversold)))
        if (period, exit_level) not in exit_cache:
            exit_cache[(period, exit_level)] = _arr(
                sig.cross_above(frames[period], float(exit_level))
            )
        out[(period, oversold, exit_level)] = (
            entry_cache[(period, oversold)],
            exit_cache[(period, exit_level)],
        )
    return out


def ema_pairs(close: pd.DataFrame, pairs: Sequence[Params]) -> dict[Params, BoolPair]:
    """EMA-cross voter: entry = fast crosses above slow, exit = fast crosses below slow.

    Args:
        pairs: distinct ``(ema_fast, ema_slow)`` tuples.
    """
    symbols = list(close.columns)
    windows = sorted({int(w) for p in pairs for w in p})
    ma = ind.ema(close, _one_or_list(windows)).ma
    frames = dict(zip(windows, _blocks(ma, len(windows), symbols), strict=True))
    return {
        (fast, slow): (
            _arr(sig.cross_above(frames[int(fast)], frames[int(slow)])),
            _arr(sig.cross_below(frames[int(fast)], frames[int(slow)])),
        )
        for fast, slow in pairs
    }


def macd_quads(close: pd.DataFrame, quads: Sequence[Params]) -> dict[Params, BoolPair]:
    """MACD voter: bullish divergence entry, histogram-crosses-below-zero exit.

    Entry = close at its rolling low while the histogram is above its own rolling low, masked
    for the first ``slow + signal + window`` bars (a global bar count, as the legacy rule).

    Args:
        quads: distinct ``(macd_fast, macd_slow, macd_signal, divergence_window)`` tuples.
    """
    symbols = list(close.columns)
    out: dict[Params, BoolPair] = {}
    hist_cache: dict[Params, pd.DataFrame] = {}
    for fast, slow, signal, window in quads:
        key = (fast, slow, signal)
        if key not in hist_cache:
            hist_cache[key] = _blocks(
                ind.macd(close, int(fast), int(slow), int(signal)).hist, 1, symbols
            )[0]
        hist = hist_cache[key]
        w = int(window)
        price_low = close.rolling(window=w, min_periods=w).min()
        hist_low = hist.rolling(window=w, min_periods=w).min()
        entry = np.array(_arr((close <= price_low) & (hist > hist_low)))  # writable copy
        entry[: int(slow) + int(signal) + w] = False
        out[(fast, slow, signal, window)] = (entry, _arr(sig.cross_below(hist, 0.0)))
    return out


def vbb_quads(
    close: pd.DataFrame, volume: pd.DataFrame, quads: Sequence[Params]
) -> dict[Params, BoolPair]:
    """Volume-BB voter: Bollinger entry AND a volume spike; exit = Bollinger middle cross.

    The exit is identical to the plain Bollinger voter's exit; the ensemble counts both
    (the double vote, spec OD3a, kept for parity).

    Args:
        quads: distinct ``(bb_period, bb_std, vol_period, vol_mult)`` tuples.
    """
    symbols = list(close.columns)
    bb = bb_pairs(close, _unique((q[0], q[1]) for q in quads))
    vol_periods = _unique(q[2] for q in quads)
    vma = ind.sma(volume, _one_or_list([int(p) for p in vol_periods])).ma
    vma_frames = dict(zip(vol_periods, _blocks(vma, len(vol_periods), symbols), strict=True))
    vol = volume.to_numpy()
    out: dict[Params, BoolPair] = {}
    for period, std, vol_period, vol_mult in quads:
        spike = vol > float(vol_mult) * vma_frames[vol_period].to_numpy()
        entry, exit_ = bb[(period, std)]
        out[(period, std, vol_period, vol_mult)] = (entry & spike, exit_)
    return out


@dataclass(frozen=True)
class VoterSpec:
    """A voter: the combination keys it depends on and how to compute its arrays."""

    name: str
    keys: tuple[str, ...]
    compute: Callable[[Bars, Sequence[Params]], dict[Params, BoolPair]]


VOTER_SPECS: dict[str, VoterSpec] = {
    "bb": VoterSpec("bb", ("bb_period", "bb_std"), lambda b, p: bb_pairs(b.close, p)),
    "rsi": VoterSpec(
        "rsi", ("rsi_period", "rsi_oversold", "rsi_exit"), lambda b, p: rsi_triples(b.close, p)
    ),
    "ema": VoterSpec("ema", ("ema_fast", "ema_slow"), lambda b, p: ema_pairs(b.close, p)),
    "macd": VoterSpec(
        "macd",
        ("macd_fast", "macd_slow", "macd_signal", "divergence_window"),
        lambda b, p: macd_quads(b.close, p),
    ),
    "vbb": VoterSpec(
        "vbb",
        ("bb_period", "bb_std", "vol_period", "vol_mult"),
        lambda b, p: vbb_quads(b.close, b.volume, p),
    ),
}

#: Defaults of the legacy constructors (``EnsembleSignal`` and the standalone strategies).
VOTER_DEFAULTS: dict[str, float | int] = {
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_exit": 50,
    "ema_fast": 20,
    "ema_slow": 50,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "divergence_window": 20,
    "vol_period": 20,
    "vol_mult": 2.0,
}


def voter_arrays(name: str, bars: Bars, combos: Sequence[dict[str, Hashable]]) -> VoterArrays:
    """Compute one voter once per distinct parameter tuple used by ``combos``."""
    spec = VOTER_SPECS[name]
    params = _unique(tuple(c[k] for k in spec.keys) for c in combos)
    return VoterArrays(name=name, keys=spec.keys, arrays=spec.compute(bars, params))
