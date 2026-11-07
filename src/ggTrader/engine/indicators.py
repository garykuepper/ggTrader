"""Indicator presets: vectorbt's own kernels with every behaviour flag written out.

vectorbt's defaults are a different strategy from ggTrader's (RSI is Cutler's SMA form,
MACD is the SMA form, Bollinger uses ddof=0, ATR is span-EWM). Each preset pins the flags
that reproduce the legacy pandas conventions (spec OD1(a), epoch E1) so a changed upstream
default can never alter a strategy silently. There is no maths here: every function is one
``vbt.<Indicator>.run`` call and returns the vectorbt indicator object.

Constant flags are hidden from the output column levels (``hide_params``); only values that
vary, such as the window or Bollinger alpha, appear as levels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import vectorbt as vbt

Frame = pd.DataFrame | pd.Series
IntOrSeq = int | Sequence[int]
FloatOrSeq = float | Sequence[float]


def values(output: Frame, like: Frame) -> Frame:
    """An indicator output with its parameter level removed: same columns as ``like``.

    A preset called with one window returns columns ``(ma_window, symbol)``; two such outputs
    do not line up for ``>`` or a crossover until the level is gone. This relabels the output
    with the columns of the input it was computed from (``like``, usually the close frame).

    Args:
        output: one output of a preset, e.g. ``sma(close, 20).ma``.
        like: the frame the preset was run on.

    Raises:
        ValueError: the output holds several parameter values (more columns than ``like``);
            pick one by label first.
    """
    if isinstance(output, pd.Series):
        return output.rename(getattr(like, "name", None))
    if output.shape[1] != like.shape[1]:
        raise ValueError(
            f"output has {output.shape[1]} columns for {like.shape[1]} inputs: it holds several "
            "parameter values; select one before stripping the parameter level"
        )
    return output.set_axis(like.columns, axis=1)


def _positive_ints(value: IntOrSeq, name: str) -> int | list[int]:
    """Validate a window (or a grid of windows); vectorbt would accept 0 or negatives."""
    values = [value] if isinstance(value, int) else list(value)
    if not values or any((not isinstance(v, int)) or v < 1 for v in values):
        raise ValueError(f"{name} must be a positive int or a sequence of them, got {value!r}")
    return values[0] if isinstance(value, int) else values


def wilder_rsi(close: Frame, n: IntOrSeq = 14) -> Any:
    """Wilder RSI of period ``n``: ``vbt.RSI.run(window=2n-1, ewm=True)``.

    Legacy equivalent: ``close.diff()`` gains/losses smoothed with
    ``ewm(alpha=1/n, adjust=False)``. vectorbt's EWM is span-based, and span ``2n-1`` has
    ``alpha = 2/(2n) = 1/n``, so the values are equal (max abs diff 0.0 on the 600-bar test
    panel). The first valid bar is ``2n-1`` (27 for n=14), not ``n``. A window with no losses
    gives 100 (the textbook value), where the legacy form gave NaN.

    Args:
        close: close prices (time x symbol).
        n: Wilder period, or a grid of periods. The ``rsi_window`` output level reports 2n-1.
    """
    window = _positive_ints(n, "n")
    windows = [2 * w - 1 for w in window] if isinstance(window, list) else 2 * window - 1
    return vbt.RSI.run(close, window=windows, ewm=True, adjust=False, hide_params=["ewm"])


def bbands(close: Frame, window: IntOrSeq = 20, alpha: FloatOrSeq = 2.0) -> Any:
    """Bollinger bands on a simple average with the sample standard deviation (ddof=1).

    ``vbt.BBANDS.run(..., ewm=False, ddof=1)``. vectorbt's default is ``ddof=0``, which moves
    the lower band by about 0.2 on a 100-priced series and flips ~8% of the entries. Legacy
    equivalent: ``close.rolling(w).mean() -/+ alpha * close.rolling(w).std()`` (pandas std is
    ddof=1); max abs diff ~3e-13 on the test panel.
    """
    return vbt.BBANDS.run(
        close,
        window=_positive_ints(window, "window"),
        alpha=alpha,
        ewm=False,
        ddof=1,
        adjust=False,
        hide_params=["ewm"],
    )


def sma(close: Frame, window: IntOrSeq) -> Any:
    """Simple moving average: ``vbt.MA.run(ewm=False)``.

    Legacy equivalent: ``close.rolling(w, min_periods=w).mean()`` (max abs diff ~3e-13).
    """
    return vbt.MA.run(
        close, window=_positive_ints(window, "window"), ewm=False, adjust=False, hide_params=["ewm"]
    )


def ema(close: Frame, window: IntOrSeq) -> Any:
    """Exponential moving average: ``vbt.MA.run(ewm=True)``.

    Legacy equivalent: ``close.ewm(span=w, adjust=False).mean()`` seeded at bar 0 (max abs
    diff 0.0). vectorbt masks the first ``w-1`` bars as NaN where the legacy form emitted
    seed-dominated values (spec OD1: adopt vbt's NaN warm-up).
    """
    return vbt.MA.run(
        close, window=_positive_ints(window, "window"), ewm=True, adjust=False, hide_params=["ewm"]
    )


def macd(close: Frame, fast: IntOrSeq = 12, slow: IntOrSeq = 26, signal: IntOrSeq = 9) -> Any:
    """Appel's MACD: ``vbt.MACD.run(macd_ewm=True, signal_ewm=True)``.

    vectorbt's default is the SMA form. The MACD line equals the legacy
    ``ewm(span=fast) - ewm(span=slow)`` (diff 0.0) from bar ``slow-1``. vectorbt seeds the
    signal EMA at the first valid MACD value where the legacy form seeded at bar 0, so the
    histogram differs by up to 0.12 early and below 1e-9 after ~200 bars.
    """
    return vbt.MACD.run(
        close,
        fast_window=_positive_ints(fast, "fast"),
        slow_window=_positive_ints(slow, "slow"),
        signal_window=_positive_ints(signal, "signal"),
        macd_ewm=True,
        signal_ewm=True,
        adjust=False,
        hide_params=["macd_ewm", "signal_ewm"],
    )


def atr(high: Frame, low: Frame, close: Frame, window: IntOrSeq = 14) -> Any:
    """Average true range as a simple average: ``vbt.ATR.run(ewm=False)``.

    vectorbt's default is span-EWM. Legacy equivalent: rolling mean of the true range
    (diff 0.0 where both are valid). vectorbt takes the first true range as ``high - low``,
    so its first ATR is one bar earlier (bar ``w-1`` against bar ``w``).
    """
    return vbt.ATR.run(
        high,
        low,
        close,
        window=_positive_ints(window, "window"),
        ewm=False,
        adjust=False,
        hide_params=["ewm"],
    )
