"""Window statistics from returns via vectorbt's returns accessor, plus ggTrader's guards.

Every statistic is a vectorbt call on ``returns.vbt.returns(freq="1D", year_freq="252 days")``;
this module adds only what vectorbt lacks (spec 5.6):

* a **finite guard**: vectorbt returns ``+inf`` Sharpe for a flat curve and ~4.5e16 for
  constant returns, ``+inf`` Sortino with no downside. Those become NaN (flat or zero-trade
  windows, fewer than two observations) before any ``idxmax``, rank or composite, so a dead
  combination can never win a fold.
* per-window **trade counts** (OD12): trades whose entry bar lies inside the scoring window,
  not the whole-simulation count.
* a **zero-trade guard** (spec 5.6): a window with no entry blanks Sharpe and Sortino, the
  trade-dependent ratios, when the caller supplies the counts.
* a **losing-day guard** (P5b, review MAJOR-1): a Sortino built on fewer than
  ``config.MIN_LOSING_DAYS`` losing days is NaN. The downside deviation of an almost flat curve
  is estimated from a handful of points, so the ratio explodes (28.6 from one losing day in the
  17-fold core run) and a near-flat combination wins the composite. Like the flat-curve guard
  this is a degeneracy guard, not a new definition: above the threshold the number is
  vectorbt's, unchanged.
* the configured **deflated Sharpe ratio** that reproduces the legacy ``gates.dsr_check``.

Definitions (owner decision 2026-11-06: where a ggTrader definition and vectorbt's differ,
vectorbt's wins): Sharpe, volatility and max drawdown at 252 bars with ddof=1; Sortino is
vectorbt's textbook downside deviation; CAGR is vectorbt's bar-annualised
``prod(1+r)**(252/N)-1``. The 252-day year and the flat-curve guard are not definition
overrides: one is the trading-day calendar (vectorbt's 365-day default is a crypto setting and
inflates equity Sharpe by about 20%), the other a safety guard.

Window convention (review MAJOR-1): ``window_stats(returns, a, b)`` scores the bars
``a..b-1``, exactly ``curve_stats(equity.iloc[a:b])``: the equity is rescaled to its first bar
and the returns start at the *second* bar, so the return into bar ``a`` is not counted. The
drawdown path additionally starts at the base bar (value 1.0), as ``curve / curve.cummax()``
does; vectorbt's own returns-based drawdown would otherwise miss a fall from the first bar.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd
import vectorbt as vbt  # noqa: F401  (registers the .vbt accessor)

from ggTrader.engine import config

#: Return std below this is a flat curve: vectorbt would divide by ~0 (spec 5.6).
MIN_STD = 1e-12
DDOF = 1
STAT_ORDER = ("sharpe", "volatility", "max_drawdown", "sortino", "cagr", "n_obs")
#: Per-statistic definition labels stamped into provenance.
DEFINITION_LABELS = {
    "sortino": "vectorbt downside deviation (textbook)",
    "sortino_guard": f"NaN below {config.MIN_LOSING_DAYS} losing days (degeneracy guard)",
    "zero_trade_guard": "NaN Sharpe and Sortino when no entry falls inside a selection window",
    "cagr": "vectorbt annualized: prod(1+r)**(252/N)-1",
}

Returns = pd.Series | pd.DataFrame


def _accessor(returns: Returns):
    """The vectorbt returns accessor with the explicit 252-day year and daily frequency."""
    return returns.vbt.returns(freq=config.BAR_FREQ, year_freq=config.YEAR_FREQ)


def _scalar_or_series(result: pd.Series | float, returns: Returns) -> float | pd.Series:
    """vectorbt returns a Series per column; a Series input gets a plain float."""
    if isinstance(returns, pd.Series):
        return float(result) if not isinstance(result, pd.Series) else float(result.iloc[0])
    return result


def _nan_if_nonfinite(values: pd.Series | float) -> pd.Series | float:
    """inf and -inf become NaN."""
    if isinstance(values, pd.Series):
        return values.where(np.isfinite(values), np.nan)
    return float(values) if np.isfinite(values) else float("nan")


def _std(returns: Returns) -> pd.Series | float:
    """Sample std ignoring NaN; only used to detect a flat curve, not as a statistic."""
    values = returns.to_numpy(dtype=float)
    # np.nanstd warns (outside errstate) for a column with fewer than two finite values; the
    # NaN it returns is exactly what the callers want.
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.filterwarnings(
            "ignore", message="Degrees of freedom <= 0", category=RuntimeWarning
        )
        std = (
            np.nanstd(values, axis=0, ddof=DDOF)
            if len(values) > DDOF
            else np.full(values.shape[1:] or (), np.nan)
        )
    if isinstance(returns, pd.Series):
        return float(std)
    return pd.Series(std, index=returns.columns)


def _n_obs(returns: Returns) -> pd.Series | int:
    counts = returns.notna().sum()
    return int(counts) if isinstance(returns, pd.Series) else counts


def _zero_trades_to_nan(out: float | pd.Series, n_trades: pd.Series | int | None):
    """NaN where a column has no trade; a column missing from the counts has none."""
    if n_trades is None:
        return out
    if isinstance(out, pd.Series):
        counts = n_trades.reindex(out.index).fillna(0) if hasattr(n_trades, "reindex") else n_trades
        return out.where(counts > 0, np.nan)
    return float("nan") if int(np.asarray(n_trades).ravel()[0]) == 0 else out


def sharpe(returns: Returns, n_trades: pd.Series | None = None) -> float | pd.Series:
    """Annualised Sharpe (252 bars, ddof=1), NaN for flat curves and zero-trade windows.

    Args:
        returns: per-bar returns, a series or a (time x combination) frame.
        n_trades: optional trades per column; zero trades makes that column NaN. The caller
            decides what "trades" means (OD12: entries inside the scoring window). A window that
            only holds carried-in positions has real returns but zero entries and is blanked
            too (spec 5.6 as written); the walk-forward applies the rule to its selection
            windows only, the out-of-sample window being a measurement of a book that exists.
    """
    raw = _scalar_or_series(_accessor(returns).sharpe_ratio(ddof=DDOF), returns)
    out = _nan_if_nonfinite(raw)
    std, n_obs = _std(returns), _n_obs(returns)
    if isinstance(out, pd.Series):
        out = out.where((std >= MIN_STD) & (n_obs >= 2), np.nan)
        return _zero_trades_to_nan(out, n_trades)
    if not (std >= MIN_STD and n_obs >= 2):
        return float("nan")
    return _zero_trades_to_nan(out, n_trades)


def volatility(returns: Returns) -> float | pd.Series:
    """Annualised volatility (252 bars, ddof=1)."""
    return _nan_if_nonfinite(
        _scalar_or_series(_accessor(returns).annualized_volatility(ddof=DDOF), returns)
    )


def max_drawdown(returns: Returns) -> float | pd.Series:
    """Maximum drawdown of the compounded path, a negative fraction."""
    return _scalar_or_series(_accessor(returns).max_drawdown(), returns)


def losing_days(returns: Returns) -> pd.Series | int:
    """Number of strictly negative returns per column (a zero return is not a losing day)."""
    count = (returns < 0).sum()
    return int(count) if isinstance(returns, pd.Series) else count


def sortino(
    returns: Returns, n_trades: pd.Series | None = None, min_losing_days: int = 0
) -> float | pd.Series:
    """vectorbt's Sortino (textbook downside deviation); NaN with no downside.

    Args:
        returns: per-bar returns, a series or a (time x combination) frame.
        n_trades: optional trades per column; zero trades makes that column NaN (as ``sharpe``).
        min_losing_days: a column with fewer losing days is NaN (degeneracy guard, see the
            module docstring). 0 turns it off; ``window_stats`` defaults to
            ``config.MIN_LOSING_DAYS``.
    """
    out = _nan_if_nonfinite(_scalar_or_series(_accessor(returns).sortino_ratio(), returns))
    if min_losing_days > 0:
        enough = losing_days(returns) >= min_losing_days
        out = (
            out.where(enough, np.nan)
            if isinstance(out, pd.Series)
            else (out if enough else float("nan"))
        )
    return _zero_trades_to_nan(out, n_trades)


def cagr(returns: Returns) -> float | pd.Series:
    """Bar-annualised compound return, ``prod(1+r)**(252/N) - 1`` (vectorbt ``annualized``)."""
    return _nan_if_nonfinite(_scalar_or_series(_accessor(returns).annualized(), returns))


def total_return(returns: Returns) -> float | pd.Series:
    """Compounded return of a window of per-bar returns, ``prod(1 + r) - 1`` (vectorbt ``total``).

    The caller passes the window's returns (``returns.iloc[start + 1:end]``, see
    :func:`window_stats`), so this equals ``equity[end - 1] / equity[start] - 1``.
    """
    return _scalar_or_series(_accessor(returns).total(), returns)


def _empty_window(window: pd.DataFrame) -> pd.DataFrame:
    """A window with fewer than two returns: every statistic NaN, ``n_obs`` as counted."""
    table = pd.DataFrame(np.nan, index=list(STAT_ORDER), columns=window.columns)
    table.loc["n_obs"] = _n_obs(window)
    return table


def window_stats(
    returns: Returns,
    start: int,
    end: int,
    n_trades: pd.Series | None = None,
    min_losing_days: int = config.MIN_LOSING_DAYS,
) -> pd.DataFrame:
    """Statistics of the equity bars ``start..end-1``, one column per combination.

    Follows the ``curve_stats`` window convention, ``curve_stats(equity.iloc[start:end])``: the
    window
    is bars ``start`` to ``end - 1`` of the equity, rescaled to start at bar ``start``, so the
    returns used are ``returns.iloc[start + 1:end]`` and ``n_obs`` is ``end - start - 1``. A
    window with fewer than two returns gives NaN for every statistic.

    Args:
        returns: per-bar returns indexed by bar date (a series or a time x combination frame).
        start: bar position of the window's first bar.
        end: bar position after the window's last bar.
        n_trades: optional entries inside the window per column (OD12); a column with none gets
            NaN Sharpe and Sortino (spec 5.6). ``None`` leaves the zero-trade guard off.
        min_losing_days: Sortino is NaN below this many losing days (``config.MIN_LOSING_DAYS``
            by default; 0 turns the guard off).

    Returns:
        Rows are ``STAT_ORDER``. A series input becomes a one-column frame.
    """
    frame = returns
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(frame.name if frame.name is not None else "returns")
        if n_trades is not None:  # one column: a scalar or one-element count belongs to it
            n_trades = pd.Series({frame.columns[0]: int(np.asarray(n_trades).ravel()[0])})
    window = frame.iloc[start + 1 : end]
    if len(window) < 2 or end - start < 2:
        return _empty_window(window)
    # vectorbt's drawdown path starts after the first return it is given, so the base bar would
    # never count as a peak; legacy ``curve / curve.cummax()`` does count it. A leading zero
    # return makes the base bar the path's first value (found on real tape, review follow-up).
    base = frame.iloc[start:end].copy()
    base.iloc[0] = 0.0
    table = pd.DataFrame(
        {
            "sharpe": sharpe(window, n_trades),
            "volatility": volatility(window),
            "max_drawdown": max_drawdown(base),
            "sortino": sortino(window, n_trades, min_losing_days),
            "cagr": cagr(window),
            "n_obs": _n_obs(window),
        }
    )
    return table.T.loc[list(STAT_ORDER)]


def trade_counts(
    trades: pd.DataFrame, start: int, end: int, groups: Sequence[Hashable] | None = None
) -> pd.Series:
    """Trades per group whose entry bar lies in ``[start, end)`` (OD12).

    Args:
        trades: ``SimResult.trades`` (needs ``group`` and ``entry_idx``).
        start: first bar position of the scoring window.
        end: bar position after the last bar of the window.
        groups: labels to report even when they have no trades.
    """
    inside = trades[(trades["entry_idx"] >= start) & (trades["entry_idx"] < end)]
    counts = inside.groupby("group", sort=False).size()
    labels = list(groups) if groups is not None else list(dict.fromkeys(trades["group"]))
    return pd.Series({label: int(counts.get(label, 0)) for label in labels}, dtype="int64")


def dsr(returns: Returns, n_trials: int) -> float | pd.Series:
    """Deflated Sharpe ratio configured to reproduce the legacy ``gates.dsr_check``.

    ``deflated_sharpe_ratio(nb_trials=n_trials, var_sharpe=252/(T-1), bias=False)``:

    * ``var_sharpe = 252/(T-1)``: vectorbt 1.2.0 divides it by the annualisation factor, so
      the per-period variance is the legacy null ``1/(T-1)``. The report value ``1/(T-1)``
      gives 0.99 against the legacy 0.62 and would loosen the 0.80 gate (spec Appendix A).
    * ``bias=False``: the legacy inputs are pandas ``skew()``/``kurtosis()``, which are the
      unbiased estimators. ``bias=True`` (the spec's first draft) differs by ~1e-6 absolute.
    * ``n_trials == 1`` (a one-combination grid): nothing was chosen among alternatives, so
      nothing is deflated and the value is the probabilistic Sharpe ratio against a zero
      benchmark. vectorbt's expected-maximum term is ``-inf`` for one trial (``ppf(0)``), so
      the call uses two trials with ``var_sharpe=0``, which makes that term exactly 0. This
      equals the legacy ``dsr_check(n_trials=1)`` to 1e-12.

    Args:
        returns: per-bar returns of the selected combination(s).
        n_trials: number of combinations tried (``len(grid)``); at least 1.

    Raises:
        ValueError: ``n_trials`` is below 1.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    out = pd.Series(np.nan, index=frame.columns, dtype=float)
    n_obs = frame.notna().sum()
    for count, columns in n_obs.groupby(n_obs):
        if count < 3:
            continue
        cols = list(columns.index)
        trials, var_sharpe = (
            (n_trials, config.TRADING_DAYS / (count - 1)) if n_trials > 1 else (2, 0.0)
        )
        value = _accessor(frame[cols]).deflated_sharpe_ratio(
            nb_trials=trials, var_sharpe=var_sharpe, bias=False
        )
        out[cols] = value.to_numpy() if isinstance(value, pd.Series) else float(value)
    out = _nan_if_nonfinite(out)
    return float(out.iloc[0]) if isinstance(returns, pd.Series) else out
