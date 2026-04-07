"""Lightweight performance metrics for live trading data.

These functions operate directly on pandas Series of trade PnL or balance
snapshots — they do NOT require VectorBT Portfolio objects (unlike
``ggTrader.core.metrics`` which is built for backtest portfolios).

Use these for daily PnL reports, dashboards, and any analysis of live
TradeTracker CSV data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_sharpe_ratio(
    returns: pd.Series, periods_per_year: int = 365, rf: float = 0.0
) -> float:
    """Annualised Sharpe ratio from a returns series.

    Args:
        returns: Series of period returns (e.g. daily). Should NOT include NaN.
        periods_per_year: 365 for daily, 252 for trading days, 8760 for hourly, etc.
        rf: Risk-free rate per period (default 0).

    Returns:
        Annualised Sharpe. NaN if std is zero or series is empty.
    """
    if returns is None or len(returns) < 2:
        return float("nan")
    excess = returns - rf
    std = float(excess.std(ddof=1))
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def compute_sortino_ratio(
    returns: pd.Series, periods_per_year: int = 365, rf: float = 0.0
) -> float:
    """Annualised Sortino ratio (penalises only downside volatility)."""
    if returns is None or len(returns) < 2:
        return float("nan")
    excess = returns - rf
    downside = excess[excess < 0]
    if len(downside) < 1:
        return float("inf") if excess.mean() > 0 else float("nan")
    downside_std = float(np.sqrt((downside ** 2).mean()))
    if downside_std == 0:
        return float("nan")
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def compute_max_drawdown(
    equity: pd.Series,
) -> tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Maximum drawdown from an equity curve.

    Returns:
        (max_drawdown_pct, peak_timestamp, trough_timestamp). max_drawdown_pct
        is negative (e.g. -0.25 for a 25% drawdown). Returns (0.0, None, None)
        if the series is empty.
    """
    if equity is None or len(equity) == 0:
        return 0.0, None, None
    eq = equity.dropna()
    if eq.empty:
        return 0.0, None, None
    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max
    trough_idx = drawdown.idxmin()
    max_dd = float(drawdown.loc[trough_idx])
    if max_dd >= 0:
        return 0.0, None, None
    peak_idx = eq.loc[:trough_idx].idxmax()
    return max_dd, peak_idx, trough_idx


def compute_calmar_ratio(equity: pd.Series, periods_per_year: int = 365) -> float:
    """Calmar = annualised return / |max drawdown|."""
    if equity is None or len(equity) < 2:
        return float("nan")
    eq = equity.dropna()
    if len(eq) < 2:
        return float("nan")
    n_periods = len(eq) - 1
    if n_periods <= 0:
        return float("nan")
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1.0
    annualised = (1.0 + total_return) ** (periods_per_year / n_periods) - 1.0
    max_dd, _, _ = compute_max_drawdown(eq)
    if max_dd >= 0:
        return float("nan")
    return float(annualised / abs(max_dd))


def compute_consecutive_losses(net_pnl: pd.Series) -> int:
    """Length of the most recent consecutive losing streak (0 if last trade was a win)."""
    if net_pnl is None or net_pnl.empty:
        return 0
    streak = 0
    for v in reversed(net_pnl.tolist()):
        if v <= 0:
            streak += 1
        else:
            break
    return streak


def equity_curve_from_balance(balances: pd.DataFrame) -> pd.Series:
    """Extract a clean equity Series from a TradeTracker balance_snapshots DataFrame."""
    if balances is None or balances.empty or "total_usd" not in balances.columns:
        return pd.Series(dtype=float)
    df = balances.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return pd.Series(df["total_usd"].values, index=df["timestamp"], name="equity")


def daily_returns_from_equity(equity: pd.Series) -> pd.Series:
    """Resample an equity Series to daily and compute pct changes."""
    if equity is None or equity.empty:
        return pd.Series(dtype=float)
    daily = equity.resample("1D").last().dropna()
    if len(daily) < 2:
        return pd.Series(dtype=float)
    return daily.pct_change().dropna()


def summarise_window(
    closes: pd.DataFrame,
    balances: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    """Aggregate stats for a trade-close window plus equity-based metrics over the same range.

    Args:
        closes: TradeTracker.get_closed_positions() DataFrame
        balances: TradeTracker.get_balance_history() DataFrame
        start, end: Timezone-aware (UTC) bounds

    Returns dict with: trades, wins, losses, win_rate, net_pnl, gross_pnl,
        best, worst, sharpe, sortino, max_dd, calmar, consecutive_losses,
        balance_start, balance_end, balance_change_pct.
    """
    out: dict = {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "net_pnl": 0.0,
        "gross_pnl": 0.0,
        "fees": 0.0,
        "best": None,
        "worst": None,
        "best_symbol": None,
        "worst_symbol": None,
        "sharpe": float("nan"),
        "sortino": float("nan"),
        "max_dd": 0.0,
        "calmar": float("nan"),
        "consecutive_losses": 0,
        "balance_start": None,
        "balance_end": None,
        "balance_change_pct": None,
    }

    # Trade-based stats
    if closes is not None and not closes.empty:
        df = closes.copy()
        df["close_timestamp"] = pd.to_datetime(df["close_timestamp"], utc=True)
        window = df[(df["close_timestamp"] >= start) & (df["close_timestamp"] <= end)]
        if not window.empty:
            window = window.sort_values("close_timestamp")
            out["trades"] = int(len(window))
            wins_df = window[window["net_pnl"] > 0]
            losses_df = window[window["net_pnl"] <= 0]
            out["wins"] = int(len(wins_df))
            out["losses"] = int(len(losses_df))
            out["win_rate"] = (out["wins"] / out["trades"] * 100.0) if out["trades"] else 0.0
            out["net_pnl"] = float(window["net_pnl"].sum())
            out["gross_pnl"] = float(window["gross_pnl"].sum())
            out["fees"] = float(window["fee_total"].sum())
            best_idx = window["net_pnl"].idxmax()
            worst_idx = window["net_pnl"].idxmin()
            out["best"] = float(window.loc[best_idx, "net_pnl"])
            out["worst"] = float(window.loc[worst_idx, "net_pnl"])
            out["best_symbol"] = str(window.loc[best_idx, "symbol"])
            out["worst_symbol"] = str(window.loc[worst_idx, "symbol"])
            out["consecutive_losses"] = compute_consecutive_losses(window["net_pnl"])

    # Equity-based stats over the same window
    if balances is not None and not balances.empty:
        equity = equity_curve_from_balance(balances)
        if not equity.empty:
            window_eq = equity[(equity.index >= start) & (equity.index <= end)]
            if len(window_eq) >= 2:
                out["balance_start"] = float(window_eq.iloc[0])
                out["balance_end"] = float(window_eq.iloc[-1])
                out["balance_change_pct"] = (
                    (out["balance_end"] / out["balance_start"]) - 1.0
                ) * 100.0
                out["max_dd"] = float(compute_max_drawdown(window_eq)[0]) * 100.0
                returns = daily_returns_from_equity(window_eq)
                if len(returns) >= 2:
                    out["sharpe"] = compute_sharpe_ratio(returns)
                    out["sortino"] = compute_sortino_ratio(returns)
                    out["calmar"] = compute_calmar_ratio(window_eq)
            elif len(window_eq) == 1:
                out["balance_end"] = float(window_eq.iloc[0])

    return out
