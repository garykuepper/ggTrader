"""Benchmark statistics: CAGR, BTC B&H, and stats enrichment."""

from typing import Any, Dict, Optional

import pandas as pd
import vectorbt as vbt

from ggTrader.core.orchestrator_utils import _as_optional_float


def _years_from_price_index(index: Any) -> float:
    """Calendar years between first and last bar (for CAGR)."""
    import math

    if index is None:
        return float("nan")
    try:
        idx = pd.to_datetime(pd.Index(index))
    except (TypeError, ValueError):
        return float("nan")
    if len(idx) < 2:
        return float("nan")
    delta = idx[-1] - idx[0]
    sec = float(delta.total_seconds())
    if sec <= 0 or not math.isfinite(sec):
        return float("nan")
    return sec / (365.25 * 24 * 3600.0)


def _cagr_percent(total_return_pct: float, years: float) -> float:
    """Annualized geometric return (%) from total return (%) over ``years``."""
    import math

    if not (years > 0 and math.isfinite(years)):
        return float("nan")
    try:
        mult = 1.0 + float(total_return_pct) / 100.0
    except (TypeError, ValueError):
        return float("nan")
    if mult <= 0:
        return float("nan")
    return (mult ** (1.0 / years) - 1.0) * 100.0


def _buy_hold_portfolio_stats(
    close_df: pd.DataFrame,
    label: str,
    config: Dict[str, Any],
    fees: float = 0.0,
    slippage: float = 0.0,
) -> Dict[str, Any]:
    """
    Generic buy-and-hold helper: buy on first valid bar, sell on last valid bar.

    Parameters
    ----------
    close_df : single-column DataFrame with DatetimeIndex
    label    : column name used for entries/exits DataFrame
    config   : pipeline config (START_CASH, FREQ)
    fees     : trading fee fraction (default 0 for index benchmarks)
    slippage : slippage fraction
    """
    empty: Dict[str, Any] = {
        "profit_pct": None,
        "cagr_pct": None,
        "sharpe": None,
        "max_drawdown": None,
        "total_trades": 0,
    }
    if close_df is None or close_df.empty or close_df.shape[0] < 2:
        return empty

    col = close_df.columns[0]
    first_valid = close_df[col].first_valid_index()
    last_valid = close_df[col].last_valid_index()
    if first_valid is None or last_valid is None or first_valid >= last_valid:
        return empty

    entries = pd.DataFrame(False, index=close_df.index, columns=[col])
    exits = pd.DataFrame(False, index=close_df.index, columns=[col])
    entries.loc[first_valid] = True
    exits.loc[last_valid] = True

    bench_pf = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries,
        exits=exits,
        init_cash=float(config.get("START_CASH", 10000.0)),
        fees=fees,
        slippage=slippage,
        freq=config.get("FREQ", "4h"),
        size=1.0,
        size_type="percent",
        cash_sharing=False,
    ).copy()

    years = _years_from_price_index(close_df.index)
    init_cash = float(config.get("START_CASH", 10000.0))
    profit_pct = float((bench_pf.total_profit().sum() / init_cash) * 100.0)
    cagr = _cagr_percent(profit_pct, years)

    try:
        _sh = bench_pf.sharpe_ratio()
        sh = float(_sh.iloc[0]) if hasattr(_sh, "iloc") else float(_sh)
    except Exception:
        sh = 0.0

    _dd = bench_pf.max_drawdown()
    _dd_f = float(_dd.iloc[0]) if hasattr(_dd, "iloc") else float(_dd)

    return {
        "profit_pct": _as_optional_float(profit_pct),
        "cagr_pct": _as_optional_float(cagr),
        "sharpe": _as_optional_float(sh),
        "max_drawdown": _as_optional_float(_dd_f * 100.0),
        "total_trades": int(bench_pf.trades.count().sum()),
    }



def _btc_buy_hold_portfolio_stats(
    close: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """BTC spot B&H: buy BTC on bar 0, sell on last bar; same vbt costs as WFO."""
    empty: Dict[str, Any] = {
        "profit_pct": None,
        "cagr_pct": None,
        "sharpe": None,
        "max_drawdown": None,
        "total_trades": 0,
    }
    if close.shape[0] < 2:
        return empty

    bench_symbol = config.get("BENCHMARK_SYMBOL", "BTC-USD")
    bench_close: Optional[pd.DataFrame] = None

    if bench_symbol in close.columns:
        bench_close = close[[bench_symbol]]
    else:
        try:
            from ggTrader.data.historical.timescaledb_loader import TimescaleDBLoader

            loader = TimescaleDBLoader()
            start = pd.to_datetime(close.index[0])
            end = pd.to_datetime(close.index[-1])
            if start.tz is None:
                start = start.tz_localize("UTC")
            if end.tz is None:
                end = end.tz_localize("UTC")

            ohlcv = loader.fetch_ohlcv(
                symbols=[bench_symbol],
                interval=config.get("INTERVAL", "4h"),
                start_date=start,
                end_date=end,
            )
            if not ohlcv.empty:
                b = ohlcv.xs("close", axis=1, level=1, drop_level=True)
                b_reindexed = b.reindex(close.index).ffill()
                if isinstance(b_reindexed, pd.Series):
                    bench_close = b_reindexed.to_frame(bench_symbol)
                else:
                    bench_close = (
                        b_reindexed[[bench_symbol]]
                        if bench_symbol in b_reindexed.columns
                        else b_reindexed.iloc[:, :1].rename(
                            columns={b_reindexed.columns[0]: bench_symbol}
                        )
                    )
        except Exception as e:
            print(f"Warning: Failed to load {bench_symbol} benchmark from DB: {e}")

    if bench_close is None or bench_close.empty:
        return empty

    stats = _buy_hold_portfolio_stats(
        bench_close,
        bench_symbol,
        config,
        fees=float(config.get("FEES", 0.001)),
        slippage=float(config.get("SLIPPAGE", 0.0005)),
    )
    stats["benchmark_symbol"] = bench_symbol
    return stats


def _enrich_final_stats_with_cagr_and_benchmark(
    final_stats: Dict[str, Any],
    combined_close: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """Add CAGR, calendar span, and equal-weight B&H benchmark fields to ``final_stats``."""
    years = _years_from_price_index(combined_close.index)
    rp = float(final_stats.get("profit_pct", 0.0) or 0.0)
    strat_cagr = _cagr_percent(rp, years)

    try:
        idx = pd.to_datetime(pd.Index(combined_close.index))
        final_stats["backtest_start"] = idx[0].strftime("%Y-%m-%d") if len(idx) else None
        final_stats["backtest_end"] = idx[-1].strftime("%Y-%m-%d") if len(idx) else None
    except (TypeError, ValueError, IndexError):
        final_stats["backtest_start"] = None
        final_stats["backtest_end"] = None

    final_stats["backtest_years"] = _as_optional_float(years)
    final_stats["cagr_pct"] = _as_optional_float(strat_cagr)

    bench = _btc_buy_hold_portfolio_stats(combined_close, config)
    bench_sym = bench.get("benchmark_symbol", "BTC-USD")
    final_stats["benchmark_label"] = (
        f"{bench_sym} buy-and-hold: bought on the first bar and sold on the "
        "last bar; same START_CASH, FEES, SLIPPAGE, and bar frequency as the strategy run."
    )
    final_stats["benchmark_profit_pct"] = bench.get("profit_pct")
    final_stats["benchmark_cagr_pct"] = bench.get("cagr_pct")
    final_stats["benchmark_sharpe"] = bench.get("sharpe")
    final_stats["benchmark_max_drawdown"] = bench.get("max_drawdown")
    final_stats["benchmark_total_trades"] = bench.get("total_trades")

