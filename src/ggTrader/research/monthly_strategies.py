"""Pluggable monthly strategies for the honest walk-forward harness.

A MonthlyStrategy maps (data <= T, point-in-time eligible universe) to
next-month selections, then simulates the forward month with those frozen
selections. The select/simulate split is what makes the generic leak check
possible: ``select`` must be a pure function of data <= T.

The harness (research/monthly_walkforward.py) guarantees ``select`` only ever
receives data truncated to <= asof.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.core.orchestrator_utils import _to_native
from ggTrader.research.equity_wfo import grid_books, wfo_strategy_tournament_one_stock


class MonthlyStrategy(Protocol):
    """Contract for a strategy runnable by run_monthly_walkforward."""

    name: str

    def select(
        self, asof: pd.Timestamp, ohlcv: pd.DataFrame, eligible: List[str]
    ) -> List[Dict[str, Any]]:
        """JSON-able selection records (each with at least "symbol"); data <= asof."""
        ...

    def simulate(
        self,
        ohlcv: pd.DataFrame,
        selections: List[Dict[str, Any]],
        asof: pd.Timestamp,
        month_end: pd.Timestamp,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """Daily portfolio returns for (asof, month_end] plus diagnostics."""
        ...


def _portfolio_exposure(pf) -> pd.Series:
    """Fraction of capital deployed per bar: 1 - cash/value (grouped portfolio)."""
    cash, value = pf.cash(), pf.value()
    if isinstance(cash, pd.DataFrame):
        cash = cash.iloc[:, 0]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return 1.0 - cash / value


class CrossSectionalMomentum:
    """12-1 cross-sectional momentum: top-N by trailing return, equal weight."""

    name = "xs_momentum"

    def __init__(
        self,
        cfg,
        base_config: Dict[str, Any],
        lookback: int = 252,
        skip: int = 21,
    ) -> None:
        self.cfg = cfg
        self.base_config = base_config
        self.lookback = lookback
        self.skip = skip

    def select(
        self, asof: pd.Timestamp, ohlcv: pd.DataFrame, eligible: List[str]
    ) -> List[Dict[str, Any]]:
        ohlcv = ohlcv.loc[:asof]  # defense in depth: invariant to post-asof rows
        scores: Dict[str, float] = {}
        for sym in eligible:
            closes = ohlcv[sym]["close"].dropna()
            if len(closes) < self.lookback + 1:
                continue
            past = float(closes.iloc[-(self.lookback + 1)])
            recent = float(closes.iloc[-(self.skip + 1)])
            if past <= 0.0 or not np.isfinite(past) or not np.isfinite(recent):
                continue
            scores[sym] = recent / past - 1.0
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self.cfg.top_n]
        if not ranked:
            return []
        weight = 1.0 / len(ranked)
        return [{"symbol": s, "weight": weight, "momentum": m} for s, m in ranked]

    def simulate(
        self,
        ohlcv: pd.DataFrame,
        selections: List[Dict[str, Any]],
        asof: pd.Timestamp,
        month_end: pd.Timestamp,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        weights = {s["symbol"]: float(s["weight"]) for s in selections}
        return simulate_hold_weights(ohlcv, weights, asof, month_end, self.base_config)


def simulate_hold_weights(
    ohlcv: pd.DataFrame,
    weights: Dict[str, float],
    asof: pd.Timestamp,
    month_end: pd.Timestamp,
    base_config: Dict[str, Any],
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Buy target weights at the first bar after ``asof``, hold to ``month_end``.

    Symbols with no price at the first forward bar are dropped (their weight
    stays in cash). Mid-month gaps are forward-filled.
    """
    empty = pd.Series(dtype=float)
    month_mask = (ohlcv.index > asof) & (ohlcv.index <= month_end)
    if not month_mask.any() or not weights:
        return empty, {"n_positions": 0, "n_trades": 0, "avg_exposure": 0.0}

    have = set(ohlcv.columns.get_level_values(0))
    close = (
        pd.concat({s: ohlcv[s]["close"] for s in weights if s in have}, axis=1)
        .loc[month_mask]
        .ffill()
    )
    close = close.dropna(axis=1)  # NaN after ffill == no price at month start
    if close.shape[1] == 0:
        return empty, {"n_positions": 0, "n_trades": 0, "avg_exposure": 0.0}

    size = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    size.iloc[0] = [weights[s] for s in close.columns]
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=size,
        size_type="targetpercent",
        init_cash=float(base_config["START_CASH"]),
        fees=float(base_config["FEES"]),
        slippage=float(base_config["SLIPPAGE"]),
        freq=base_config["FREQ"],
        cash_sharing=True,
        group_by=np.full(close.shape[1], 0),
        call_seq="auto",
    ).copy()

    returns = pf.returns()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    diags = {
        "n_positions": int(close.shape[1]),
        "n_trades": int(pf.trades.count().sum()),
        "avg_exposure": float(_portfolio_exposure(pf).mean()),
        "month_return_pct": float((1.0 + returns).prod() - 1.0) * 100,
    }
    return returns, diags


class DualMomentum(CrossSectionalMomentum):
    """Cross-sectional momentum + absolute filter: negative-momentum picks go to cash.

    Weights are NOT renormalized — a dropped pick's 1/N slot stays in cash,
    so the portfolio de-risks as breadth deteriorates.
    """

    name = "dual_momentum"

    def select(
        self, asof: pd.Timestamp, ohlcv: pd.DataFrame, eligible: List[str]
    ) -> List[Dict[str, Any]]:
        picks = super().select(asof, ohlcv, eligible)
        return [p for p in picks if p["momentum"] >= 0.0]


def _select_worker(args: Tuple) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Tournament for one stock on its trailing window (process-pool worker)."""
    (
        symbol,
        sym_ohlcv,
        base_config,
        entries,
        exits,
        entry_book,
        exit_book,
        n_splits,
        test_ratio,
    ) = args
    try:
        res = wfo_strategy_tournament_one_stock(
            sym_ohlcv, base_config, entries, exits, entry_book, exit_book, n_splits, test_ratio
        )
    except Exception as exc:
        print(f"    [{symbol}] tournament failed: {exc!r}")
        return symbol, None
    best = res.get("best")
    if best is None or not np.isfinite(best.get("oos_robustness", float("nan"))):
        return symbol, None

    avg_hold = float("nan")
    try:
        replay_cfg = {
            **base_config,
            "ENTRY_STRATEGY": best["entry"],
            "EXIT_STRATEGY": best["exit"],
        }
        engine = FastBacktest(sym_ohlcv, best["params"], config=replay_cfg)
        engine.run(show_progress=False)
        avg_hold = engine.get_stats().get("avg_holding_days", float("nan"))
    except Exception:
        pass

    return symbol, {
        "symbol": symbol,
        "entry": best["entry"],
        "exit": best["exit"],
        "params": _to_native(best["params"]),
        "oos_robustness": float(best["oos_robustness"]),
        "fold_consistency": float(best["fold_consistency"]),
        "is_robustness": float(best["is_robustness"]),
        "avg_holding_days": float(avg_hold) if np.isfinite(avg_hold) else None,
    }


class WfoTournamentStrategy:
    """Per-stock entry x exit WFO tournament; top-N by OOS robustness."""

    name = "wfo_tournament"

    def __init__(
        self,
        cfg,
        base_config: Dict[str, Any],
        entry_book: Optional[Dict[str, Dict[str, Any]]] = None,
        exit_book: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if entry_book is None or exit_book is None:
            entry_book, exit_book = grid_books(cfg.grid_book)
        self.cfg = cfg
        self.base_config = base_config
        self.entry_book = entry_book
        self.exit_book = exit_book

    def select(
        self, asof: pd.Timestamp, ohlcv: pd.DataFrame, eligible: List[str]
    ) -> List[Dict[str, Any]]:
        cfg = self.cfg
        ohlcv = ohlcv.loc[:asof]  # defense in depth: invariant to post-asof rows
        jobs = [
            (
                sym,
                ohlcv[[sym]].tail(cfg.lookback_bars),
                self.base_config,
                cfg.entries,
                cfg.exits,
                self.entry_book,
                self.exit_book,
                cfg.n_splits,
                cfg.test_ratio,
            )
            for sym in eligible
        ]
        results: List[Dict[str, Any]] = []
        if cfg.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=cfg.n_jobs) as pool:
                for _sym, rec in pool.map(_select_worker, jobs, chunksize=1):
                    if rec is not None:
                        results.append(rec)
        else:
            for job in jobs:
                _sym, rec = _select_worker(job)
                if rec is not None:
                    results.append(rec)
        results.sort(key=lambda r: r["oos_robustness"], reverse=True)
        return results[: cfg.top_n]

    def simulate(
        self,
        ohlcv: pd.DataFrame,
        selections: List[Dict[str, Any]],
        asof: pd.Timestamp,
        month_end: pd.Timestamp,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        cfg, base_config = self.cfg, self.base_config
        empty = pd.Series(dtype=float)
        month_index = ohlcv.index[(ohlcv.index > asof) & (ohlcv.index <= month_end)]
        if len(month_index) == 0 or not selections:
            return empty, {"n_positions": 0, "n_trades": 0, "avg_exposure": 0.0}

        all_entries, all_exits, all_close = [], [], []
        for sel in selections:
            sym = sel["symbol"]
            if sym not in ohlcv.columns.get_level_values(0):
                continue
            window = ohlcv[[sym]].loc[:month_end].tail(cfg.warmup_bars + len(month_index))
            sig_cfg = {
                **base_config,
                "ENTRY_STRATEGY": sel["entry"],
                "EXIT_STRATEGY": sel["exit"],
            }
            try:
                engine = FastBacktest(window, sel["params"], config=sig_cfg)
                engine.run(show_progress=False)
            except Exception as exc:
                print(f"    [{sym}] forward signal generation failed: {exc!r}")
                continue
            entries = engine.entries.droplevel("param_combo", axis=1)
            exits = engine.exits.droplevel("param_combo", axis=1)
            entries.loc[entries.index <= asof] = False  # no pre-month positions
            all_entries.append(entries)
            all_exits.append(exits)
            all_close.append(window.xs("close", axis=1, level=1, drop_level=True))

        if not all_entries:
            return empty, {"n_positions": 0, "n_trades": 0, "avg_exposure": 0.0}

        entries_df = pd.concat(all_entries, axis=1).fillna(False)
        exits_df = pd.concat(all_exits, axis=1).fillna(False)
        close_df = pd.concat(all_close, axis=1)

        pf = vbt.Portfolio.from_signals(
            close=close_df,
            entries=entries_df,
            exits=exits_df,
            init_cash=float(base_config["START_CASH"]),
            fees=float(base_config["FEES"]),
            slippage=float(base_config["SLIPPAGE"]),
            freq=base_config["FREQ"],
            size=cfg.max_position_pct,
            size_type="percent",
            cash_sharing=True,
            group_by=np.full(entries_df.shape[1], 0),
        ).copy()

        returns = pf.returns()
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        in_month = (returns.index > asof) & (returns.index <= month_end)
        month_returns = returns.loc[in_month]
        exposure = _portfolio_exposure(pf).loc[in_month]

        dur = np.array(pf.trades.duration.values, dtype=np.float64, copy=True)
        diags = {
            "n_positions": int(len(selections)),
            "n_trades": int(pf.trades.count().sum()),
            "avg_holding_days": float(dur.mean()) if dur.size else None,
            "avg_exposure": float(exposure.mean()) if len(exposure) else 0.0,
            "month_return_pct": float((1.0 + month_returns).prod() - 1.0) * 100,
        }
        return month_returns, diags


STRATEGY_NAMES = ("wfo_tournament", "xs_momentum", "dual_momentum")


def build_strategy(
    name: str,
    cfg,
    base_config: Dict[str, Any],
    mom_lookback: int = 252,
    mom_skip: int = 21,
) -> "MonthlyStrategy":
    if name == "wfo_tournament":
        return WfoTournamentStrategy(cfg, base_config)
    if name == "xs_momentum":
        return CrossSectionalMomentum(cfg, base_config, mom_lookback, mom_skip)
    if name == "dual_momentum":
        return DualMomentum(cfg, base_config, mom_lookback, mom_skip)
    raise ValueError(f"Unknown strategy {name!r}. Available: {STRATEGY_NAMES}")
