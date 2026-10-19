"""Event-loop replica of the live paper trader's portfolio rules, for parity backtests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ggTrader.paper.cash_sweep import (
    DEFAULT_BUY_TRIGGER_PCT,
    DEFAULT_MIN_CLIP_USD,
    DEFAULT_RESERVE_PCT,
    SWEEP_TRADE_REASON,
    compute_sweep_buy,
    compute_sweep_sell_for_funding,
)
from ggTrader.paper.catastrophe_stop import (
    CATASTROPHE_STOP_REASON,
    DEFAULT_CATASTROPHE_STOP_PCT,
    find_catastrophe_stops,
)
from ggTrader.paper.risk import RiskConfig, RiskGuard

REASON_SIGNAL = "signal"
REASON_DELISTED = "delisted"
TRADE_COLUMNS = ["date", "symbol", "side", "notional", "price", "reason"]


@dataclass(frozen=True)
class ReplicaConfig:
    """Pinned portfolio settings, independent of live environment flags."""

    start_cash: float = 100_000.0
    risk: RiskConfig = RiskConfig(max_positions=3)
    reserve_pct: float = DEFAULT_RESERVE_PCT
    buy_trigger_pct: float = DEFAULT_BUY_TRIGGER_PCT
    min_clip_usd: float = DEFAULT_MIN_CLIP_USD
    catastrophe_stop_pct: float = DEFAULT_CATASTROPHE_STOP_PCT
    stock_cost_bps: float = 5.0
    spy_cost_bps: float = 1.0
    signal_lag_bars: int = 1


@dataclass
class ReplicaResult:
    """Daily NAV, fills, exposures and blocked-decision counts."""

    equity: pd.Series
    trades: pd.DataFrame
    exposure: pd.DataFrame
    halted_days: int
    truncated_days: int


def run_replica(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    spy_close: pd.Series,
    cfg: ReplicaConfig = ReplicaConfig(),
    cash_rate_daily: pd.Series | None = None,
) -> ReplicaResult:
    """Replay lagged signals through the live pure risk and sweep decisions."""
    if close.empty or close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise ValueError("Replica needs a nonempty, sorted, unique close index")
    if "SPY" in close.columns or close.columns.has_duplicates:
        raise ValueError("Stock symbols must be unique and exclude the SPY sweep")
    for frame in (entries, exits):
        if not frame.index.equals(close.index) or not frame.columns.equals(close.columns):
            raise ValueError("Replica signals must align exactly to stock closes")
        if frame.isna().any().any():
            raise ValueError("Replica signals cannot contain missing values")
    if cfg.start_cash <= 0 or cfg.signal_lag_bars < 0:
        raise ValueError("Replica needs positive start cash and a nonnegative signal lag")
    if not 0 <= cfg.stock_cost_bps < 10_000 or not 0 <= cfg.spy_cost_bps < 10_000:
        raise ValueError("Replica costs must be between 0 and 10000 bps")
    index, symbols = close.index, list(close.columns)
    prices = close.to_numpy(dtype=float)
    if np.isinf(prices).any() or (prices[np.isfinite(prices)] <= 0).any():
        raise ValueError("Stock closes must be positive or missing")
    marks = close.ffill().to_numpy(dtype=float)
    spy = spy_close.reindex(index).to_numpy(dtype=float)
    if np.isinf(spy).any() or (spy[np.isfinite(spy)] <= 0).any():
        raise ValueError("SPY closes must be positive or missing")
    spy_marks = pd.Series(spy, index=index).ffill().to_numpy()
    rates = (
        np.zeros(len(index))
        if cash_rate_daily is None
        else cash_rate_daily.reindex(index).to_numpy(dtype=float)
    )
    if not np.isfinite(rates).all() or (rates <= -1).any():
        raise ValueError("Cash rates must be finite and greater than -100% on every bar")
    last_valid = np.where(np.isfinite(prices), np.arange(len(index))[:, None], -1).max(axis=0)
    entry_values = entries.to_numpy(dtype=bool)
    exit_values = exits.to_numpy(dtype=bool)
    exit_counts = exit_values.cumsum(axis=0)
    symbol_cols = {symbol: col for col, symbol in enumerate(symbols)}
    positions: dict[str, dict] = {}
    cash = cfg.start_cash
    spy_shares = 0.0
    guard = RiskGuard(cfg.risk)
    guard.update_peak(cfg.start_cash)
    equity = np.full(len(index), cfg.start_cash)
    exposure = np.zeros((len(index), 3))
    exposure[:, 2] = 1.0
    trades: list[dict] = []
    halted_days = truncated_days = 0

    def sell_stock(symbol: str, bar: int, reason: str) -> None:
        """Liquidate at the marked close, including sell-side costs."""
        nonlocal cash
        position = positions.pop(symbol)
        price = marks[bar, symbol_cols[symbol]] * (1 - cfg.stock_cost_bps / 1e4)
        proceeds = position["shares"] * price
        cash += proceeds
        trades.append(
            dict(
                date=index[bar],
                symbol=symbol,
                side="sell",
                notional=proceeds,
                price=price,
                reason=reason,
            )
        )

    def record(bar: int) -> None:
        """Mark NAV after fills and retain the cash conservation check."""
        stock_value = sum(p["shares"] * marks[bar, symbol_cols[s]] for s, p in positions.items())
        spy_value = spy_shares * spy_marks[bar] if spy_shares else 0.0
        nav = cash + stock_value + spy_value
        if cash < 0 or not np.isfinite(nav) or nav <= 0:
            raise ValueError(f"Invalid replica NAV or cash on {index[bar]}")
        equity[bar] = nav
        exposure[bar] = np.array([stock_value, spy_value, cash]) / nav

    for i in range(cfg.signal_lag_bars, len(index)):
        s = i - cfg.signal_lag_bars
        cash *= 1 + rates[i]
        for symbol in list(positions):
            if last_valid[symbol_cols[symbol]] < i:
                sell_stock(symbol, i, REASON_DELISTED)
        record(i)
        nav = equity[i]
        prev_nav = equity[i - 1] if i else cfg.start_cash
        halted = guard.check_drawdown_halt(nav)[0] or guard.check_daily_loss(nav, prev_nav)[0]
        guard.update_peak(nav)
        if halted:
            halted_days += 1
            continue

        live_positions = {
            symbol: {
                "cost_basis": p["cost"],
                "unrealized_pl": p["shares"] * marks[i, symbol_cols[symbol]] - p["cost"],
            }
            for symbol, p in positions.items()
        }
        stopped = find_catastrophe_stops(live_positions, cfg.catastrophe_stop_pct)
        for symbol in stopped:
            sell_stock(symbol, i, CATASTROPHE_STOP_REASON)
        for symbol in list(positions):
            col = symbol_cols[symbol]
            fill_i = positions[symbol]["fill_i"]
            catch_up = s > fill_i and exit_counts[s, col] > exit_counts[fill_i, col]
            if exit_values[s, col] or catch_up:
                sell_stock(symbol, i, REASON_SIGNAL)

        candidates = sorted(
            symbol
            for col, symbol in enumerate(symbols)
            if entry_values[s, col]
            and symbol not in positions
            and symbol not in stopped
            and np.isfinite(prices[i, col])
        )
        slots = guard.max_new_positions(len(positions))
        truncated_days += len(candidates) > slots
        notional = guard.position_notional(nav)
        buys = [
            symbol
            for symbol in candidates[:slots]
            if not guard.check_concentration(symbol, {}, nav, notional)
        ]
        spy_value = spy_shares * spy_marks[i] if spy_shares else 0.0
        if np.isfinite(spy[i]):
            funding = compute_sweep_sell_for_funding(
                cash, nav, notional * len(buys), spy_value, cfg.reserve_pct
            )
            if funding.side == "sell":
                shares = min(spy_shares, funding.notional / spy[i])
                price = spy[i] * (1 - cfg.spy_cost_bps / 1e4)
                proceeds = shares * price
                cash += proceeds
                spy_shares -= shares
                trades.append(
                    dict(
                        date=index[i],
                        symbol="SPY",
                        side="sell",
                        notional=proceeds,
                        price=price,
                        reason=SWEEP_TRADE_REASON,
                    )
                )
        for symbol in buys:
            if notional <= 0 or notional > cash:
                continue
            price = prices[i, symbol_cols[symbol]] * (1 + cfg.stock_cost_bps / 1e4)
            positions[symbol] = dict(
                shares=notional / price, cost=notional, fill_i=i, fill_date=index[i]
            )
            cash -= notional
            trades.append(
                dict(
                    date=index[i],
                    symbol=symbol,
                    side="buy",
                    notional=notional,
                    price=price,
                    reason=REASON_SIGNAL,
                )
            )
        if np.isfinite(spy[i]):
            sweep = compute_sweep_buy(
                cash, nav, cfg.reserve_pct, cfg.min_clip_usd, cfg.buy_trigger_pct
            )
            if sweep.side == "buy" and sweep.notional <= cash:
                price = spy[i] * (1 + cfg.spy_cost_bps / 1e4)
                spy_shares += sweep.notional / price
                cash -= sweep.notional
                trades.append(
                    dict(
                        date=index[i],
                        symbol="SPY",
                        side="buy",
                        notional=sweep.notional,
                        price=price,
                        reason=SWEEP_TRADE_REASON,
                    )
                )
        record(i)
    return ReplicaResult(
        equity=pd.Series(equity, index=index, name="equity"),
        trades=pd.DataFrame(trades, columns=TRADE_COLUMNS),
        exposure=pd.DataFrame(exposure, index=index, columns=["stocks", "spy", "cash"]),
        halted_days=int(halted_days),
        truncated_days=int(truncated_days),
    )
