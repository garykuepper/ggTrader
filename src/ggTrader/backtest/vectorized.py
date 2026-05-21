"""Vectorized backtest engine for new-architecture Strategy classes.

Phase 3.5 changes:
- ``Pricer`` Callable removed. The engine reads ``mid_price`` from the
  feature store as a standard feature.
- Roll detection: when ``CarryUniverse.active_future`` advances at a roll
  buffer, the old future leaves the target set and a new future enters.
  The naive engine treated this as exit+entry → two fee events. Now the
  engine groups same-strategy/same-asset-class/same-direction swaps into
  a single ``RollEvent`` charging one calendar-spread fee schedule
  (typically the max of the two legs' taker bps × notional, applied once).
- Reads ``Signal.target_notional_usd`` as a typed field; fails loudly if a
  signal omits it (no metadata-string fallback).
- Optional ``TradingCalendar`` to skip closed sessions (equities, Phase 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ggTrader.core.instrument import Instrument
from ggTrader.core.signal import Direction, Signal
from ggTrader.features.base import FeatureStore
from ggTrader.strategies.base import Strategy


@dataclass
class _Position:
    instrument: Instrument
    direction: Direction
    quantity: Decimal
    entry_ts: datetime
    entry_price: Decimal
    metadata: dict[str, str] = field(default_factory=dict)

    def mark_to_market(self, price: Decimal) -> Decimal:
        sign = Decimal("1") if self.direction is Direction.LONG else Decimal("-1")
        return sign * (price - self.entry_price) * self.quantity


@dataclass
class Trade:
    ts: datetime
    instrument_symbol: str
    direction: Direction
    quantity: Decimal
    price: Decimal
    fee: Decimal
    reason: str  # "entry", "exit"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RollEvent:
    ts: datetime
    strategy_id: str
    old_symbol: str
    new_symbol: str
    direction: Direction
    old_quantity: Decimal
    new_quantity: Decimal
    old_price: Decimal
    new_price: Decimal
    spread_fee: Decimal
    realized_pnl: Decimal  # P&L from closing the old leg


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade]
    rolls: list[RollEvent]
    starting_equity: Decimal
    ending_equity: Decimal

    def metrics(self) -> dict[str, float]:
        eq = self.equity_curve.astype(float)
        if len(eq) < 2:
            return {}
        rets = eq.pct_change().dropna()
        ann_factor = 365.0

        mean_ret = float(rets.mean())
        std_ret = float(rets.std())
        downside = rets[rets < 0]
        sortino_denom = float(downside.std()) if len(downside) > 0 else 0.0

        sharpe = (mean_ret / std_ret * np.sqrt(ann_factor)) if std_ret > 0 else 0.0
        sortino = (mean_ret / sortino_denom * np.sqrt(ann_factor)) if sortino_denom > 0 else 0.0

        running_max = eq.cummax()
        drawdown = (eq - running_max) / running_max
        max_dd = float(drawdown.min())

        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0
        calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

        return {
            "starting_equity": float(self.starting_equity),
            "ending_equity": float(self.ending_equity),
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "n_trades": float(len(self.trades)),
            "n_rolls": float(len(self.rolls)),
            "years": years,
        }


def _mid_price(feature_store: FeatureStore, instrument: Instrument, ts: datetime) -> Decimal:
    """Resolve the venue mid for ``instrument`` at ``ts`` via the feature store."""
    row = feature_store.get_at("mid_price", instrument, ts)
    return Decimal(str(row[instrument.symbol]))


def _is_roll_candidate(
    closing: _Position, opening_signal: Signal, opening_strategy_id: str
) -> bool:
    """Same strategy, same base currency, same direction, both dated futures."""
    if closing.metadata.get("strategy_id") != opening_strategy_id:
        return False
    if closing.direction is not opening_signal.direction:
        return False
    if closing.instrument.base_currency != opening_signal.instrument.base_currency:
        return False
    if closing.instrument.asset_class.value != "crypto_dated_future":
        return False
    if opening_signal.instrument.asset_class.value != "crypto_dated_future":
        return False
    return True


def run_backtest(
    strategy: Strategy,
    feature_store: FeatureStore,
    start: datetime,
    end: datetime,
    starting_equity: Decimal = Decimal("100000"),
    calendar: Optional[object] = None,
    fee_bps_overrides: Optional[dict[str, Decimal]] = None,
    position_carry_fn: Optional[
        "Callable[[Instrument, Direction, Decimal, datetime, FeatureStore], Decimal]"
    ] = None,
) -> BacktestResult:
    """Run ``strategy`` from ``start`` to ``end`` and return a ``BacktestResult``."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start/end must be tz-aware UTC")

    timestamps = pd.date_range(start=start, end=end, freq="1D", tz=timezone.utc)
    open_positions: dict[str, _Position] = {}
    trades: list[Trade] = []
    rolls: list[RollEvent] = []
    realized_pnl = Decimal("0")
    equity_history: list[tuple[datetime, Decimal]] = []

    def _fee(instrument: Instrument, notional: Decimal) -> Decimal:
        bps = (fee_bps_overrides or {}).get(instrument.symbol, instrument.taker_fee_bps)
        return notional * bps / Decimal("10000")

    for ts in timestamps:
        ts_py = ts.to_pydatetime()
        if calendar is not None and not calendar.is_open(ts_py):  # type: ignore[attr-defined]
            equity_history.append(
                (ts_py, equity_history[-1][1] if equity_history else starting_equity)
            )
            continue

        signals: list[Signal] = strategy.generate_signals(ts_py, feature_store)
        target_keyed = {sig.instrument.symbol: sig for sig in signals}

        # Phase 3.5: fail loudly if a strategy emits a sizing signal without
        # target_notional_usd (the old metadata-string hack is gone).
        for sig in signals:
            if sig.target_notional_usd is None:
                raise ValueError(
                    f"Strategy {strategy.strategy_id} emitted Signal for "
                    f"{sig.instrument.symbol} without target_notional_usd. "
                    "Backtest engine cannot size this without a Sizer."
                )

        # Categorize: pure-exit positions, pure-new entries, and roll pairs.
        closing_syms = [sym for sym in open_positions if sym not in target_keyed]
        opening_syms = [sym for sym in target_keyed if sym not in open_positions]

        rolled_close: set[str] = set()
        rolled_open: set[str] = set()

        # Match closes ↔ opens within the same strategy / base / direction / class.
        for close_sym in closing_syms:
            closing = open_positions[close_sym]
            for open_sym in opening_syms:
                if open_sym in rolled_open:
                    continue
                if _is_roll_candidate(closing, target_keyed[open_sym], strategy.strategy_id):
                    rolled_close.add(close_sym)
                    rolled_open.add(open_sym)
                    roll_event, realized_delta = _execute_roll(
                        ts=ts_py,
                        feature_store=feature_store,
                        closing=closing,
                        opening_signal=target_keyed[open_sym],
                        fee_fn=_fee,
                        strategy_id=strategy.strategy_id,
                    )
                    realized_pnl += realized_delta
                    rolls.append(roll_event)
                    del open_positions[closing.instrument.symbol]
                    open_positions[target_keyed[open_sym].instrument.symbol] = _Position(
                        instrument=target_keyed[open_sym].instrument,
                        direction=target_keyed[open_sym].direction,
                        quantity=roll_event.new_quantity,
                        entry_ts=ts_py,
                        entry_price=roll_event.new_price,
                        metadata={
                            "strategy_id": strategy.strategy_id,
                            **{k: str(v) for k, v in target_keyed[open_sym].metadata.items()},
                        },
                    )
                    break

        # Pure exits (non-roll): close + book fee + book P&L
        for sym in closing_syms:
            if sym in rolled_close:
                continue
            pos = open_positions.pop(sym)
            price = _mid_price(feature_store, pos.instrument, ts_py)
            pnl = pos.mark_to_market(price)
            notional = price * pos.quantity
            fee = _fee(pos.instrument, notional)
            realized_pnl += pnl - fee
            trades.append(
                Trade(
                    ts=ts_py,
                    instrument_symbol=sym,
                    direction=pos.direction,
                    quantity=pos.quantity,
                    price=price,
                    fee=fee,
                    reason="exit",
                )
            )

        # Pure entries (non-roll)
        for sym in opening_syms:
            if sym in rolled_open:
                continue
            sig = target_keyed[sym]
            price = _mid_price(feature_store, sig.instrument, ts_py)
            notional_target = sig.target_notional_usd
            if notional_target is None or notional_target <= 0 or price <= 0:
                continue
            quantity = notional_target / price
            fee = _fee(sig.instrument, notional_target)
            realized_pnl -= fee
            open_positions[sym] = _Position(
                instrument=sig.instrument,
                direction=sig.direction,
                quantity=quantity,
                entry_ts=ts_py,
                entry_price=price,
                metadata={
                    "strategy_id": strategy.strategy_id,
                    **{k: str(v) for k, v in sig.metadata.items()},
                },
            )
            trades.append(
                Trade(
                    ts=ts_py,
                    instrument_symbol=sym,
                    direction=sig.direction,
                    quantity=quantity,
                    price=price,
                    fee=fee,
                    reason="entry",
                    metadata=dict(sig.metadata),
                )
            )

        # Position carry (funding/borrow) accrual — book once per bar
        if position_carry_fn is not None:
            for pos in open_positions.values():
                try:
                    price = _mid_price(feature_store, pos.instrument, ts_py)
                except (LookupError, KeyError):
                    continue
                notional = price * pos.quantity
                try:
                    carry = position_carry_fn(
                        pos.instrument, pos.direction, notional, ts_py, feature_store
                    )
                except LookupError:
                    carry = Decimal("0")
                realized_pnl += carry

        # Mark-to-market
        unrealized = Decimal("0")
        for pos in open_positions.values():
            try:
                price = _mid_price(feature_store, pos.instrument, ts_py)
            except (LookupError, KeyError):
                continue
            unrealized += pos.mark_to_market(price)
        equity_history.append((ts_py, starting_equity + realized_pnl + unrealized))

    equity_index = pd.DatetimeIndex([row[0] for row in equity_history])
    equity_values = pd.Series(
        [float(row[1]) for row in equity_history], index=equity_index, name="equity"
    )

    return BacktestResult(
        equity_curve=equity_values,
        trades=trades,
        rolls=rolls,
        starting_equity=starting_equity,
        ending_equity=Decimal(str(float(equity_values.iloc[-1]))),
    )


def _execute_roll(
    ts: datetime,
    feature_store: FeatureStore,
    closing: _Position,
    opening_signal: Signal,
    fee_fn: "Callable[[Instrument, Decimal], Decimal]",
    strategy_id: str,
) -> tuple[RollEvent, Decimal]:
    """Compute the close P&L, the new-leg size, and the single calendar-spread fee.

    Real exchanges charge a single fee schedule (or a calendar-spread discount).
    We approximate with one fee on the *larger* of the two legs' notionals at
    the new contract's taker rate. Strictly better than charging twice.

    Returns: (RollEvent describing the transition, realized P&L delta to book).
    """
    old_inst = closing.instrument
    new_inst = opening_signal.instrument
    old_price = _mid_price(feature_store, old_inst, ts)
    new_price = _mid_price(feature_store, new_inst, ts)

    pnl_close = closing.mark_to_market(old_price)

    notional_target = opening_signal.target_notional_usd
    if notional_target is None or notional_target <= 0:
        raise ValueError(f"Roll into {new_inst.symbol} requires target_notional_usd on the signal")
    new_quantity = notional_target / new_price

    notional_old = old_price * closing.quantity
    notional_new = new_price * new_quantity
    spread_fee = fee_fn(new_inst, max(notional_old, notional_new))

    event = RollEvent(
        ts=ts,
        strategy_id=strategy_id,
        old_symbol=old_inst.symbol,
        new_symbol=new_inst.symbol,
        direction=closing.direction,
        old_quantity=closing.quantity,
        new_quantity=new_quantity,
        old_price=old_price,
        new_price=new_price,
        spread_fee=spread_fee,
        realized_pnl=pnl_close,
    )
    return event, pnl_close - spread_fee


__all__ = ["BacktestResult", "RollEvent", "Trade", "run_backtest"]
