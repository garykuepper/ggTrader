"""Tax-ledger execution of the monthly replay (and its SPY twin) through ``TaxLotBook``.

The planner (``replay.plan_trades``) says what to hold; this module trades it in a real
tax-lot book with costs, year-end tax and dead-stock exits, and reports the result.

``TaxLotBook`` is loaded by file path from ``research/paper-system/tax-lots/code`` (never
copied, never imported as a package): the study conventions forbid importing another study's
``code/`` by name, and the book needs its module registered in ``sys.modules`` for its
dataclasses.

Order of events on each trading day of the window:

1. Dead exits and successor exits due that day: every held stock whose series ends on this
   bar is sold on that last bar at ``exit_close``'s price (written down by the dead-stock loss
   for a dead stock, unchanged for a bought-out or merged one). A stock whose series has ended
   is never bought again; such a buy is skipped.
2. Tax: on the first trading day of a new year the old years are assessed (``year_end``) and
   paid from the book's own cash (requirements R9: no outside top-up, no tax reserve). When cash
   is short, positions are first sold pro rata, through the book and with cost, enough to cover
   the bill, at most three rounds (a sale can change the assessment); anything still unpaid is
   reported as ``unpaid_tax_left`` and stays reserved from spending.
3. Rebalance on that day's ``TargetStep``. Sizing equity is ``marks + cash - unpaid_tax`` (the
   tax bill is not spendable); target shares are ``weight * equity / price``. All sells go
   before all buys; buys are scaled down proportionally when cash (less the unpaid tax) is
   short, so cash never goes negative. A target stock with no close that day (halted, or ended)
   is not bought, its cash is kept and ``skipped_buys`` counts it; no other stock replaces it.
   A held stock with no close that day is marked at its last print and cannot be traded.

On the last day of the window everything is sold at the day's close (at its last print if the
stock has no bar), the final year is assessed (R12) and the tax paid from cash; the rebalance of
that day is dropped because it would only buy what is sold the same minute. ``TaxLotBook``
applies the wash-sale rule itself, so a loss sold and re-bought within thirty days is deferred.
Every buy fills at ``price * (1 + c)`` and every sell at ``price * (1 - c)``, so the cost is in
the lot basis and the proceeds; ``Trade.price`` is the price before cost. The book's loss
carry-forward and no-outside-income rules are taken as they are (R15).

Pre-tax equity on a day is the marks plus book cash plus all tax paid so far; the after-tax
final value is the book's cash after the last liquidation and payment.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from replay import (
    DEAD_EXIT_REASON,
    LIQUIDATION_REASON,
    REBALANCE_REASON,
    TAX_SALE_REASON,
    TargetStep,
    Trade,
)

from ggTrader.utils.study import repo_root

TAX_LOTS_PATH = Path("research/paper-system/tax-lots/code/tax_lots.py")
TAX_LOTS_MODULE = "tax_lots_dead_stock_first_test"
MAX_TAX_SALE_ROUNDS = 3  # a pro-rata sale can change the assessment, so repeat a few times
MIN_TRADE_DOLLARS = 1e-6  # smaller trades are float dust, not decisions
TAX_SALE_MARGIN = 1e-9  # sell this much extra (relative) so rounding cannot leave a sliver unpaid
DAYS_PER_YEAR = 365.25
__all__ = ["BookResult", "TaxLotBook", "TaxParams", "run_book", "run_spy"]


def _load_tax_lots() -> ModuleType:
    """Load ``tax_lots.py`` by file path and register it under a private module name."""
    if TAX_LOTS_MODULE in sys.modules:
        return sys.modules[TAX_LOTS_MODULE]
    path = repo_root(__file__) / TAX_LOTS_PATH
    spec = importlib.util.spec_from_file_location(TAX_LOTS_MODULE, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the tax-lot library from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[TAX_LOTS_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(TAX_LOTS_MODULE, None)
        raise
    return module


_TAX_LOTS = _load_tax_lots()
TaxLotBook = _TAX_LOTS.TaxLotBook
TaxParams = _TAX_LOTS.TaxParams


@dataclass(frozen=True)
class BookResult:
    """What one run through the tax book produced.

    Attributes:
        pretax_equity: Marks plus cash plus tax paid so far, one value per trading day of the
            window.
        after_tax_final: Book cash after the last liquidation and the final tax payment.
        tax_paid: All tax paid out of the book.
        cost_paid: All trading cost, dead exits included (``qty * price * c`` per fill).
        n_dead_exits: Held stocks sold at the end of their series that are not successors.
        dead_exit_loss_realised: Dollars written off by the dead-stock loss on those exits
            (quantity times last close minus sale price, before cost).
        cagr_pretax: Yearly rate from the start cash to the last pre-tax equity (calendar days
            over 365.25).
        cagr_after_tax: The same from the start cash to ``after_tax_final``.
        trades: Every fill, in order, at its price before cost.
        unpaid_tax_left: Assessed tax the book could not pay even after selling everything.
        skipped_buys: Target stocks that were not held and could not be bought (no close that
            day, or the series had ended).
        chain_verified: ``book.verify_chain()`` after the run.
        min_cash: Lowest book cash seen after any fill or payment (never below zero).
    """

    pretax_equity: pd.Series
    after_tax_final: float
    tax_paid: float
    cost_paid: float
    n_dead_exits: int
    dead_exit_loss_realised: float
    cagr_pretax: float
    cagr_after_tax: float
    trades: list[Trade]
    unpaid_tax_left: float = 0.0
    skipped_buys: int = 0
    chain_verified: bool = True
    min_cash: float = 0.0


def _naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(None) if index.tz is not None else index


def _naive_stamp(stamp: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(stamp)
    return stamp.tz_convert(None) if stamp.tzinfo is not None else stamp


def _cagr(start_value: float, end_value: float, days: int) -> float:
    """Annualised rate over ``days`` calendar days; a wiped-out account is -100 percent."""
    if end_value <= 0.0:
        return -1.0
    if days <= 0:
        return 0.0
    return float((end_value / start_value) ** (DAYS_PER_YEAR / days) - 1.0)


class _Execution:
    """Mutable state of one run: the book, the shares held and the running totals."""

    def __init__(
        self,
        columns: pd.Index,
        params: object,
        one_way_cost: float,
        start_cash: float,
        successors: Collection[str],
        ended: dict[str, pd.Timestamp],
    ) -> None:
        self.pos = {str(s): j for j, s in enumerate(columns)}
        self.symbols = [str(s) for s in columns]
        self.cost = one_way_cost
        self.book = TaxLotBook(params, cash=start_cash)
        self.qty: dict[str, float] = {}
        self.trades: list[Trade] = []
        self.cost_paid = 0.0
        self.tax_paid = 0.0
        self.n_dead = 0
        self.dead_loss = 0.0
        self.skipped = 0
        self.min_cash = start_cash
        self.successors = frozenset(successors)
        self.ended = ended

    def _note_cash(self) -> None:
        self.min_cash = min(self.min_cash, self.book.cash)

    def sell(self, symbol: str, qty: float, price: float, date: pd.Timestamp, reason: str) -> None:
        """Sell ``qty`` shares at ``price`` less the cost; a sale of all shares is exact."""
        held = self.qty[symbol]
        qty = held if qty >= held * (1.0 - 1e-12) else qty
        self.book.sell(symbol, qty, price * (1.0 - self.cost), date)
        self.cost_paid += qty * price * self.cost
        self.trades.append(Trade(date, symbol, -qty, price, reason))  # type: ignore[arg-type]
        if qty >= held:
            del self.qty[symbol]
        else:
            self.qty[symbol] = held - qty
        self._note_cash()

    def buy(self, symbol: str, qty: float, price: float, date: pd.Timestamp) -> None:
        """Buy ``qty`` shares at ``price`` plus the cost."""
        self.book.buy(symbol, qty, price * (1.0 + self.cost), date)
        self.cost_paid += qty * price * self.cost
        self.trades.append(Trade(date, symbol, qty, price, REBALANCE_REASON))
        self.qty[symbol] = self.qty.get(symbol, 0.0) + qty
        self._note_cash()

    def marks(self, mark_row: np.ndarray) -> float:
        """Value of the held shares at the (forward-filled) closes."""
        return float(sum(q * mark_row[self.pos[s]] for s, q in self.qty.items()))

    def dead_exit(
        self, symbol: str, raw_row: np.ndarray, exit_row: np.ndarray, date: pd.Timestamp
    ) -> None:
        """Sell a held stock on the last bar of its series at the exit price."""
        if symbol not in self.qty:
            return
        j = self.pos[symbol]
        last = float(raw_row[j])
        price = float(exit_row[j]) if math.isfinite(exit_row[j]) else last
        qty = self.qty[symbol]
        self.sell(symbol, qty, price, date, DEAD_EXIT_REASON)
        if symbol not in self.successors:
            self.n_dead += 1
            self.dead_loss += qty * (last - price)

    def pay(self, amount: float, date: pd.Timestamp) -> None:
        if amount > 0.0:
            self.book.pay_tax(amount, date)
            self.tax_paid += amount
            self._note_cash()

    def settle_tax(self, date: pd.Timestamp, raw_row: np.ndarray) -> None:
        """Pay the unpaid tax from cash, selling pro rata first when cash is short."""
        for _ in range(MAX_TAX_SALE_ROUNDS):
            need = self.book.unpaid_tax
            if need <= MIN_TRADE_DOLLARS * 1e-3:
                return
            if self.book.cash >= need:
                self.pay(need, date)
                return
            if not self._sell_pro_rata(need - self.book.cash, date, raw_row):
                break
        self.pay(min(self.book.cash, self.book.unpaid_tax), date)

    def _sell_pro_rata(self, shortfall: float, date: pd.Timestamp, raw_row: np.ndarray) -> bool:
        """Sell the same fraction of every tradable position; False if nothing could be sold."""
        tradable = {s: q for s, q in self.qty.items() if math.isfinite(raw_row[self.pos[s]])}
        net = sum(q * raw_row[self.pos[s]] * (1.0 - self.cost) for s, q in tradable.items())
        if net <= 0.0:
            return False
        fraction = min(1.0, shortfall / net * (1.0 + TAX_SALE_MARGIN))
        for symbol, qty in tradable.items():
            self.sell(
                symbol, qty * fraction, float(raw_row[self.pos[symbol]]), date, TAX_SALE_REASON
            )
        return True

    def rebalance(
        self,
        targets: dict[str, float],
        date: pd.Timestamp,
        raw_row: np.ndarray,
        mark_row: np.ndarray,
    ) -> None:
        """Move the book to the target weights: sells first, then scaled buys."""
        equity = max(0.0, self.marks(mark_row) + self.book.cash - self.book.unpaid_tax)
        for symbol in list(self.qty):
            price = raw_row[self.pos[symbol]]
            if not math.isfinite(price):
                continue  # halted: cannot trade it today
            held = self.qty[symbol]
            want = targets.get(symbol, 0.0) * equity / price
            if held - want > 0.0 and (held - want) * price > MIN_TRADE_DOLLARS:
                self.sell(symbol, held - want, float(price), date, REBALANCE_REASON)
        wanted: list[tuple[str, float, float]] = []
        for symbol, weight in targets.items():
            j = self.pos.get(symbol)
            price = raw_row[j] if j is not None else math.nan
            gone = symbol in self.ended and self.ended[symbol] <= date
            if j is None or not math.isfinite(price) or gone:
                if symbol not in self.qty:
                    self.skipped += 1
                continue
            add = weight * equity / price - self.qty.get(symbol, 0.0)
            if add * price > MIN_TRADE_DOLLARS:
                wanted.append((symbol, add, float(price)))
        needed = sum(q * p * (1.0 + self.cost) for _, q, p in wanted)
        available = max(0.0, self.book.cash - self.book.unpaid_tax)
        scale = min(1.0, available / needed) if needed > 0.0 else 1.0
        for symbol, add, price in wanted:
            if add * scale * price > MIN_TRADE_DOLLARS:
                self.buy(symbol, add * scale, price, date)

    def liquidate(self, date: pd.Timestamp, mark_row: np.ndarray) -> None:
        """Sell everything at the day's close (last print if no bar), assess and pay."""
        for symbol in list(self.qty):
            self.sell(
                symbol,
                self.qty[symbol],
                float(mark_row[self.pos[symbol]]),
                date,
                LIQUIDATION_REASON,
            )
        self.book.year_end(date.year, date)
        self.pay(min(self.book.cash, self.book.unpaid_tax), date)


def run_book(
    close: pd.DataFrame,
    steps: list[TargetStep],
    dead_exits: pd.Series,
    exit_close: pd.DataFrame,
    *,
    params: TaxParams,
    one_way_cost: float,
    start: str,
    end: str,
    start_cash: float = 100_000.0,
    successors: Collection[str] = frozenset(),
) -> BookResult:
    """Trade ``steps`` through a tax-lot book over the trading days in ``[start, end]``.

    Args:
        close: Daily closes, trading days x symbols (NaN where there is no bar).
        steps: Rebalance targets; each ``TargetStep.date`` must be a trading day in the window.
        dead_exits: Symbol to the last bar of its series, for every series that ends in the
            window, successors included (``replay.span_end_dates``).
        exit_close: ``close`` with each ending stock's last bar replaced by its sale price
            (``replay.exit_close``); read only on those last bars.
        params: Tax rates.
        one_way_cost: Cost per side as a fraction of traded value.
        start: First day of the window, ``YYYY-MM-DD``.
        end: Last day of the window; everything is sold on that day.
        start_cash: Opening cash.
        successors: Symbols in ``dead_exits`` that are bought-out or merged stocks, which are
            not counted as dead exits (their price is unchanged by ``exit_close``).

    Returns:
        A ``BookResult``.

    Raises:
        ValueError: on an empty window, a step outside the window or a negative cost.
    """
    if one_way_cost < 0.0 or one_way_cost >= 1.0:
        raise ValueError("one_way_cost must be in [0, 1)")
    index = _naive(close.index)
    inside = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    days = index[inside]
    if len(days) == 0:
        raise ValueError("no trading days in the window")
    raw = close.to_numpy(dtype=float)[inside]
    marks = close.ffill().to_numpy(dtype=float)[inside]
    exit_px = (
        exit_close.set_axis(_naive(exit_close.index), axis=0)
        .reindex(index=days, columns=close.columns)
        .to_numpy(dtype=float)
    )
    by_date: dict[pd.Timestamp, TargetStep] = {}
    for item in steps:
        when = _naive_stamp(item.date)
        if when not in days:
            raise ValueError(f"step date {when.date()} is not a trading day in the window")
        by_date[when] = item
    ended = {str(s): _naive_stamp(t) for s, t in dead_exits.items()}
    exits_on: dict[pd.Timestamp, list[str]] = {}
    for symbol, when in ended.items():
        exits_on.setdefault(when, []).append(symbol)
    run = _Execution(close.columns, params, one_way_cost, start_cash, successors, ended)
    equity = np.empty(len(days))
    last = len(days) - 1
    for p, day in enumerate(days):
        for symbol in exits_on.get(day, []):
            run.dead_exit(symbol, raw[p], exit_px[p], day)
        if p > 0 and day.year > days[p - 1].year:
            for year in range(days[p - 1].year, day.year):
                run.book.year_end(year, day)
            run.settle_tax(day, raw[p])
        if p == last:
            run.liquidate(day, marks[p])
        elif day in by_date:
            run.rebalance(by_date[day].targets, day, raw[p], marks[p])
        equity[p] = run.marks(marks[p]) + run.book.cash + run.tax_paid
    after_tax = float(run.book.cash)
    span_days = (days[-1] - days[0]).days
    return BookResult(
        pretax_equity=pd.Series(equity, index=close.index[inside], name="pretax_equity"),
        after_tax_final=after_tax,
        tax_paid=float(run.tax_paid),
        cost_paid=float(run.cost_paid),
        n_dead_exits=run.n_dead,
        dead_exit_loss_realised=float(run.dead_loss),
        cagr_pretax=_cagr(start_cash, float(equity[-1]), span_days),
        cagr_after_tax=_cagr(start_cash, after_tax, span_days),
        trades=run.trades,
        unpaid_tax_left=float(run.book.unpaid_tax),
        skipped_buys=run.skipped,
        chain_verified=bool(run.book.verify_chain()),
        min_cash=float(run.min_cash),
    )


def run_spy(
    spy_close: pd.Series,
    *,
    params: TaxParams,
    one_way_cost: float,
    start: str,
    end: str,
    start_cash: float = 100_000.0,
) -> BookResult:
    """Buy SPY with all cash on the first day, sell on the last: same book, costs and tax.

    Args:
        spy_close: SPY daily closes.
        params: Tax rates.
        one_way_cost: Cost per side as a fraction of traded value.
        start: First day of the window.
        end: Last day of the window.
        start_cash: Opening cash.

    Returns:
        A ``BookResult`` for the buy-and-hold twin.
    """
    frame = spy_close.rename("SPY").to_frame()
    index = _naive(frame.index)
    inside = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    if not inside.any():
        raise ValueError("no trading days in the window")
    first = frame.index[inside][0]
    twin = TargetStep(decision_date=first, date=first, targets={"SPY": 1.0})
    return run_book(
        frame,
        [twin],
        pd.Series([], index=pd.Index([], dtype=object), dtype="datetime64[ns]"),
        frame,
        params=params,
        one_way_cost=one_way_cost,
        start=start,
        end=end,
        start_cash=start_cash,
    )
