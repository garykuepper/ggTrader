"""Legacy portfolio simulator used by the iterative Trading engine."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from ggTrader.core.position import Position


class Portfolio:
    """Tracks cash, positions, trades, and equity curve for the legacy engine."""

    def __init__(self, cash: int = 10000, transaction_fee: float = 0.004) -> None:
        self.trades: list[Position] = []
        self.positions: list[Position] = []
        self.cash = cash
        self.starting_cash = cash
        self.transaction_fee = transaction_fee
        self.equity_curve = pd.Series(dtype=float)
        self.transaction_fee_total = 0.0
        self.realized_profit: float = 0.0

    def add_position(self, position: Position) -> None:
        position.entry_fee = position.cost * self.transaction_fee
        self.transaction_fee_total += position.entry_fee
        self.cash -= position.cost + position.entry_fee
        self.positions.append(position)
        self.trades.append(position)

    def close_position(self, position: Position, date: datetime) -> None:
        exit_value = position.current_value
        position.exit_fee = exit_value * self.transaction_fee
        self.transaction_fee_total += position.exit_fee
        self.realized_profit += position.profit - position.exit_fee - position.entry_fee

        self.cash += exit_value - position.exit_fee
        position.exit_date = date
        position.status = "closed"
        position.exit_price = position.current_price
        self.positions.remove(position)

    def update_position_price(self, symbol: str, price: float, date: datetime) -> None:
        position = self.get_position(symbol)
        if position:
            position.update_price(price, date)

    def get_position(self, symbol: str) -> Optional[Position]:
        for position in self.positions:
            if position.symbol == symbol:
                return position
        return None

    def in_position(self, symbol: str) -> bool:
        return any(p.symbol == symbol for p in self.positions)

    @property
    def total_position_value(self) -> float:
        return float(sum(p.current_value for p in self.positions))

    @property
    def total_value(self) -> float:
        return float(self.cash + self.total_position_value)

    @property
    def profit(self) -> float:
        return float(self.total_value - self.starting_cash)

    @property
    def profit_pct(self) -> float:
        return float(self.profit / self.starting_cash) if self.starting_cash else 0.0

    @property
    def unrealized_profit(self) -> float:
        return float(sum(p.profit - p.entry_fee for p in self.positions))

    def record_equity(self, date: datetime) -> None:
        ts = pd.Timestamp(date)
        self.equity_curve.loc[ts] = float(self.total_value)

    def get_profit_per_symbol(self) -> dict[str, float]:
        profit_per_symbol: dict[str, float] = defaultdict(float)
        for trade in self.trades:
            profit_per_symbol[trade.symbol] += float(trade.profit)
        return dict(profit_per_symbol)

    def print_positions(self) -> None:
        pos = [p.as_dict() for p in self.positions]
        print("\nPositions:")
        print(tabulate(pos, headers="keys", tablefmt="github", showindex=True))

    def print_trades(self) -> None:
        trades = [t.as_dict() for t in self.trades]
        print("\nTrades:")
        print(tabulate(trades, headers="keys", tablefmt="github", showindex=True))

    def plot_equity_curve(
        self,
        figsize: tuple[int, int] = (16, 8),
        title: str = "Equity Curve",
        fill: bool = True,
        show: bool = True,
    ):
        if self.equity_curve.empty:
            print("No equity data to plot.")
            return None

        eq = self.equity_curve.sort_index().astype(float)
        x = eq.index
        baseline = float(self.starting_cash)

        above = eq.where(eq >= baseline)
        below = eq.where(eq < baseline)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, above, color="tab:green", lw=1.5, label="Above start")
        ax.plot(x, below, color="tab:red", lw=1.5, label="Below start")
        ax.axhline(
            baseline,
            color="blue",
            ls="--",
            lw=1.2,
            label=f"Starting Cash (${baseline:,.0f})",
        )

        if fill:
            ax.fill_between(
                x,
                above,
                baseline,
                where=above.notna(),
                alpha=0.12,
                interpolate=True,
                color="tab:green",
            )
            ax.fill_between(
                x,
                below,
                baseline,
                where=below.notna(),
                alpha=0.12,
                interpolate=True,
                color="tab:red",
            )

        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity ($)")
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()

        if show:
            plt.show()
        return ax

    def plot_value(self, **kwargs):
        if "show" not in kwargs:
            kwargs["show"] = False
        ax = self.plot_equity_curve(**kwargs)
        return ax.get_figure() if ax is not None else None

    def qty_to_buy(self, price: float, percent: float = 0.1) -> float:
        """Return quantity to buy as a percent of current portfolio value."""
        if price <= 0:
            return 0.0
        buy_value = self.total_value * percent
        if buy_value > self.cash:
            return 0.0
        return float(buy_value / price)


if __name__ == "__main__":
    port = Portfolio(cash=100000)
    pos = Position("BTC", 3, 10000, datetime(2024, 1, 1))
    port.add_position(pos)
    port.update_position_price("BTC", 30500.0, datetime.now())
    port.print_positions()

