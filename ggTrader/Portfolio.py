from ggTrader.Position import Position
from tabulate import tabulate

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


class Portfolio:
    def __init__(self, cash: int = 10000, transaction_fee: float = 0.004):
        self.trades: list[Position] = []
        self.positions: list[Position] = []
        self.cash = cash
        self.starting_cash = cash
        self.transaction_fee = transaction_fee  # max maker fee
        self.equity_curve = pd.Series(dtype=float)
        self.profit_per_symbol = {}
        self.transaction_fee_total = 0.0

        # NEW: track realized profit separately from unrealized
        self.realized_profit: float = 0.0

    def add_position(self, position: Position):

        position.entry_fee = position.cost * self.transaction_fee
        self.transaction_fee_total += position.entry_fee
        self.cash -= position.cost + position.entry_fee
        self.positions.append(position)
        self.trades.append(position)

    def close_position(self, position: Position, date: datetime):
        # LOCK IN REALIZED PROFIT (accounting for fees)
        exit_value = position.current_value
        position.exit_fee = exit_value * self.transaction_fee
        self.transaction_fee_total += position.exit_fee
        self.realized_profit += position.profit - position.exit_fee - position.entry_fee

        self.cash += exit_value - position.exit_fee
        position.exit_date = date
        position.status = 'closed'
        position.exit_price = position.current_price
        # self.trades.append(position.__dict__.copy())
        self.positions.remove(position)

    def update_position_stop_loss(self, symbol: str, stop_loss: float):
        position = self.get_position(symbol)
        if position:
            position.stop_loss = stop_loss

    def update_position_price(self, symbol: str, price: float, date: datetime):
        position = self.get_position(symbol)
        if position:
            position.update_price(price, date)

    @property
    def profit(self):
        return self.total_value - self.starting_cash

    @property
    def profit_pct(self):
        return self.profit / self.starting_cash

    # NEW: unrealized gain/loss for open positions
    @property
    def unrealized_profit(self) -> float:
        total = 0.0
        for position in self.positions:
            total += position.profit - position.entry_fee
        return total

    # NEW: realized + unrealized

    def get_position(self, symbol: str):
        for position in self.positions:
            if position.symbol == symbol:
                return position
        print(f"Position for {symbol} not found")
        return None

    def in_position(self, symbol: str):
        for position in self.positions:
            if position.symbol == symbol:
                return True
        return False

    def print_positions(self):
        pos = []
        for position in self.positions:
            pos.append(position.as_dict())
        print("\nPositions:")
        print(tabulate(pos, headers="keys", tablefmt="github", showindex=True))

    def print_trades(self):
        trades = []
        for trade in self.trades:
            trades.append(trade.as_dict())
        print("\nTrades:")
        print(tabulate(trades, headers="keys", tablefmt="github", showindex=True))

    @property
    def total_value(self):
        return self.cash + self.total_position_value

    @property
    def total_position_value(self):
        total = 0.0
        for position in self.positions:
            total += position.current_value
        return total

    def get_profit_per_symbol(self):
        from collections import defaultdict

        profit_per_symbol = defaultdict(float)

        for trade in self.trades:
            symbol = trade.symbol
            profit_per_symbol[symbol] += trade.profit

        return dict(profit_per_symbol)

    def print_profit_per_symbol(self):
        print("\nProfit per Symbol:")
        profits = self.get_profit_per_symbol()
        if not profits:
            print("  (no trades)")
            return

        table = []
        for symbol, profit in sorted(profits.items(), key=lambda x: x[1], reverse=True):
            table.append([symbol, f"${profit:,.2f}"])
        print(tabulate(table, headers=["Symbol", "Profit"], tablefmt="github"))

    def record_equity(self, date: datetime):
        """
        Snapshot total equity at the end of a bar.
        """
        ts = pd.Timestamp(date)
        total = self.total_value
        # Ensure monotonic index insertion
        self.equity_curve.loc[ts] = float(total)

    def max_drawdown(self) -> float:
        """
        Compute maximum drawdown as a positive fraction (0.25 == 25%).
        Uses the current equity_curve series. Returns 0.0 if not enough data.
        """
        eq = getattr(self, "equity_curve", None)
        if not isinstance(eq, pd.Series) or eq.empty or len(eq) < 2:
            return 0.0

        # Ensure sorted by time
        eq = eq.sort_index().astype(float)

        # running maximum
        running_max = eq.cummax()
        # drawdown series: (peak - trough) / peak
        dd = (running_max - eq) / running_max
        # maximum drawdown
        max_dd = float(dd.max(skipna=True)) if not dd.empty else 0.0
        # guard against nan
        if pd.isna(max_dd):
            return 0.0
        return max_dd

    def max_drawdown_pct(self) -> float:
        """
        Return max drawdown in percent (e.g. 25.0 for 25%).
        """
        return self.max_drawdown() * 100.0

    def print_stats(self):
        print("\nStats:")
        print(f"Cash: ${self.cash:,.2f}")
        print(f"Total Position Value: ${self.total_position_value:,.2f}")
        print(f"Total Value: ${self.total_value:,.2f}")
        print(f"Realized Profit: ${self.realized_profit:,.2f}")
        print(f"Unrealized Profit: ${self.unrealized_profit:,.2f}")
        print(f"Total Profit: ${self.profit:,.2f}")
        print(f"Profit Pct: {self.profit_pct * 100:.2f}%")
        print(f"Transaction Fees: ${self.transaction_fee_total:,.2f}")
        print(f"Total Trades: {len(self.trades)}")

    def reconcile(self):
        """
        Diagnostic helper: checks fee bookkeeping and P&L reconciliation.
        Prints:
          - sum of entry fees recorded on positions/trades
          - sum of exit fees recorded on closed positions
          - transaction_fee_total (tracked)
          - realized + unrealized vs total_value - starting_cash
          - any trades missing entry_fee/exit_fee attributes
        """
        sum_entry = 0.0
        sum_exit = 0.0
        missing_entry = 0
        missing_exit = 0

        # examine all trades (they are Position objects in self.trades)
        for t in self.trades:
            ef = getattr(t, "entry_fee", None)
            xf = getattr(t, "exit_fee", None)
            if ef is None:
                missing_entry += 1
            else:
                sum_entry += float(ef)
            if xf is None:
                # treat missing exit_fee as zero for open positions; count if trade is closed and missing it
                if getattr(t, "status", None) == "closed":
                    missing_exit += 1
                # else open position -> no exit_fee yet
            else:
                sum_exit += float(xf)

        left = self.starting_cash + self.realized_profit + self.unrealized_profit
        right = self.total_value

        print("\nReconciliation:")
        print(f"  starting_cash: ${self.starting_cash:,.2f}")
        print(f"  realized_profit: ${self.realized_profit:,.2f}")
        print(f"  unrealized_profit: ${self.unrealized_profit:,.2f}")
        print(f"  starting_cash + realized + unrealized = ${left:,.2f}")
        print(f"  total_value = ${right:,.2f}")
        print(f"  difference (left - right) = ${left - right:,.2f}")
        print()
        print(f"  transaction_fee_total (tracked): ${self.transaction_fee_total:,.2f}")
        print(f"  sum_entry_fees (from positions/trades): ${sum_entry:,.2f}")
        print(f"  sum_exit_fees  (from positions/trades): ${sum_exit:,.2f}")
        print(f"  missing entry_fee on trades: {missing_entry}")
        print(f"  missing exit_fee on closed trades: {missing_exit}")

    def stats_dict(self):
        return {
            "cash": self.cash,
            "total_position_value": self.total_position_value,
            "total_value": self.total_value,
            "realized_profit": self.realized_profit,
            "unrealized_profit": self.unrealized_profit,
            "total_profit": self.profit,
            "profit_pct": self.profit_pct * 100,
            "transaction_fee_total": self.transaction_fee_total,
            "total_trades": len(self.trades),
            "sharpe": self.sharpe_ratio(),
            "max_drawdown": self.max_drawdown(),          # fraction (0..1)
            "max_drawdown_pct": self.max_drawdown_pct()   # percent (0..100)
 
        }

    @staticmethod
    def dict_to_text(d, float_fmt="{:.2f}"):
        lines = []
        for k, v in d.items():
            if isinstance(v, (int, float)):
                lines.append(f"{k}: {float_fmt.format(v)}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def update_stop_loss(self, stop_loss: float):
        self.stop_loss = stop_loss

    def get_stats_df(self):
        return pd.DataFrame([self.stats_dict()])

    def print_stats_df(self):
        t = self.get_stats_df().T.reset_index().round(2)
        t.columns = ["Metric", "Value"]
        print(tabulate(t, headers="keys", tablefmt="github", showindex=False))

    def plot_equity_curve(
            self,
            figsize=(16, 8),
            title: str = "Equity Curve",
            fill=True,
            show=True,
    ):
        if self.equity_curve.empty:
            print("No equity data to plot.")
            return

        eq = self.equity_curve.sort_index().astype(float)
        x = eq.index
        baseline = float(self.starting_cash)

        # Split into above/below segments (NaNs break the line)
        above = eq.where(eq >= baseline)
        below = eq.where(eq < baseline)

        fig, ax = plt.subplots(figsize=figsize)

        # Lines
        ax.plot(x, above, color="tab:green", lw=1.5, label="Above start")
        ax.plot(x, below, color="tab:red", lw=1.5, label="Below start")

        # Baseline
        ax.axhline(baseline, color="blue", ls="--", lw=1.2,
                   label=f"Starting Cash (${baseline:,.0f})")

        # Optional soft fill to baseline
        if fill:
            ax.fill_between(x, above, baseline, where=above.notna(),
                            alpha=0.12, interpolate=True, color="tab:green")
            ax.fill_between(x, below, baseline, where=below.notna(),
                            alpha=0.12, interpolate=True, color="tab:red")
        #text
        text = self.dict_to_text(self.stats_dict())
        ax.text(
            0.01, .95,  # x, y in axis coordinates
            text,
            transform=ax.transAxes,  # use axes coordinates (0..1)
            fontsize=10,
            va='top',
            ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='black')
        )

        # Cosmetics
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity ($)")
        ax.grid(alpha=0.3, linestyle="--")
        # ax.legend(frameon=False)
        fig.tight_layout()

        if show:
            plt.show()
        return ax

    def sharpe_ratio(self, periods_per_year: int | None = None, rf_annual: float = 0.01, method: str = "log") -> float:
        """
        Compute annualized Sharpe ratio from the portfolio equity_curve.

        Args:
            periods_per_year: bars per year (e.g. 4H bars -> 6*365). If None, infer from index spacing.
            rf_annual: annual risk-free rate (decimal).
            method: "log" for log returns, "simple" for simple pct returns.

        Returns:
            float: annualized Sharpe ratio (0.0 if not enough data).
        """
        import numpy as np
        import pandas as pd

        eq = getattr(self, "equity_curve", None)
        if not isinstance(eq, pd.Series) or eq.empty or len(eq) < 3:
            return 0.0

        eq = eq.sort_index()

        # infer periods per year from index spacing if not provided
        if periods_per_year is None:
            try:
                delta_seconds = (eq.index[1] - eq.index[0]).total_seconds()
                bars_per_day = (24 * 3600) / delta_seconds if delta_seconds > 0 else 1.0
                periods_per_year = max(1, int(round(bars_per_day * 365)))
            except Exception:
                periods_per_year = 6 * 365  # fallback (4h)

        # compute per-period returns
        if method == "log":
            rets = np.log(eq / eq.shift(1)).dropna()
        else:
            rets = eq.pct_change().dropna()

        if rets.empty:
            return 0.0

        # convert annual rf to per-period
        rf_per_period = (1 + rf_annual) ** (1 / periods_per_year) - 1.0

        excess = rets - rf_per_period
        mu = excess.mean()
        sigma = excess.std(ddof=1)

        if sigma == 0 or np.isnan(sigma):
            return 0.0

        sharpe = float((mu / sigma) * np.sqrt(periods_per_year))
        return sharpe
