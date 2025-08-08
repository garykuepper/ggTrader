from tabulate import tabulate
import numpy as np
import pandas as pd
import mplfinance as mpf

from portfolio import Portfolio, Position


class BacktestSimulator:
    def __init__(self, symbol_strategy_map, price_data_map, position_pct_map, initial_cash=10000):
        self.symbol_strategy_map = symbol_strategy_map
        self.price_data_map = price_data_map
        self.position_pct_map = position_pct_map
        self.portfolio = Portfolio(cash=initial_cash)
        self.initial_cash = initial_cash
        self.trades = []
        self.equity_curve = []  # list of {'date': <ts>, 'equity': <float>}

    def run(self):
        all_dates = set()
        for df in self.price_data_map.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)

        # Precompute signals once per symbol
        for symbol, strategy in self.symbol_strategy_map.items():
            df = self.price_data_map[symbol]
            if not hasattr(strategy, 'signal_df') or strategy.signal_df.empty:
                strategy.find_crossovers(df)

        for date in all_dates:
            for symbol, strategy in self.symbol_strategy_map.items():
                df = self.price_data_map[symbol]
                if date not in df.index:
                    continue
                row = df.loc[date]

                # Get signal safely
                try:
                    srow = strategy.signal_df.loc[date]
                    signal = srow.get('signal', 0)
                    if pd.isna(signal):
                        signal = 0
                except KeyError:
                    signal = 0

                # Update current position first (price + trailing stop tracking)
                if self.portfolio.position_exists(symbol):
                    self.portfolio.update_position(symbol, current_price=row['close'])
                    position = self.portfolio.get_position_by_symbol(symbol)
                    if position.trailing_stop and position.trailing_stop.is_triggered(row['close']):
                        qty_to_report = position.qty
                        print(f"{date} {symbol} TRAILING STOP SELL @ {row['close']:.2f} "
                              f"(stop: {position.trailing_stop.stop_price:.2f}) total: {position.current_value:.2f}")
                        self.portfolio.exit_position_by_symbol(symbol)
                        self.trades.append({
                            'date': date, 'symbol': symbol, 'action': 'trailing_stop_sell',
                            'price': float(row['close']), 'qty': float(qty_to_report)
                        })
                        continue

                # Position sizing
                position_pct = self.position_pct_map.get(symbol, 0.10)

                # Entry only if flat
                if signal == 1 and not self.portfolio.position_exists(symbol):
                    price = float(row['close'])
                    if price <= 0:
                        continue
                    cash_to_use = self.portfolio.cash * position_pct
                    qty = cash_to_use / price
                    if qty <= 0:
                        continue
                    pos = Position(symbol, qty, price, date, trailing_pct=self.symbol_strategy_map[symbol].trailing_pct)
                    if self.portfolio.enter_position(pos):
                        total = qty * price
                        print(f"{date} {symbol} BUY  @ {price:.2f} total: {total:.2f}")
                        self.trades.append({'date': date, 'symbol': symbol, 'action': 'buy', 'price': price, 'qty': float(qty)})
                    else:
                        print(f"{date} {symbol} BUY signal but insufficient cash.")

                # Exit if signaled
                elif signal == -1:
                    position = self.portfolio.get_position_by_symbol(symbol)
                    if position:
                        price = float(row['close'])
                        qty_to_report = position.qty
                        total = position.current_value
                        print(f"{date} {symbol} SELL @ {price:.2f} total: {total:.2f}")
                        self.portfolio.exit_position_by_symbol(symbol)
                        self.trades.append({'date': date, 'symbol': symbol, 'action': 'sell', 'price': price, 'qty': float(qty_to_report)})
                    else:
                        # no position to sell; skip
                        pass

            # capture equity at end of this date
            self.equity_curve.append({'date': date, 'equity': float(self.portfolio.total_equity())})

    def print_trade_history(self):
        print(tabulate(self.trades, headers="keys", tablefmt="github"))
        print(f"Ending cash: {self.portfolio.cash:.2f} | "
              f"Positions value: {self.portfolio.total_positions_value():.2f} | "
              f"Total equity: {self.portfolio.total_equity():.2f}")

    # ===== New: Performance reporting =====

    def _build_round_trips(self):
        """
        Pair each buy with the next sell/stop for each symbol.
        Returns a list of dicts with per-trade metrics.
        """
        if not self.trades:
            return []

        # Ensure trades are sorted by date
        trades_sorted = sorted(self.trades, key=lambda x: x['date'])
        open_by_symbol = {}
        trips = []

        for t in trades_sorted:
            action = t.get('action')
            sym = t.get('symbol')
            if action == 'buy':
                open_by_symbol[sym] = t
            elif action in ('sell', 'trailing_stop_sell'):
                if sym in open_by_symbol:
                    buy = open_by_symbol.pop(sym)
                    qty = float(min(buy.get('qty', 0.0), t.get('qty', 0.0)) or 0.0)
                    buy_price = float(buy.get('price', 0.0) or 0.0)
                    sell_price = float(t.get('price', 0.0) or 0.0)
                    if qty > 0 and buy_price > 0 and sell_price > 0:
                        pnl = qty * (sell_price - buy_price)
                        ret = (sell_price / buy_price) - 1.0
                        holding_days = None
                        try:
                            holding_days = (t['date'] - buy['date']).days
                        except Exception:
                            pass
                        trips.append({
                            'symbol': sym,
                            'buy_date': buy['date'],
                            'sell_date': t['date'],
                            'qty': qty,
                            'buy_price': buy_price,
                            'sell_price': sell_price,
                            'pnl': pnl,
                            'return_pct': ret * 100.0,
                            'action_exit': action,
                            'holding_days': holding_days
                        })
        return trips

    def _equity_series(self):
        """
        Return equity curve as a pandas Series indexed by date.
        """
        if not self.equity_curve:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.equity_curve).dropna()
        df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        return pd.Series(df['equity'].values, index=df['date'])

    @staticmethod
    def _max_drawdown(equity):
        """
        Max drawdown from an equity series.
        """
        if equity.empty:
            return 0.0, pd.Timedelta(0)
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        mdd = drawdown.min()
        dd_periods = (drawdown != 0).astype(int)
        max_duration = pd.Timedelta(0)
        if not dd_periods.empty:
            current_start = None
            for idx, val in dd_periods.items():
                if val and current_start is None:
                    current_start = idx
                elif not val and current_start is not None:
                    max_duration = max(max_duration, idx - current_start)
                    current_start = None
        return float(mdd), max_duration

    def performance_summary(self):
        trips = self._build_round_trips()
        equity = self._equity_series()

        rows = []
        by_symbol = {}
        for tr in trips:
            by_symbol.setdefault(tr['symbol'], []).append(tr)

        for sym, items in by_symbol.items():
            pnls = [x['pnl'] for x in items]
            rets = [x['return_pct'] for x in items]
            wins = [x for x in pnls if x > 0]
            losses = [x for x in pnls if x <= 0]
            n = len(pnls)
            win_rate = (len(wins) / n * 100.0) if n else 0.0
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            avg_ret = np.mean(rets) if rets else 0.0
            total_pnl = float(np.sum(pnls))
            rows.append({
                'Symbol': sym,
                'Trades': n,
                'Wins': len(wins),
                'Losses': len(losses),
                'Win Rate %': round(win_rate, 2),
                'Avg Win': round(avg_win, 2),
                'Avg Loss': round(avg_loss, 2),
                'Avg Trade Return %': round(avg_ret, 2),
                'Total PnL': round(total_pnl, 2),
            })

        per_symbol_df = pd.DataFrame(rows).sort_values(['Symbol']).reset_index(drop=True)

        final_equity = float(equity.iloc[-1]) if not equity.empty else float(self.portfolio.total_equity())
        total_pnl = final_equity - float(self.initial_cash)
        total_return_pct = (final_equity / float(self.initial_cash) - 1.0) * 100.0 if self.initial_cash else 0.0

        if not equity.empty and len(equity) >= 2:
            daily_returns = equity.pct_change().dropna()
            ann_factor = 365.0
            vol_annual = float(daily_returns.std() * np.sqrt(ann_factor)) if not daily_returns.empty else 0.0
            sharpe = float((daily_returns.mean() * ann_factor) / vol_annual) if vol_annual > 0 else 0.0
            days = (equity.index[-1] - equity.index[0]).days or 1
            years = days / 365.0
            cagr = (final_equity / float(self.initial_cash)) ** (1.0 / years) - 1.0 if years > 0 else 0.0
            mdd, mdd_dur = self._max_drawdown(equity)
        else:
            vol_annual = 0.0
            sharpe = 0.0
            cagr = 0.0
            mdd, mdd_dur = 0.0, pd.Timedelta(0)

        portfolio_row = {
            'Final Equity': round(final_equity, 2),
            'Total PnL': round(total_pnl, 2),
            'Total Return %': round(total_return_pct, 2),
            'CAGR %': round(cagr * 100.0, 2),
            'Volatility % (ann)': round(vol_annual * 100.0, 2),
            'Sharpe': round(sharpe, 2),
            'Max Drawdown %': round(mdd * 100.0, 2),
            'MDD Duration': str(mdd_dur),
            'Trades': int(sum(per_symbol_df['Trades'])) if not per_symbol_df.empty else 0
        }
        portfolio_df = pd.DataFrame([portfolio_row])
        return per_symbol_df, portfolio_df

    def print_performance_summary(self):
        per_symbol_df, portfolio_df = self.performance_summary()
        if not per_symbol_df.empty:
            print("\nPer-symbol performance")
            per_symbol_df = per_symbol_df.sort_values('Avg Trade Return %', ascending=False)
            print(tabulate(per_symbol_df, headers="keys", tablefmt="github", showindex=False))
        else:
            print("\nPer-symbol performance\n(no completed trades)")
        print("\nPortfolio performance")
        print(tabulate(portfolio_df, headers="keys", tablefmt="github", showindex=False))

    # ===== New: Charting with mplfinance =====

    def _get_symbol_trades(self, symbol, start=None, end=None):
        """
        Return buy and sell dates for a symbol within optional window.
        """
        buys, sells = [], []
        for t in self.trades:
            if t.get('symbol') != symbol:
                continue
            dt = t.get('date')
            if start is not None and dt < start:
                continue
            if end is not None and dt > end:
                continue
            action = t.get('action')
            if action == 'buy':
                buys.append(dt)
            elif action in ('sell', 'trailing_stop_sell'):
                sells.append(dt)
        return buys, sells

    def plot_symbol(self, symbol, start=None, end=None, style='yahoo', volume=True, figsize=(12, 8)):
        """
        Plot a single symbol with candles, EMAs, and buy/sell markers.
        - start/end: optional datetime bounds (must match index tz)
        """
        if symbol not in self.price_data_map:
            print(f"No price data for {symbol}")
            return

        df = self.price_data_map[symbol].copy()
        if df.empty:
            print(f"Empty price data for {symbol}")
            return

        # Clip range
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        if df.empty:
            print(f"No data for {symbol} in the selected window.")
            return

        # Ensure EMAs and signals exist via strategy
        strat = self.symbol_strategy_map.get(symbol)
        if strat is None:
            print(f"No strategy for {symbol}")
            return
        if not hasattr(strat, 'signal_df') or strat.signal_df.empty:
            strat.find_crossovers(self.price_data_map[symbol])

        # Align strategy df to chart df
        sdf = strat.signal_df.reindex(df.index).copy()
        # Fallback: if EMA columns missing, compute
        if 'ema_fast' not in sdf.columns or 'ema_slow' not in sdf.columns:
            sdf = strat.calculate_emas(df)

        # mpf requires specific column names
        mpf_df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })

        # Create EMA addplots
        apds = [
            mpf.make_addplot(sdf['ema_fast'], color='blue', width=1.0),
            mpf.make_addplot(sdf['ema_slow'], color='orange', width=1.0),
        ]

        # Build buy/sell marker series
        buys, sells = self._get_symbol_trades(symbol, start=start, end=end)

        buy_marks = pd.Series(np.nan, index=mpf_df.index)
        if buys:
            idx = [d for d in buys if d in mpf_df.index]
            buy_marks.loc[idx] = mpf_df.loc[idx, 'Close']
            apds.append(mpf.make_addplot(buy_marks, type='scatter', markersize=80, marker='^', color='green'))

        sell_marks = pd.Series(np.nan, index=mpf_df.index)
        if sells:
            idx = [d for d in sells if d in mpf_df.index]
            sell_marks.loc[idx] = mpf_df.loc[idx, 'Close']
            apds.append(mpf.make_addplot(sell_marks, type='scatter', markersize=80, marker='v', color='red'))

        title = f"{symbol} | {strat.name} | Fast EMA={getattr(strat, 'ema_fast', '?')} Slow EMA={getattr(strat, 'ema_slow', '?')}"

        mpf.plot(
            mpf_df,
            type='candle',
            style=style,
            addplot=apds,
            volume=volume,
            figsize=figsize,
            tight_layout=True,
            title=title,
            warn_too_much_data=10000
        )

    def plot_all(self, style='yahoo', volume=True, figsize=(12, 8)):
        """
        Plot all symbols managed by this backtest.
        """
        for symbol in self.price_data_map.keys():
            self.plot_symbol(symbol, style=style, volume=volume, figsize=figsize)
