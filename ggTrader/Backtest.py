import pandas as pd
import numpy as np



trading_map = {'4h': 6 * 365, '1h': 24 * 365, '1d': 365}


class Backtest:
    def __init__(self, signals: pd.DataFrame,
                 interval: str = '4h',
                 transaction_fee: float = 0.004,
                 start_equity: float = 1000.0,):
        self.signals = signals
        self.transaction_fee = transaction_fee
        self.stats = {}
        self.profit_df = pd.DataFrame()
        self.interval = interval
        self.start_equity = start_equity

    def run(self):
        filtered_entry = self.signals['filtered_entry']
        filtered_exit = self.signals['filtered_exit']
        in_position = self.signals['in_position']
        close = self.signals['close']
        start = self.signals.index[0]
        end = self.signals.index[-1]

        # Backtest
        exits_price = self.signals['ce_l'].shift(-1).ffill() * filtered_exit
        entries_price = close * filtered_entry

        returns = self.calc_returns(self.signals, exits_price, transaction_fee=self.transaction_fee)
        equity = self.start_equity * (1.0 + returns).cumprod()
        window_ret = self.profit_per_position(returns, in_position)
        self.profit_df = pd.concat([entries_price, exits_price, close, returns, window_ret, equity], axis=1)
        self.profit_df.columns = ['Entry_Price', 'Exit_Price', 'close', 'Returns', 'Window_Profit', 'Equity']
        sharpe = self.sharpe_ratio(returns, self.interval)
        sortino = self.sortino_ratio(returns, self.interval)
        wins, loss, win_rate = self.calc_wins_losses(self.profit_df['Window_Profit'])
        self.stats['trading_days'] = (end - start).days
        self.stats['total_profit'] = equity.iloc[-1] - equity.iloc[0]
        self.stats['total_profit_pct'] = self.stats['total_profit'] / equity.iloc[0] * 100
        self.stats['sharpe'] = sharpe
        self.stats['sortino'] = sortino
        self.stats['wins'] = wins
        self.stats['losses'] = loss
        self.stats['win_rate'] = win_rate

        return self.stats, self.profit_df


    @staticmethod
    def calc_wins_losses(profit: pd.Series):
        wins = (profit > 0).sum()
        loss = (profit < 0).sum()
        win_rate = wins / (wins + loss) * 100.0
        return wins, loss, win_rate

    @staticmethod
    def profit_per_position(returns: pd.Series, in_position: pd.Series) -> pd.DataFrame:
        # Ensure aligned indices
        returns = returns.reindex(in_position.index).astype(float)
        in_pos = in_position.fillna(False).astype(bool)

        # Create group ids for contiguous True runs; 0 for out of position
        # Each time in_pos flips from previous value, we increment a counter
        flips = in_pos.ne(in_pos.shift(fill_value=False)).cumsum()
        grp = flips.where(in_pos, 0)

        # Filter to only groups > 0 (actual positions)
        mask = grp > 0
        if not mask.any():
            return pd.Series(dtype=float)

        # Compute compounded return per position
        # Drop NaNs in returns for stability
        r = returns.fillna(0.0)
        window_ret = ((1.0 + r[mask]).groupby(grp[mask]).prod() - 1.0)

        if window_ret.empty:
            return window_ret

        # Map each group id to its exit timestamp: last index where that grp appears
        # If the last bar is still in an open position, exclude that group since there's no exit yet
        last_index_per_grp = grp[mask].groupby(grp[mask]).apply(lambda s: s.index[-1])

        # Keep only groups present in window_ret (they should match)
        last_index_per_grp = last_index_per_grp.reindex(window_ret.index)

        # Assign exit timestamps as the index
        window_ret.index = pd.Index(last_index_per_grp.values, name=returns.index.name)
        window_ret.index = pd.DatetimeIndex(window_ret.index).tz_localize('UTC')
        return pd.DataFrame(window_ret)

    @staticmethod
    def calc_returns(signals: pd.DataFrame, exits_price: pd.Series,
                     transaction_fee: float = 0.004):
        # returns
        filtered_entry_exit = signals[['filtered_entry', 'filtered_exit', 'in_position']]
        close = signals['close']
        price = close.copy().astype("float64")
        price[filtered_entry_exit['filtered_exit']] = exits_price[filtered_entry_exit['filtered_exit']]

        in_position = filtered_entry_exit['in_position'].astype(bool)
        r_cc = price.pct_change().fillna(0.0)  # close-to-close returns
        enter_bar = filtered_entry_exit['filtered_entry'].astype(bool)
        exit_bar = filtered_entry_exit['filtered_exit'].astype(bool)

        returns = r_cc.where(in_position, 0.0)
        returns = returns.mask(enter_bar, returns - transaction_fee)  # apply fee at entry
        returns = returns.mask(exit_bar, returns - transaction_fee)  # apply fee at exit
        return returns

    def calc_profits(self, signals: pd.DataFrame):
        filtered_entry = signals['filtered_entry']
        filtered_exit = signals['filtered_exit']
        # Calc profits
        close = signals['close']
        exits_price = signals['ce_l'].shift(-1).ffill() * filtered_exit
        entries_price = close * filtered_entry
        returns = self.calc_returns(signals, exits_price, transaction_fee=self.transaction_fee)
        start_equity = 1000.0
        equity = start_equity * (1.0 + returns).cumprod()
        window_ret = self.profit_per_position(returns, in_position)

        profit_df = pd.concat([entries_price, exits_price, close, returns, window_ret, equity], axis=1)
        profit_df.columns = ['Entry_Price', 'Exit_Price', 'close', 'Returns', 'Window_Profit', 'Equity']
        return profit_df

    @staticmethod
    def sharpe_ratio(returns: pd.Series, interval: str):
        trading_map = {'4h': 6 * 365, '1h': 24 * 365, '1d': 365}
        trading_periods = trading_map.get(interval)
        return returns.mean() / returns.std() * np.sqrt(trading_periods)

    @staticmethod
    def sortino_ratio(returns: pd.Series, interval: str):
        trading_map = {'4h': 6 * 365, '1h': 24 * 365, '1d': 365}
        trading_periods = trading_map.get(interval)
        return returns.mean() / returns[returns < 0].std() * np.sqrt(trading_periods)

if __name__ == "__main__":
    pass
