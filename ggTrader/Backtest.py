import pandas as pd
import numpy as np

trading_map = {'4h': 6 * 365, '1h': 24 * 365, '1d': 365}


class Backtest:
    def __init__(self, signals: pd.DataFrame, interval: str = '4h', transaction_fee: float = 0.004):
        self.signals = signals
        self.transaction_fee = transaction_fee
        self.stats = {}
        self.profit_df = pd.DataFrame()
        self.interval = interval

    def run(self):
        filtered_entry = self.signals['filtered_entry']
        filtered_exit = self.signals['filtered_exit']

        # Backtest
        entry_times = list(filtered_entry[filtered_entry].index)
        exit_times = list(filtered_exit[filtered_exit].index)
        pairs = self.position_time_pairs(entry_times, exit_times)

        # Calc profits
        close = self.signals['close']
        exits_price = self.signals['ce_l'].shift(-1).ffill() * filtered_exit

        entries_price = close * filtered_entry

        self.profit_df = self.backtest_from_pairs(pairs, exits_price, entries_price)

        wins, loss = self.calc_wins_losses(self.profit_df['Profit'])
        self.stats['wins'] = wins
        self.stats['loss'] = loss
        self.stats['win_rate'] = wins / (wins + loss) * 100
        self.stats['trading_days'] = (self.signals.index[-1] - self.signals.index[0]).days
        self.stats['sharpe_ratio'] = self.sharpe_ratio(self.profit_df['Profit_Pct'], self.interval)
        self.stats['sortino_ratio'] = self.sortino_ratio(self.profit_df['Profit_Pct'], self.interval)
        self.stats['total_trades'] = len(pairs)
        self.stats['ave_profit'] = self.profit_df['Profit'].mean()
        self.stats['std_profit'] = self.profit_df['Profit'].std()
        self.stats['ave_profit_pct'] = self.profit_df['Profit_Pct'].mean()
        self.stats['std_profit_pct'] = self.profit_df['Profit_Pct'].std()

    @staticmethod
    def position_time_pairs(entry_times: pd.Series, exit_times: pd.Series):
        # 2) Pair entries with the next exit after each entry
        pairs = []
        j = 0
        for et in entry_times:
            while j < len(exit_times) and exit_times[j] <= et:
                j += 1
            if j < len(exit_times):
                pairs.append((et, exit_times[j]))
                j += 1
        return pairs

    def backtest_from_pairs(self, pairs: list, exits_price: pd.Series, entries_price: pd.Series):
        profit = pd.Series(0.0, index=self.signals.index)
        profit_pct = profit.copy()
        for pair in pairs:
            exit_price = exits_price.loc[pair[1]]
            entry_price = entries_price.loc[pair[0]]
            profit.loc[pair[1]] = exit_price - entry_price - (self.transaction_fee * (exit_price + entry_price))
            profit_pct.loc[pair[1]] = profit.loc[pair[1]] / entry_price * 100

        profit_cum = profit.cumsum()
        profit[profit == 0.0] = np.nan
        profit_pct[profit_pct == 0.0] = np.nan
        profit_df = pd.concat([entries_price, exits_price, profit, profit_cum, profit_pct], axis=1)
        profit_df.columns = ['Entry_Price', 'Exit_Price', 'Profit', 'Cumulative_Profit', 'Profit_Pct']

        return profit_df

    @staticmethod
    def calc_wins_losses(profit: pd.Series):
        wins = (profit > 0).sum()
        loss = (profit < 0).sum()
        return wins, loss

    @staticmethod
    def sharpe_ratio(profit: pd.Series, interval: str):
        trading_periods = trading_map.get(interval)
        return profit.mean() / profit.std() * np.sqrt(trading_periods)

    @staticmethod
    def sortino_ratio(profit: pd.Series, interval: str):
        trading_periods = trading_map.get(interval)
        std_below_zero = profit[profit < 0].std()
        return profit.mean() / std_below_zero * np.sqrt(trading_periods)


if __name__ == "__main__":
    pass
