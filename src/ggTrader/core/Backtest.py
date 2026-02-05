import pandas as pd
import numpy as np

trading_map = {'4h': 6 * 365, '1h': 24 * 365, '1d': 365}


class Backtest:
    def __init__(self, signals: pd.DataFrame,
                 interval: str = '4h',
                 transaction_fee: float = 0.004,
                 start_equity: float = 1000.0, ):
        self.signals = signals
        self.transaction_fee = transaction_fee
        self.stats = {}
        self.profit_df = pd.DataFrame()
        self.interval = interval
        self.start_equity = start_equity

    def run(self):
        filtered_entry = self.signals['filtered_entry'].astype(bool)
        filtered_exit = self.signals['filtered_exit'].astype(bool)
        in_position = self.signals['in_position'].astype(bool)
        close = self.signals['close'].astype(float)
        start = self.signals.index[0]
        end = self.signals.index[-1]

        ce_l = self.signals['ce_l'].astype(float)
        next_ce_l = ce_l.shift(-1)

        # Reporting (only at the actual entry/exit bars)
        exits_price = pd.Series(np.nan, index=self.signals.index, dtype=float)
        exits_price.loc[filtered_exit] = next_ce_l.loc[filtered_exit]
        entries_price = pd.Series(np.nan, index=self.signals.index, dtype=float)
        entries_price.loc[filtered_entry] = close.loc[filtered_entry]

        returns = self.calc_returns(self.signals, next_ce_l, transaction_fee=self.transaction_fee)
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
        win_rate = wins / (wins + loss) * 100.0 if (wins + loss) > 0 else np.nan
        return wins, loss, win_rate

    @staticmethod
    def profit_per_position(returns: pd.Series, in_position: pd.Series) -> pd.Series:
        returns = returns.reindex(in_position.index).astype(float)
        in_pos = in_position.astype(bool)

        flips = in_pos.ne(in_pos.shift(fill_value=False)).cumsum()
        grp = flips.where(in_pos, 0)

        mask = grp > 0
        if not mask.any():
            return pd.Series(dtype=float, index=returns.index)

        r = returns.fillna(0.0)
        window_ret = ((1.0 + r[mask]).groupby(grp[mask]).prod() - 1.0)

        last_index_per_grp = grp[mask].groupby(grp[mask]).apply(lambda s: s.index[-1])
        last_index_per_grp = last_index_per_grp.reindex(window_ret.index)

        raw_idx = pd.DatetimeIndex(last_index_per_grp.values)
        returns_tz = getattr(returns.index, "tz", None)
        if returns_tz is not None:
            if raw_idx.tz is None:
                idx = raw_idx.tz_localize(returns_tz)
            else:
                idx = raw_idx.tz_convert(returns_tz)
        else:
            idx = raw_idx

        window_ret.index = idx
        return window_ret

    @staticmethod
    def calc_returns(signals: pd.DataFrame, next_ce_l: pd.Series,
                         transaction_fee: float = 0.004):
        fe = signals['filtered_entry'].astype(bool)
        fx = signals['filtered_exit'].astype(bool)
        close = signals['close'].astype(float)

        # Build in-position from entry/exit ensuring exit bar is included in the segment
        in_pos = pd.Series(False, index=signals.index, dtype=bool)
        pos = False
        for t in signals.index:
            if not pos and fe.loc[t]:
                pos = True
                in_pos.loc[t] = True
            elif pos:
                in_pos.loc[t] = True
                if fx.loc[t]:
                    pos = False

        # Identify contiguous segments
        seg_id = in_pos.ne(in_pos.shift(fill_value=False)).cumsum()
        seg_id = seg_id.where(in_pos, 0)

        # Per-bar returns initialized to 0 (flat when out of position)
        returns = pd.Series(0.0, index=close.index, dtype=float)
        fee_factor = 1.0 - transaction_fee

        for sid in pd.unique(seg_id[seg_id > 0]):
            mask = (seg_id == sid)
            idx = close.index[mask]
            if len(idx) == 0:
                continue

            # Entry bar: apply entry fee only via a one-time multiplicative drop
            entry_t = idx[0]
            # Represent fee as a return: new_price = old_price * fee_factor => return = fee_factor - 1
            returns.loc[entry_t] = fee_factor - 1.0

            # Iterate within the segment
            for i in range(1, len(idx)):
                t = idx[i]
                prev_t = idx[i - 1]
                prev_close = close.loc[prev_t]
                is_last = (i == len(idx) - 1)

                if not np.isfinite(prev_close) or prev_close == 0:
                    # If invalid previous price, skip bar (keeps return 0)
                    continue

                if is_last:
                    exec_price = next_ce_l.loc[t]
                    if not np.isfinite(exec_price):
                        exec_price = close.loc[t]
                    if not np.isfinite(exec_price):
                        continue
                    # Price move from prev close to exit execution price
                    price_ratio = exec_price / prev_close
                    # Combine price move and exit fee in one step: (1+r) = price_ratio * fee_factor
                    returns.loc[t] = price_ratio * fee_factor - 1.0
                else:
                    curr_close = close.loc[t]
                    if not np.isfinite(curr_close):
                        continue
                    # Price move only
                    returns.loc[t] = curr_close / prev_close - 1.0

        # Ensure first bar is 0 to avoid a jump at series start
        if len(returns) > 0:
            returns.iloc[0] = 0.0
        return returns

    @staticmethod
    def sharpe_ratio(returns: pd.Series, interval: str):
        trading_map = {'4h': 6 * 365, '1h': 24 * 365, '1d': 365}
        trading_periods = trading_map.get(interval)
        risk_free_rate = 0.0001
        mu = returns.mean() - risk_free_rate
        sigma = returns.std(ddof=0)
        if not np.isfinite(sigma) or sigma == 0:
            return np.nan
        return mu / sigma * np.sqrt(trading_periods)

    @staticmethod
    def sortino_ratio(returns: pd.Series, interval: str):
        trading_map = {'4h': 6 * 365, '1h': 24 * 365, '1d': 365}
        trading_periods = trading_map.get(interval)
        risk_free_rate = 0.0001
        downside = returns[returns < 0]
        mu = returns.mean() - risk_free_rate
        ds = downside.std(ddof=0)
        if not np.isfinite(ds) or ds == 0:
            return np.nan
        return mu / ds * np.sqrt(trading_periods)


if __name__ == "__main__":
    pass
