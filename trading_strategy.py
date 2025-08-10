import pandas as pd
import ta
from tabulate import tabulate
from typing import Dict, Optional, Union, List, Any
from dataclasses import dataclass
from trailing_stop import TrailingStop
import numpy as np  # added


@dataclass
class TradeResult:
    buy_time: pd.Timestamp
    buy_price: float
    planned_exit_time: pd.Timestamp
    planned_exit_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str  # "signal" or "trailing"


class TradingStrategy:
    def __init__(self, name: str, params: Dict, trailing_pct: float):
        self.name = name
        self.params = params
        self.trailing_pct = trailing_pct
        self.signal_df: pd.DataFrame = pd.DataFrame()


class EMAStrategy(TradingStrategy):
    def __init__(self, name: str, params: Dict, trailing_pct: float):
        super().__init__(name, params, trailing_pct)
        self.ema_fast: int = int(params['ema_fast'])
        self.ema_slow: int = int(params['ema_slow'])
        if self.ema_slow <= self.ema_fast:
            raise ValueError("ema_slow must be greater than ema_fast")

    @staticmethod
    def _ensure_input_df(price_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(price_df, pd.DataFrame):
            raise TypeError("price_series must be a pandas DataFrame with a 'close' column.")
        must_have_cols = {'close'}
        if not must_have_cols.issubset(price_df.columns):
            raise KeyError("Input DataFrame must contain a 'close' column.")
        df = price_df.copy()
        # Ensure OHLCV numeric (trailing stop loop uses open/high/low)
        for col in ('open', 'high', 'low', 'close', 'volume'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Ensure datetime-like, UTC, sorted
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        return df

    def calculate_emas(self, price_series: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fast and slow EMAs and return as a new DataFrame with columns:
        ['close', 'ema_fast', 'ema_slow'].
        """
        df = self._ensure_input_df(price_series)
        ema_fast_ind = ta.trend.EMAIndicator(close=df['close'], window=self.ema_fast)
        ema_slow_ind = ta.trend.EMAIndicator(close=df['close'], window=self.ema_slow)
        df['ema_fast'] = ema_fast_ind.ema_indicator()
        df['ema_slow'] = ema_slow_ind.ema_indicator()
        return df

    def find_crossovers(self, price_series: pd.DataFrame) -> pd.DataFrame:
        """
        Build signals:
        1 for bullish crossover (fast crosses above slow),
        -1 for bearish crossover (fast crosses below slow),
        0 otherwise. Avoid signals during EMA warm-up (NaNs).

        Returns the DataFrame with ['close', 'ema_fast', 'ema_slow', 'signal'].
        """
        df = self.calculate_emas(price_series)
        df['signal'] = pd.Series(0, index=df.index, dtype='int8')

        fast = df['ema_fast']
        slow = df['ema_slow']
        valid = fast.notna() & slow.notna()

        bullish = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        bearish = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        df.loc[valid & bullish, 'signal'] = 1
        df.loc[valid & bearish, 'signal'] = -1

        self.signal_df = df
        return df

    def print_signals(self) -> None:
        df = self.signal_df
        if df.empty or 'signal' not in df:
            print("No signals computed yet. Call find_crossovers() first.")
            return
        new_df = df[df['signal'] != 0][['close', 'ema_fast', 'ema_slow', 'signal']]
        if new_df.empty:
            print("No crossover signals found.")
            return
        print(tabulate(new_df, headers='keys', tablefmt='github'))

    def get_latest_signal_row(self) -> Optional[pd.Series]:
        df = self.signal_df
        if df.empty or 'signal' not in df:
            return None
        signal_row = df[df['signal'] != 0]
        if not signal_row.empty:
            return signal_row.iloc[-1]
        return None

    def get_latest_signal(self) -> int:
        row = self.get_latest_signal_row()
        return int(row['signal']) if row is not None else 0

    def get_latest_signal_time(self) -> Optional[pd.Timestamp]:
        row = self.get_latest_signal_row()
        return row.name if row is not None else None

    def get_latest_signal_price(self) -> Optional[float]:
        row = self.get_latest_signal_row()
        return float(row['close']) if row is not None else None

    def _bar_price(self, row: pd.Series, mode: str) -> float:
        """
        Compute a representative price for trailing stop update/trigger according to mode.
        Fallback to 'close' if needed.
        """
        close = float(row['close']) if 'close' in row and pd.notna(row['close']) else np.nan
        if mode == 'ohlc4':
            vals = []
            for k in ('open', 'high', 'low', 'close'):
                if k in row and pd.notna(row[k]):
                    vals.append(float(row[k]))
            if vals:
                return float(np.mean(vals))
            return close if not np.isnan(close) else 0.0
        elif mode == 'hl2':
            h = float(row['high']) if 'high' in row and pd.notna(row['high']) else np.nan
            l = float(row['low']) if 'low' in row and pd.notna(row['low']) else np.nan
            if not np.isnan(h) and not np.isnan(l):
                return (h + l) / 2.0
            return close if not np.isnan(close) else 0.0
        elif mode == 'close':
            return close if not np.isnan(close) else 0.0
        else:
            # default handled separately for "highlow"
            return close if not np.isnan(close) else 0.0

    def backtest(
        self,
        price_df: pd.DataFrame,
        starting_cash: float = 1000.0,
        min_hold_bars: int = 0,
        use_trailing: bool = True,
        trailing_price_mode: str = "highlow"  # new: 'highlow' (default), 'ohlc4', 'hl2', 'close'
    ) -> Dict[str, Any]:
        """
        Simple long-only backtest:
        - Enter on +1 (golden cross), exit on next -1 (death cross).
        - Optional trailing stop:
          highlow: update with bar high, trigger with bar low (original behavior).
          ohlc4:   use (O+H+L+C)/4 for update & trigger.
          hl2:     use (H+L)/2 for update & trigger.
          close:   use Close for update & trigger.
        - Compounds the cash across trades.
        """
        df = self.find_crossovers(price_df).copy()
        if df.empty:
            return {'final_cash': starting_cash, 'total_return_pct': 0.0, 'trades': [], 'signal_df': df}

        buy_times = df.index[df['signal'] == 1]
        sell_times = df.index[df['signal'] == -1]

        trades: List[TradeResult] = []
        cash = float(starting_cash)

        for buy_time in buy_times:
            future_sells = sell_times[sell_times >= buy_time]
            planned_exit_time = future_sells[0] if len(future_sells) > 0 else df.index[-1]

            buy_price = float(df.at[buy_time, 'close'])
            planned_exit_price = float(df.at[planned_exit_time, 'close'])

            exit_time = planned_exit_time
            exit_price = planned_exit_price
            exit_reason = "signal"

            if use_trailing and all(c in df.columns for c in ('high', 'low', 'open', 'close')):
                window = df.loc[buy_time:planned_exit_time]
                if not window.empty:
                    ts = TrailingStop(self.trailing_pct, initial_price=buy_price)
                    bars_held = 0
                    for ts_time, row in window.iterrows():
                        # Determine update and trigger prices based on mode
                        if trailing_price_mode == 'highlow':
                            upd_price = float(row['high']) if pd.notna(row['high']) else float(row['close'])
                            trig_price = float(row['low']) if pd.notna(row['low']) else float(row['close'])
                        else:
                            # averaged modes use the same representative price for both update and trigger
                            rep = self._bar_price(row, trailing_price_mode)
                            upd_price = rep
                            trig_price = rep

                        # Update trailing stop every bar
                        ts.update(upd_price)

                        # Enforce minimum hold bars before allowing triggers
                        if bars_held < min_hold_bars:
                            bars_held += 1
                            continue

                        if trig_price <= ts.stop_price:
                            exit_time = ts_time
                            # Exit price choice: keep using bar open for realism
                            exit_price = float(row['open']) if pd.notna(row['open']) else float(row['close'])
                            exit_reason = "trailing"
                            break

            # Apply trade PnL and compound
            if buy_price > 0:
                qty = cash / buy_price
                cash = qty * exit_price

            trades.append(TradeResult(
                buy_time=buy_time,
                buy_price=buy_price,
                planned_exit_time=planned_exit_time,
                planned_exit_price=planned_exit_price,
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason=exit_reason
            ))

        total_return_pct = ((cash / starting_cash) - 1.0) * 100.0 if starting_cash > 0 else 0.0
        return {
            'final_cash': cash,
            'total_return_pct': total_return_pct,
            'trades': trades,
            'signal_df': df
        }
