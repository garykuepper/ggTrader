# run_ggtrade_live.py
# A rough live-loop that mirrors run_ggtrade_old.py logic, uses ccxt Kraken data, and trades
# with Kraken via ccxt. It invests in the top 20 cryptos, mirrors available cash on Kraken,
# and manages exits with a live ATR-based stop computed in the loop (no ATRTrailingStop class used).

# language: python

import time
from datetime import datetime, timedelta, timezone
import os
import pandas as pd
import numpy as np
import ccxt
from tabulate import tabulate
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# Project-specific imports (adjust paths if needed)
from utils.kraken_yfinance_cmc import get_kraken_asset_pairs_usd
from utils.top_crypto import get_top_cmc, get_top_kraken_usd_pairs  # get_top_cmc or get_top_kraken_usd_pairs as needed
from ggTrader.Portfolio import Portfolio
from ggTrader.Position import Position

# Helper: map a symbol like "BTC/USD" to Kraken style pair (e.g., "BTCUSD")
def to_kraken_pair(symbol: str) -> str:
    s = symbol.replace("/", "")
    if s.endswith("USDT"):
        s = s[:-4] + "USD"
    if not s.endswith("USD"):
        s += "USD"
    return s

# Helper: compute end timestamp for the next 4-hour window
def next_window_expire_ts(now: datetime, window_hours: int = 4) -> int:
    # Align to end of current window and add window
    # Use UTC
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    end_of_window = now.replace(tzinfo=timezone.utc, minute=0, second=0, microsecond=0)
    hours = end_of_window.hour
    # floor to window
    start_of_window = end_of_window.replace(hour=(hours // window_hours) * window_hours)
    expire = start_of_window + timedelta(hours=window_hours)
    return int(expire.timestamp())

# Fetch top 20 cryptos and map to Kraken USD pairs
def build_live_universe(kraken: ccxt.Exchange, top_n: int = 20):
    # Use top 20 by market cap (as you suggested)
    df_top = get_top_cmc(limit=top_n, print_table=False)
    # Kraken USD pairs available
    usd_pairs = get_kraken_asset_pairs_usd()
    allowed_bases = {row["base_common"] for row in usd_pairs}
    # keep only symbols Kraken supports in USD pairs
    df_live = df_top[df_top["Symbol"].isin(allowed_bases)].copy()
    df_live = df_live.head(top_n)
    universe = []
    for _, row in df_live.iterrows():
        symbol = row.get("Symbol")
        if symbol:
            universe.append(f"{symbol}/USD")
    return universe

# Compute signals for a given ohlcv dataframe
def calc_signals_for_ohlcv(df: pd.DataFrame, ema_fast: int = 5, ema_slow: int = 20, atr_window: int = 14, atr_mult: float = 1.0):
    if df is None or df.empty:
        return pd.DataFrame(dtype=float)

    signals = pd.DataFrame()
    signals['close'] = df['close'].copy()
    signals['ema_fast'] = EMAIndicator(close=df['close'], window=ema_fast, fillna=False).ema_indicator()
    signals['ema_slow'] = EMAIndicator(close=df['close'], window=ema_slow, fillna=False).ema_indicator()
    signals['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=atr_window, fillna=False).average_true_range()
    # Basic exit trigger via ATR-based level
    signals['atr_sell'] = df['close'] - signals['atr'] * atr_mult
    signals['atr_sell'] = signals['atr_sell'].shift(1)
    signals['atr_sell_signal'] = df['close'] < signals['atr_sell']
    signals['signal'] = (signals['ema_fast'] > signals['ema_slow']).astype(int).diff().fillna(0)
    return signals

# Size a position as a percent of current total value (cash + positions)
def position_sizing(portfolio: Portfolio, price: float, percent_of_value: float = 0.05) -> float:
    total_value = portfolio.get_total_value()
    alloc = total_value * percent_of_value
    max_affordable = portfolio.cash  # already accounts for fees in portfolio methods
    alloc = min(alloc, max_affordable)
    if price <= 0:
        return 0.0
    qty = alloc / price
    return qty

# Main live loop
def main_loop_live(kraken: ccxt.Exchange, portfolio: Portfolio, window_hours: int = 4, use_top_n: int = 20):
    # 1) fetch universe
    universe = build_live_universe(kraken, top_n=use_top_n)
    if not universe:
        print("No universe available for live trading this cycle.")
        return

    # 2) fetch Kraken balance to mirror cash
    try:
        bal = kraken.fetch_balance()
        available_cash = 0.0
        if "USD" in bal["free"]:
            available_cash = float(bal["free"]["USD"])
        elif "USDT" in bal["free"]:
            available_cash = float(bal["free"]["USDT"])  # fallback
        else:
            # pick first fiat-like balance if USD not found
            available_cash = float(next(iter(bal["free"].values())))
        portfolio.cash = available_cash
        print(f"Mirroring Kraken cash: {portfolio.cash:.2f} USD")
    except Exception as e:
        print(f"Warning: could not fetch Kraken balance: {e}")

    # 3) fetch OHLCV for all symbols
    kraken_ohlcv = {}
    for sym in universe:
        pair = to_kraken_pair(sym)
        try:
            df_raw = kraken.fetch_ohlcv(pair, timeframe='4h', limit=200)
            if not df_raw:
                continue
            df = pd.DataFrame(df_raw, columns=["ts", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.set_index("datetime")[["open","high","low","close","volume"]]
            kraken_ohlcv[sym] = df
        except Exception as e:
            print(f"Error fetching OHLCV for {sym}: {e}")

    # 4) compute signals per symbol
    signals = {}
    for sym, df in kraken_ohlcv.items():
        sig = calc_signals_for_ohlcv(df, ema_fast=5, ema_slow=20, atr_window=14, atr_mult=1.0)
        signals[sym] = sig

    # 5) determine actions
    now = datetime.now(timezone.utc)
    window_expire_ts = next_window_expire_ts(now, window_hours)

    # Update and/or place orders
    for sym in universe:
        if sym not in kraken_ohlcv or sym not in signals:
            continue
        df = kraken_ohlcv[sym]
        sig_df = signals[sym]
        if df.empty or sig_df is None or sig_df.empty:
            continue

        latest_date = sig_df.index[-1]
        close_price = float(df.loc[latest_date, "close"])
        signal_val = int(sig_df.loc[latest_date, "signal"]) if "signal" in sig_df else 0
        atr_sell_signal = bool(sig_df.loc[latest_date, "atr_sell_signal"]) if "atr_sell_signal" in sig_df else False

        # If in a position, update price and enforce live ATR stop
        if portfolio.in_position(sym):
            pos = portfolio.get_position(sym)
            if pos:
                pos.update_price(close_price, latest_date)
                # compute live stop and apply
                atr_val = float(sig_df.loc[latest_date, "atr"]) if "atr" in sig_df else 0.0
                live_stop = close_price - (atr_val * 1.5)  # ATR_MULTIPLIER hard-coded as 1.5
                pos.stop_loss = live_stop
                if close_price <= pos.stop_loss or atr_sell_signal:
                    # exit the position
                    portfolio.close_position(pos, date=latest_date)
                    print(f"{latest_date}: EXIT {sym} @ {close_price:.2f} via stop/ATR")
        else:
            if signal_val == 1:
                # size and place a GTD buy order for the next window
                qty = position_sizing(portfolio, close_price, percent_of_value=0.05)
                if qty <= 0:
                    continue
                try:
                    pair = to_kraken_pair(sym)
                    # Kraken GTD: expiretm is UNIX seconds
                    expire_ts = window_expire_ts
                    order = kraken.create_order(
                        symbol=pair,
                        type="limit",
                        side="buy",
                        amount=str(qty),
                        price=str(close_price),
                        params={"timeInForce": "GTD", "expiretm": expire_ts}
                    )
                    # Mirror: create internal Position
                    new_pos = Position(sym, qty, close_price, latest_date, trail_pct=0.0, hold_min=3)
                    portfolio.add_position(new_pos)
                    print(f"{latest_date}: BUY {sym} {qty:.6f} @ {close_price:.2f} (GTd expire {expire_ts})")
                except Exception as e:
                    print(f"Buy GTD failed for {sym}: {e}")

    # 6) record equity for this cycle
    portfolio.record_equity(now)
    print(f"Cycle complete. Total value: {portfolio.get_total_value():.2f} USD")

def main():
    # Initialize Kraken object and mirror cash
    kraken = ccxt.kraken()
    kraken.load_markets()

    # Mirror Kraken cash into our live portfolio
    # We'll fetch balance once at startup to initialize, then keep mirroring each cycle
    try:
        bal = kraken.fetch_balance()
        base_cash = bal.get("free", {}).get("USD", 0.0)
        if base_cash == 0.0:
            # if USD isn't found, try first fiat-like balance
            if bal.get("free"):
                base_cash = float(list(bal["free"].values())[0])
        portfolio = Portfolio(cash=base_cash)
        print(f"Initialized live portfolio with cash: {base_cash:.2f} USD (Kraken balance)")
    except Exception as e:
        print(f"Could not read Kraken balance at startup: {e}. Starting with 0 cash.")
        portfolio = Portfolio(cash=0.0)

    # Run loop
    while True:
        try:
            main_loop_live(kraken, portfolio, window_hours=4, use_top_n=20)
        except Exception as e:
            print("Live loop error:", e)

        # Sleep until next 4-hour boundary with a small buffer
        now = datetime.now(timezone.utc)
        next_bound = now.floor("4H") + pd.Timedelta(hours=4)
        sleep_seconds = max(0, (next_bound - now).total_seconds()) + 60
        time.sleep(sleep_seconds)

if __name__ == "__main__":
    main()
