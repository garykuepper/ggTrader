# language: python
import re
import time
from datetime import datetime, timezone
from typing import Optional
import requests
import pandas as pd

# Map human-friendly intervals to Kraken interval minutes
_INTERVAL_MAP = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "12h": 720,
    "1d": 1440,
}

_KRAKEN_OHLC = "https://api.kraken.com/0/public/OHLC"


def _normalize_symbol_to_kraken(pair: str) -> str:
    """
    Convert yfinance-like symbol (BTC-USD, BTCUSD, BTC/USD) to Kraken pair altname like XBTUSD or ETHUSDT.
    - Uses XBT for BTC (Kraken uses XBT in many endpoints).
    - Preserves USDT if provided (ETHUSDT etc).
    """
    s = re.sub(r"[^A-Za-z0-9]", "", pair).upper()
    # Accept forms like BTCUSD or BTCUSDT etc.
    # Common mapping: BTC -> XBT on Kraken
    if s.startswith("BTC"):
        base = "XBT"
        rest = s[3:]
    else:
        # take letters until USD/USDT/EUR etc
        # very simple handling: assume last 3 or 4 chars are quote
        if len(s) > 6 and s[-4:] in {"USDT", "USDC"}:
            base = s[:-4]
            rest = s[-4:]
        else:
            base = s[:-3]
            rest = s[-3:]
        # If base is already e.g. XETH leave it
    if base == "":
        raise ValueError(f"Cannot parse symbol: {pair}")
    # Replace possible "BTC" leftover
    if base == "BTC":
        base = "XBT"
    # Kraken commonly uses e.g., XBTUSD or ETHUSDT as altname
    return f"{base}{rest}"


def _interval_to_minutes(interval: str) -> int:
    interval = interval.lower()
    if interval not in _INTERVAL_MAP:
        raise ValueError(f"Unsupported interval '{interval}'. Supported: {list(_INTERVAL_MAP.keys())}")
    return _INTERVAL_MAP[interval]


def _request_ohlc(pair: str, interval_min: int, since: Optional[int] = None, timeout: int = 30):
    params = {"pair": pair, "interval": interval_min}
    if since:
        params["since"] = int(since)
    r = requests.get(_KRAKEN_OHLC, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken error: {data['error']}")
    return data["result"]


def _kraken_bars_to_df(bars):
    # Kraken bar format: [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(bars, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    # convert numeric columns
    for col in ["open", "high", "low", "close", "vwap", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("time")
    # keep only OHLCV similar to yfinance and lowercase column names
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def get_kraken_data(symbol: str, interval: str, start_date: datetime, end_date: datetime):
    """
    Fetch OHLC bars from Kraken and return DataFrame similar to yfinance output.
    - symbol: e.g., "BTC-USD", "BTCUSD", "ETH-USD"
    - interval: e.g., "4h", "1h", "1d", "15m"
    - start_date, end_date: timezone-aware or naive datetimes. naive are treated as UTC.
    """
    if start_date.tzinfo is None:
        start_ts = int(start_date.replace(tzinfo=timezone.utc).timestamp())
    else:
        start_ts = int(start_date.astimezone(timezone.utc).timestamp())
    if end_date.tzinfo is None:
        end_ts = int(end_date.replace(tzinfo=timezone.utc).timestamp())
    else:
        end_ts = int(end_date.astimezone(timezone.utc).timestamp())

    print(f"\nDownloading {symbol:10} {interval} {start_date}--->{end_date}", end=" ")

    kraken_pair = _normalize_symbol_to_kraken(symbol)
    interval_min = _interval_to_minutes(interval)

    # Kraken 'since' is inclusive and expressed in seconds (they also sometimes return ms but endpoint expects seconds).
    # We'll page using 'last' until we fetch bars that reach or pass end_date.
    all_bars = []
    since = start_ts
    max_loops = 1000
    loop = 0
    while True:
        loop += 1
        if loop > max_loops:
            raise RuntimeError("Too many paging iterations while fetching Kraken OHLC")
        res = _request_ohlc(kraken_pair, interval_min, since)
        # find OHLC list key (response has pair key + 'last')
        ohlc_key = next((k for k in res.keys() if k != "last"), None)
        if ohlc_key is None:
            break
        bars = res[ohlc_key]
        last = int(res.get("last", 0))
        if not bars:
            # no data returned
            break
        # append and decide if we can stop
        all_bars.extend(bars)
        # If newest returned bar time >= end_ts, stop
        newest_bar_time = int(bars[-1][0])
        if newest_bar_time >= end_ts:
            break
        # otherwise advance since to last + 1
        since = last + 1
        # polite sleep
        time.sleep(0.5)

    if not all_bars:
        print("...No data")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = _kraken_bars_to_df(all_bars)
    # Filter to requested start/end range (inclusive start, exclusive end to match yfinance behavior)
    # Convert end_ts to pandas datetime aware UTC
    start_dt = pd.to_datetime(start_ts, unit="s", utc=True)
    end_dt = pd.to_datetime(end_ts, unit="s", utc=True)
    df = df.loc[(df.index >= start_dt) & (df.index < end_dt)]
    # yfinance returns columns lowercased already; we've done that
    entries = len(df)
    print(f"...Complete. {entries} dates", end=" ")
    return df
