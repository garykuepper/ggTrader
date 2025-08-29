# language: python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Set

import warnings
import pandas as pd
import requests
import yfinance as yf

# -------------------------
# Module constants / maps
# -------------------------
UNIFIED_COLS = ["open", "high", "low", "close", "volume", "vwap", "trades"]

# Canonical per-provider supported intervals (lowercased)
YF_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "4h",
    "1d", "5d", "1wk", "1mo", "3mo"
}

KRAKEN_INTERVAL_MIN = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "60m": 60, "1h": 60, "4h": 240,
    "1d": 1440, "1w": 10080, "15d": 21600,
}

KRAKEN_MAX_BARS = 720  # Kraken public OHLC endpoints return at most ~720 rows


# ===============================
# Abstract base
# ===============================
class DataProvider(ABC):
    """
    Abstract market data provider.

    Implementations must return a pandas DataFrame with:
      - UTC DatetimeIndex named 'datetime'
      - columns: ["open","high","low","close","volume","vwap","trades"]
    """

    _UNIFIED_COLS = UNIFIED_COLS

    @abstractmethod
    def get_data(self,
                 symbol: str,
                 interval: str,
                 start_date: datetime,
                 end_date: datetime) -> pd.DataFrame:
        ...

    # ---- shared utility methods ----
    @staticmethod
    def _to_utc(dt: datetime) -> datetime:
        """
        Return a timezone-aware UTC datetime. Naive datetimes are assumed UTC.
        """
        if dt is None:
            raise ValueError("dt must be provided")
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _interval_to_timedelta(interval: str) -> timedelta:
        """
        Convert human interval strings like "4h", "15m", "1d", "1wk" to timedelta.
        """
        i = interval.lower()
        if i.endswith("m"):
            return timedelta(minutes=int(i[:-1]))
        if i.endswith("h"):
            return timedelta(hours=int(i[:-1]))
        if i.endswith("d"):
            return timedelta(days=int(i[:-1]))
        if i.endswith("w"):
            return timedelta(weeks=int(i[:-1]))
        if i == "1wk":
            return timedelta(weeks=1)
        if i == "1mo":
            return timedelta(days=30)
        if i == "3mo":
            return timedelta(days=90)
        raise ValueError(f"Unsupported interval: {interval}")

    def _drop_current_partial_bar(self, df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """
        Drop the last bar if it’s still forming (avoid look-ahead bias).
        Assumes right-labeled bars (index == bar end time).
        """
        if df.empty:
            return df
        last_ts = df.index[-1]
        now_utc = datetime.now(timezone.utc)
        if last_ts > now_utc or (now_utc - last_ts) < self._interval_to_timedelta(interval):
            return df.iloc[:-1]
        return df

    def _ensure_unified_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in self._UNIFIED_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        # Return columns in canonical order
        return df[self._UNIFIED_COLS].sort_index()


# ===============================
# yfinance provider (deep history)
# ===============================
@dataclass
class YFinanceProvider(DataProvider):
    """
    yfinance-backed provider (great for deep history/backtests).
    """
    _YF_INTERVALS: Set[str] = field(default_factory=lambda: set(YF_INTERVALS))

    def get_data(self,
                 symbol: str,
                 interval: str,
                 start_date: datetime,
                 end_date: datetime) -> pd.DataFrame:
        if interval not in self._YF_INTERVALS:
            raise ValueError(f"[yfinance] Unsupported interval '{interval}'. Supported: {sorted(self._YF_INTERVALS)}")

        start_dt = self._to_utc(start_date)
        end_dt = self._to_utc(end_date)

        print(f"\n[yf]  {symbol:10} {interval} {start_dt:%Y-%m-%d-%H}--->{end_dt:%Y-%m-%d-%H}", end=" ")

        df = yf.download(
            symbol,
            interval=interval,
            start=start_dt,
            end=end_dt,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )

        if df is None or df.empty:
            print("...0 rows", end=" ")
            return pd.DataFrame(columns=self._UNIFIED_COLS).set_index(pd.DatetimeIndex([], name="datetime"))

        # Normalize columns & timezone
        df.columns = df.columns.str.lower()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "datetime"

        if "adj close" in df.columns:
            df = df.drop(columns=["adj close"])

        # Fill missing provider-specific columns
        if "vwap" not in df.columns:
            df["vwap"] = pd.NA
        if "trades" not in df.columns:
            df["trades"] = pd.NA

        df = self._ensure_unified_schema(df)
        # ensure requested bounds
        df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
        df = self._drop_current_partial_bar(df, interval)

        print(f"...{len(df)} rows", end=" ")
        return df


# ===============================
# Kraken provider (recent/live, ~720-bar cap)
# ===============================
@dataclass
class KrakenProvider(DataProvider):
    """
    Kraken public OHLC provider (fast recent bars; capped to ~720 most recent per interval).
    """
    asset_class: Optional[str] = None
    _KRAKEN_INTERVAL_MIN: Dict[str, int] = field(default_factory=lambda: dict(KRAKEN_INTERVAL_MIN))

    def get_data(self,
                 symbol: str,
                 interval: str,
                 start_date: datetime,
                 end_date: datetime) -> pd.DataFrame:
        key = interval.lower()
        if key not in self._KRAKEN_INTERVAL_MIN:
            raise ValueError(f"[kraken] Unsupported interval '{interval}'. Choose from: {sorted(self._KRAKEN_INTERVAL_MIN)}")

        pair = self._to_pair(symbol)
        start_dt = self._to_utc(start_date)
        end_dt = self._to_utc(end_date)

        print(f"\n[kr]  {symbol:10} {interval} {start_dt:%Y-%m-%d-%H}--->{end_dt:%Y-%m-%d-%H}", end=" ")

        # Detect expected number of bars and warn/trim if it exceeds Kraken's limit
        interval_minutes = self._KRAKEN_INTERVAL_MIN[key]
        interval_seconds = interval_minutes * 60
        span_seconds = max(1, int((end_dt - start_dt).total_seconds()))
        expected_bars = (span_seconds + interval_seconds - 1) // interval_seconds  # ceil
        if expected_bars > KRAKEN_MAX_BARS:
            new_start_dt = end_dt - timedelta(seconds=KRAKEN_MAX_BARS * interval_seconds)
            msg = (f"[kr][WARN] Requested span would produce ~{expected_bars} bars; Kraken returns ~{KRAKEN_MAX_BARS}. "
                   f"Truncating start to {new_start_dt.isoformat()} to fetch the most recent {KRAKEN_MAX_BARS} bars.")
            warnings.warn(msg, UserWarning, stacklevel=2)
            start_dt = new_start_dt

        params = {"pair": pair, "interval": self._KRAKEN_INTERVAL_MIN[key]}
        if self.asset_class:
            params["asset_class"] = self.asset_class

        r = requests.get("https://api.kraken.com/0/public/OHLC", params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if payload.get("error"):
            raise RuntimeError(f"Kraken API error: {payload['error']}")

        result = payload.get("result", {})
        keys = [k for k in result.keys() if k != "last"]
        if not keys:
            print("...0 rows", end=" ")
            return pd.DataFrame(columns=self._UNIFIED_COLS).set_index(pd.DatetimeIndex([], name="datetime"))

        raw = result[keys[0]]

        start_unix = int(start_dt.timestamp())
        end_unix = int(end_dt.timestamp())

        rows = []
        for row in raw:
            # row: [time, open, high, low, close, vwap, volume, count]
            ts = int(row[0])
            if ts < start_unix or ts > end_unix:
                continue
            rows.append({
                "datetime": datetime.fromtimestamp(ts, tz=timezone.utc),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "vwap": float(row[5]),
                "volume": float(row[6]),
                "trades": int(row[7]),
            })

        df = pd.DataFrame(rows).set_index("datetime").sort_index()
        df.index.name = "datetime"

        df = self._ensure_unified_schema(df)
        df = self._drop_current_partial_bar(df, interval)

        print(f"...{len(df)} rows", end=" ")
        return df

    @staticmethod
    def _to_pair(symbol: str) -> str:
        s = symbol.upper().replace("-", "").replace("/", "")
        s = s.replace("BTC", "XBT")
        if s.endswith("USDT"):
            s = s[:-4] + "USD"
        if not s.endswith("USD"):
            s += "USD"
        return s


# ===============================
# Hybrid provider (composition)
# ===============================
# python
@dataclass
class HybridProvider(DataProvider):
    """
    Convenience wrapper:
      - mode="auto": choose provider by span (if Kraken can cover the requested range use Kraken,
                    otherwise use yfinance)
      - mode="backtest": force yfinance
      - mode="live":     force Kraken
    Selection in "auto" now takes the interval density into account: e.g. "4h" -> 6 bars/day,
    with Kraken's ~720-bar cap that covers ~120 days at 4h resolution.
    """
    mode: str = "auto"
    yf_provider: YFinanceProvider = field(default_factory=YFinanceProvider)
    kr_provider: KrakenProvider = field(default_factory=KrakenProvider)

    def get_data(self,
                 symbol: str,
                 interval: str,
                 start_date: datetime,
                 end_date: datetime) -> pd.DataFrame:
        start_dt = self._to_utc(start_date)
        end_dt = self._to_utc(end_date)

        mode = self.mode
        if mode == "auto":
            # Default fallback: day-based rule (keeps prior behavior if interval parsing fails)
            span_days = max(1, (end_dt - start_dt).days)

            # Try to compute bars/day from interval (e.g. "4h" -> 6 bars/day)
            try:
                interval_td = self._interval_to_timedelta(interval)
            except Exception:
                # Unsupported interval string -> fall back to simple day-based threshold (20 days)
                mode = "backtest" if span_days > 20 else "live"
            else:
                interval_seconds = max(1, int(interval_td.total_seconds()))
                bars_per_day = max(1, 86400 // interval_seconds)

                # Resolve Kraken max bars constant (prefer module/global definition if available)
                KRAKEN_MAX = globals().get("KRAKEN_MAX_BARS", 720)

                # How many days of history Kraken can provide for this interval
                max_days_for_kraken = KRAKEN_MAX / float(bars_per_day)

                # Choose provider: if requested span exceeds Kraken window, prefer yfinance/backtest
                mode = "backtest" if span_days > max_days_for_kraken else "live"

        if mode == "backtest":
            return self.yf_provider.get_data(symbol, interval, start_dt, end_dt)
        elif mode == "live":
            return self.kr_provider.get_data(symbol, interval, start_dt, end_dt)
        else:
            raise ValueError("mode must be one of {'auto','backtest','live'}")


# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    end = datetime.now(timezone.utc)
    start_long = end - timedelta(days=365)   # backtest -> yfinance
    start_short = end - timedelta(days=7)    # live -> kraken

    # Direct providers
    yf_dp = YFinanceProvider()
    kr_dp = KrakenProvider()

    df_yf = yf_dp.get_data("BTC-USD", "4h", start_long, end)
    print("\nYF:", df_yf.shape, df_yf.index.min(), "→", df_yf.index.max())

    df_kr = kr_dp.get_data("BTC-USD", "4h", start_short, end)
    print("\nKR:", df_kr.shape, df_kr.index.min(), "→", df_kr.index.max())

    # Hybrid/autoselect
    auto_dp = HybridProvider(mode="auto")
    df_auto = auto_dp.get_data("BTC-USD", "4h", start_short, end)
    print("\nAUTO:", df_auto.shape, df_auto.index.min(), "→", df_auto.index.max())