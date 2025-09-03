#!/usr/bin/env python3
"""
live_trader_kraken.py — Live LIMIT buys on Kraken with GTD expiry = 50% of your analysis interval.

- Universe: utils.kraken_yfinance_cmc.get_top_kraken_usd_pairs (Yahoo for signals, Kraken pairs for orders)
- Signals: EMA(30/84) on last completed N-hour bar (N = INTERVAL_HOURS)
- Entry: fresh +1 cross OR bullish regime (fast>slow) if BUY_ON_BULL_REGIME=1
- Sizing: POSITION_PCT * equity_usd (default 10%), capped by CASH_UTILIZATION * cash_usd
- Order: LIMIT + GTD, expires at now + (INTERVAL_HOURS / 2)
- Excludes stablecoins (USDT, USDC, DAI, TUSD, USDP, FDUSD, PYUSD, GUSD, USDD, USDE, USDX, BUSD, EURT, EURS, USDG)

This script runs once. Schedule it externally (cron/systemd or a Docker loop).
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator

from utils.kraken_yfinance_cmc import get_top_kraken_usd_pairs
from utils.DataProvider import HybridProvider

# ================= Config via ENV =================
INTERVAL_HOURS    = int(os.getenv("INTERVAL_HOURS", "4"))   # candle interval & scheduling cadence
TOP_N             = int(os.getenv("TOP_N", "20"))
LOOKBACK_DAYS     = int(os.getenv("LOOKBACK_DAYS", "120"))
BUY_ON_BULL       = bool(int(os.getenv("BUY_ON_BULL_REGIME", "1")))

POSITION_PCT      = float(os.getenv("POSITION_PCT", "0.10"))    # 10% of equity per entry
CASH_UTILIZATION  = float(os.getenv("CASH_UTILIZATION", "1.00"))# cap notional by available cash
POST_ONLY         = bool(int(os.getenv("POST_ONLY", "0")))      # 1 = maker-only

KRAKEN_KEY        = os.getenv("KRAKEN_KEY")
KRAKEN_SECRET     = os.getenv("KRAKEN_SECRET")

# ================= Exclusions =================
STABLE_BLACKLIST = {
    "USDT-USD", "USDC-USD", "DAI-USD", "USDP-USD", "TUSD-USD",
    "FDUSD-USD", "PYUSD-USD", "GUSD-USD", "USDD-USD",
    "USDX-USD", "USDE-USD", "BUSD-USD", "EURT-USD", "EURS-USD",
    "USDG-USD"
}

# ================= Time helpers =================
def last_completed_bar_end(now_utc: datetime, interval_hours: int) -> datetime:
    """
    Return the last completed N-hour bar end in UTC.
    Example: now=10:07, N=4 -> returns 08:00.
    """
    now_utc = now_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    k = interval_hours
    floored = (now_utc.hour // k) * k
    end = now_utc.replace(hour=floored)
    if end == now_utc:
        end -= timedelta(hours=k)
    return end

# ================= Signals =================
class EMASignal:
    def __init__(self, fast=30, slow=84):
        self.fast, self.slow = fast, slow

    def _emas(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        ema_f = EMAIndicator(close, self.fast, fillna=False).ema_indicator()
        ema_s = EMAIndicator(close, self.slow, fillna=False).ema_indicator()
        return ema_f, ema_s

    def last_signal(self, close: pd.Series) -> int:
        ema_f, ema_s = self._emas(close)
        bull = (ema_f > ema_s).astype(int)
        x = bull.diff()
        v = x.iloc[-1]
        return int(v) if v in (1, -1) else 0

    def is_bull_now(self, close: pd.Series) -> int:
        ema_f, ema_s = self._emas(close)
        return int((ema_f > ema_s).iloc[-1])

# ================= Kraken broker =================
class KrakenBroker:
    """
    Live Kraken spot wrapper:
    - Equity & cash in USD (marks balances with <ASSET>USD last price when available)
    - Places LIMIT + GTD orders that expire at now + (INTERVAL_HOURS / 2)
    """
    def __init__(self, api_key: str, api_secret: str):
        from kraken.spot import Trade, Market  # lazy import to avoid hard dependency in tests
        self.trade = Trade(key=api_key, secret=api_secret)
        self.mkt = Market()
        self._pair_meta: Dict[str, Dict[str, Any]] = {}

    # ----- symbol helpers -----
    @staticmethod
    def _strip_xz(asset: str) -> str:
        a = asset.upper()
        if len(a) >= 4 and a[0] in "XZ":
            return a[1:]
        return a

    @staticmethod
    def _alias_common(base: str) -> str:
        if base == "XBT": return "BTC"
        if base == "XDG": return "DOGE"
        return base

    def _common_base(self, asset: str) -> str:
        return self._alias_common(self._strip_xz(asset))

    # ----- pair metadata / precision -----
    def _get_pair_meta(self, pair: str) -> Dict[str, Any]:
        if pair not in self._pair_meta:
            data = self.mkt.get_tradable_asset_pairs(pair=pair)
            res = data.get("result") or {}
            meta = next(iter(res.values())) if res else {}
            if not meta:
                raise ValueError(f"Unknown pair {pair}")
            self._pair_meta[pair] = meta
        return self._pair_meta[pair]

    def _round_px_qty(self, pair: str, price: float, qty: float) -> Tuple[float, float]:
        meta = self._get_pair_meta(pair)
        p_dec = int(meta.get("pair_decimals", 5))
        q_dec = int(meta.get("lot_decimals", 8))
        rp = float(f"{price:.{p_dec}f}")
        rq = float(f"{qty:.{q_dec}f}")
        return rp, rq

    def _check_mins(self, pair: str, price: float, qty: float) -> None:
        meta = self._get_pair_meta(pair)
        ordermin = float(meta.get("ordermin", 0)) if meta.get("ordermin") else 0.0
        costmin = float(meta.get("costmin", 0)) if meta.get("costmin") else 0.0
        notional = price * qty
        if ordermin and qty + 1e-15 < ordermin:
            raise ValueError(f"qty {qty} < ordermin {ordermin} for {pair}")
        if costmin and notional + 1e-12 < costmin:
            raise ValueError(f"notional ${notional:.2f} < costmin ${costmin:.2f} for {pair}")

    # ----- prices / equity -----
    def _last_price_usd(self, base_common: str) -> Optional[float]:
        pair = f"{base_common}USD"
        try:
            tick = self.mkt.get_ticker_information(pair=pair)
            res = tick.get("result") or {}
            meta = next(iter(res.values())) if res else {}
            if not meta:
                return None
            return float(meta.get("c", ["0", "0"])[0]) or None
        except Exception:
            return None

    def get_equity_and_cash_usd(self, include_usd_stables: bool = True) -> Tuple[float, float, Dict[str, float]]:
        """
        equity_usd = cash_usd + Σ(other_asset_qty * last_price_usd)
        cash_usd   = ZUSD + (optional) USD stables (assumed 1:1)
        Returns (equity_usd, cash_usd, balances_map)
        """
        bals = self.trade.get_account_balance()
        result = bals.get("result", bals)
        qty_map: Dict[str, float] = {}
        for asset, amt in result.items():
            try:
                q = float(amt)
            except Exception:
                continue
            if q > 0:
                qty_map[asset] = q

        cash_usd = float(qty_map.get("ZUSD", 0.0))
        if include_usd_stables:
            for a in ("USDT", "USDC", "DAI", "USDP", "TUSD", "FDUSD", "PYUSD", "GUSD"):
                cash_usd += float(qty_map.get(a, 0.0))

        equity_usd = cash_usd
        for asset, qty in qty_map.items():
            if asset in ("ZUSD", "USDT", "USDC", "DAI", "USDP", "TUSD", "FDUSD", "PYUSD", "GUSD"):
                continue
            base = self._common_base(asset)
            px = self._last_price_usd(base)
            if px:
                equity_usd += qty * px

        return equity_usd, cash_usd, qty_map

    # ----- orders -----
    @staticmethod
    def _expire_unix_in_half_interval(interval_hours: int) -> int:
        expire_dt = datetime.now(timezone.utc) + timedelta(hours=interval_hours / 2.0)
        return int(expire_dt.timestamp())

    def limit_buy_gtd(self, pair: str, qty: float, limit_price: float,
                      interval_hours: int, client_id: Optional[str] = None,
                      post_only: bool = False) -> str:
        """
        GTD expiry = now + interval_hours/2  (e.g., 4h interval -> 2h GTD)
        """
        price, qty = self._round_px_qty(pair, limit_price, qty)
        self._check_mins(pair, price, qty)
        extra = {
            "ordertype": "limit",
            "timeinforce": "GTD",
            "expiretm": self._expire_unix_in_half_interval(interval_hours),
        }
        if post_only:
            extra["oflags"] = "post"
        if client_id:
            extra["cl_ord_id"] = client_id

        resp = self.trade.add_order(
            pair=pair,
            side="buy",
            ordertype="limit",
            price=str(price),
            volume=str(qty),
            extra_params=extra
        )
        tx = resp.get("txid") or resp.get("result", {}).get("txid")
        return tx[0] if isinstance(tx, list) else str(tx)

# ================= Utilities =================
def map_row_to_symbols(row: pd.Series) -> Tuple[str, str]:
    """Return (yf_symbol, kraken_pair) from a universe row."""
    return str(row.get("YF Ticker", "-")), str(row.get("Kraken Pair", ""))

def looks_stable_like(yf_symbol: str) -> bool:
    return yf_symbol in STABLE_BLACKLIST

# ================= Main (one-shot) =================
def run_once():
    if not KRAKEN_KEY or not KRAKEN_SECRET:
        raise RuntimeError("Set KRAKEN_KEY and KRAKEN_SECRET environment variables.")

    interval_str = f"{INTERVAL_HOURS}h"
    provider = HybridProvider()
    engine = EMASignal()
    broker = KrakenBroker(KRAKEN_KEY, KRAKEN_SECRET)

    now = datetime.now(timezone.utc)
    bar_end = last_completed_bar_end(now, INTERVAL_HOURS)

    # Equity & cash
    equity_usd, cash_usd, _ = broker.get_equity_and_cash_usd(include_usd_stables=False)
    target_notional = POSITION_PCT * equity_usd
    remaining_cash_cap = CASH_UTILIZATION * cash_usd
    print(f"[ACCT] equity=${equity_usd:,.2f} cash=${cash_usd:,.2f} "
          f"target/pos={POSITION_PCT:.2%} → ${target_notional:,.2f} "
          f"(cash cap=${remaining_cash_cap:,.2f})")

    if remaining_cash_cap <= 0:
        print("[ACCT] No available cash; skipping.")
        return

    # Universe (oversample to survive filters)
    df_uni = get_top_kraken_usd_pairs(top_n=max(30, TOP_N), require_yf=False)

    entries = 0
    for _, row in df_uni.iterrows():
        yf_sym, kr_pair = map_row_to_symbols(row)
        if yf_sym == "-" or not kr_pair or looks_stable_like(yf_sym):
            continue

        # Data for signals
        start_dt = bar_end - timedelta(days=LOOKBACK_DAYS)
        try:
            df = provider.get_data(yf_sym, interval_str, start_dt, bar_end)
        except Exception as e:
            print(f"[WARN] data fetch failed for {yf_sym}: {e}")
            continue
        if df is None or df.empty or "close" not in df.columns:
            continue

        ts = bar_end if bar_end in df.index else df.index[-1]
        close = float(df.loc[ts, "close"])

        sig  = engine.last_signal(df["close"])
        bull = engine.is_bull_now(df["close"]) if BUY_ON_BULL else 0
        if not ((sig == 1) or (BUY_ON_BULL and bull == 1)):
            continue

        # Per-trade notional: 10% of equity, capped by remaining cash
        notional = min(target_notional, remaining_cash_cap)
        if notional <= 0:
            break

        qty = notional / close
        if qty <= 0:
            continue

        try:
            txid = broker.limit_buy_gtd(
                pair=kr_pair,
                qty=qty,
                limit_price=close,                 # optionally discount for maker bias (e.g., close*0.998)
                interval_hours=INTERVAL_HOURS,     # GTD = now + interval/2
                client_id=f"{kr_pair}-{int(time.time())}",
                post_only=POST_ONLY
            )
            print(f"[LIVE BUY] {kr_pair} ${notional:,.2f} qty={qty:.8f} @ {close:.8f} "
                  f"GTD=now+{INTERVAL_HOURS/2:.1f}h txid={txid}")
            entries += 1
            remaining_cash_cap -= notional
            if remaining_cash_cap <= 0 or entries >= TOP_N:
                break
        except Exception as e:
            print(f"[ERROR] buy {kr_pair} failed: {e}")

    print(f"Done. Placed {entries} GTD limit buy(s).")

if __name__ == "__main__":
    run_once()
