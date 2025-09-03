"""
paper_trader.py — JSON-only 4h paper trader with EMA signals, trailing stop (hold-min), cooldown,
and robust universe management (always manage open positions even if they fall out of Top-N).

Dependencies (see requirements.txt):
  pandas, numpy, ta, yfinance, python-dateutil, pytz, requests

Project assumptions:
  - utils.DataProvider.HybridProvider.get_data(symbol, interval, start_dt, end_dt) -> DataFrame with index as timestamps
    and columns at least: ["open","high","low","close","volume"] (close required)
  - utils.kraken_yfinance_cmc.get_top_kraken_usd_pairs(top_n, require_yf=True) -> DataFrame with column "YF Ticker"

State:
  - JSON at STATE_PATH (default ./state/paper_state.json)
"""

import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator

from utils.DataProvider import HybridProvider
from utils.kraken_yfinance_cmc import get_top_kraken_usd_pairs


# ============================ Helpers & Metrics ============================

FOUR_HOURS = timedelta(hours=4)

def nearest_4h_bar_end(dt_utc: datetime) -> datetime:
    """Floor to the current 4h boundary (UTC), then subtract 4h to get the LAST completed bar end."""
    dt_utc = dt_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    floored_hour = (dt_utc.hour // 4) * 4
    bar_end = dt_utc.replace(hour=floored_hour)
    if bar_end == dt_utc:
        bar_end -= FOUR_HOURS
    return bar_end

def periods_per_year_for_4h() -> int:
    return 6 * 365  # 6 bars/day

def sharpe_from_equity(equity_curve: pd.Series, periods_per_year=2190, rf_annual=0.01) -> float:
    if equity_curve is None or len(equity_curve) < 3:
        return 0.0
    rets = np.log(equity_curve / equity_curve.shift(1)).dropna()
    if rets.empty:
        return 0.0
    rf_per_period = (1 + rf_annual) ** (1 / periods_per_year) - 1
    excess = rets - rf_per_period
    mu, sigma = excess.mean(), excess.std(ddof=1)
    return 0.0 if sigma == 0 or np.isnan(sigma) else float((mu / sigma) * np.sqrt(periods_per_year))


# ============================ State (JSON only) ============================

class PaperStateStore:
    def __init__(self, json_path: Optional[str] = None):
        self.json_path = json_path or os.getenv("STATE_PATH", "./state/paper_state.json")

    def load(self) -> Dict[str, Any]:
        default = {
            "cash": 10_000.0,
            "transaction_fee": 0.004,     # 0.4%
            "positions": {},              # sym -> {qty, entry_price, current_price, entry_ts_iso, trailing}
            "next_entry_time": {},        # sym -> iso timestamp
            "equity_curve": [],           # [{ts, equity}]
            "trades": []                  # [{symbol, side, qty, price, ts, fees}]
        }
        try:
            with open(self.json_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
            return default

    def save(self, state: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        with open(self.json_path, "w") as f:
            json.dump(state, f, indent=2, default=str)


# ============================ Broker (paper fills) ============================

class PaperBroker:
    def __init__(self, state: Dict[str, Any]):
        self.state = state

    @property
    def cash(self) -> float:
        return float(self.state.get("cash", 0.0))

    def _fee(self, notional: float) -> float:
        return notional * float(self.state.get("transaction_fee", 0.0))

    def in_position(self, symbol: str) -> bool:
        return symbol in self.state["positions"]

    def pos(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.state["positions"].get(symbol)

    def _ts_iso(self, ts: datetime) -> str:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc).isoformat()

    def buy(self, symbol: str, qty: float, price: float, ts: datetime, trail_pct: float, hold_min: int) -> None:
        notional = qty * price
        fee = self._fee(notional)
        total_cost = notional + fee
        if total_cost > self.cash + 1e-9:
            print(f"[WARN] Not enough cash to buy {symbol}: need {total_cost:.2f}, have {self.cash:.2f}")
            return

        self.state["cash"] -= total_cost
        trailing = {
            "type": "FixedPct",
            "pct": float(trail_pct),
            "level": price * (1 - trail_pct / 100.0),
            "hold_min": int(hold_min),
            "consec_hits": 0,
            "triggered": False,
            "last_update_ts": self._ts_iso(ts),
        }
        self.state["positions"][symbol] = {
            "qty": float(qty),
            "entry_price": float(price),
            "current_price": float(price),
            "entry_ts_iso": self._ts_iso(ts),
            "trailing": trailing,
        }
        self.state["trades"].append({
            "symbol": symbol, "side": "BUY", "qty": float(qty), "price": float(price),
            "ts": self._ts_iso(ts), "fees": float(fee)
        })

    def sell(self, symbol: str, price: float, ts: datetime) -> None:
        pos = self.pos(symbol)
        if not pos:
            return
        qty = float(pos["qty"])
        notional = qty * price
        fee = self._fee(notional)
        proceeds = notional - fee
        self.state["cash"] += proceeds
        self.state["trades"].append({
            "symbol": symbol, "side": "SELL", "qty": float(qty), "price": float(price),
            "ts": self._ts_iso(ts), "fees": float(fee)
        })
        del self.state["positions"][symbol]

    def update_mark(self, symbol: str, price: float) -> None:
        pos = self.pos(symbol)
        if pos:
            pos["current_price"] = float(price)


# ============================ Signals ============================

class EMASignalEngine:
    def __init__(self, fast=30, slow=84):
        self.fast = fast
        self.slow = slow

    def last_signal(self, df: pd.DataFrame) -> int:
        ema_fast = EMAIndicator(df["close"], self.fast, fillna=False).ema_indicator()
        ema_slow = EMAIndicator(df["close"], self.slow, fillna=False).ema_indicator()
        bull = (ema_fast > ema_slow).astype(int)
        cross = bull.diff()
        val = cross.iloc[-1]
        return int(val) if val in (1, -1) else 0


# ============================ Runner ============================

class Runner:
    def __init__(
            self,
            interval: str = "4h",
            lookback_days: int = 130,
            fast_window: int = 30,
            slow_window: int = 84,
            top_n: int = 20,
            position_share_pct: float = 10.0,    # % of equity per new entry
            trail_pct: float = 4.5,              # trailing stop percent
            hold_min_bars: int = 3,              # consecutive bars under stop to trigger exit
            cooldown_bars: int = 8               # bars to wait after any exit before new entry
    ):
        self.interval = interval
        self.lookback_days = lookback_days
        self.top_n = top_n
        self.position_share_pct = float(position_share_pct)
        self.trail_pct = float(trail_pct)
        self.hold_min_bars = int(hold_min_bars)
        self.cooldown_bars = int(cooldown_bars)

        self.store = PaperStateStore()
        self.state = self.store.load()
        self.broker = PaperBroker(self.state)
        self.engine = EMASignalEngine(fast_window, slow_window)
        self.provider = HybridProvider()

        self.bar_delta = FOUR_HOURS

    # ---------- Universe ----------

    def build_universe(self) -> Tuple[List[str], List[str], List[str]]:
        """Returns (top_base, open_syms, manage_set)."""
        symbols_df = get_top_kraken_usd_pairs(top_n=max(30, self.top_n), require_yf=True)
        top_base = symbols_df["YF Ticker"].head(self.top_n).tolist()
        open_syms = list(self.state["positions"].keys())
        manage_set = sorted(set(top_base) | set(open_syms))
        return top_base, open_syms, manage_set

    # ---------- Cooldown ----------

    def cooldown_ok(self, symbol: str, bar_ts: datetime) -> bool:
        next_iso = self.state["next_entry_time"].get(symbol)
        if not next_iso:
            return True
        try:
            next_allowed = datetime.fromisoformat(next_iso)
        except Exception:
            return True
        return bar_ts >= next_allowed

    def set_cooldown(self, symbol: str, bar_ts: datetime) -> None:
        next_allowed = bar_ts + self.bar_delta * self.cooldown_bars
        self.state["next_entry_time"][symbol] = next_allowed.astimezone(timezone.utc).isoformat()

    # ---------- Trailing stop ----------

    def update_trailing_stop_and_maybe_exit(self, symbol: str, price: float, bar_ts: datetime) -> bool:
        """Ratchet trailing stop up. Count consecutive hits; exit if >= hold_min. Returns True if exited."""
        pos = self.broker.pos(symbol)
        if not pos:
            return False
        tr = pos["trailing"]

        candidate = price * (1 - self.trail_pct / 100.0)
        if tr["level"] is None or candidate > tr["level"]:
            tr["level"] = float(candidate)

        if price <= float(tr["level"]):
            tr["consec_hits"] = int(tr.get("consec_hits", 0)) + 1
        else:
            tr["consec_hits"] = 0

        if tr["consec_hits"] >= int(tr.get("hold_min", self.hold_min_bars)):
            tr["triggered"] = True
            print(f"[EXIT-TS] {symbol} stop={tr['level']:.6f} price={price:.6f} hits={tr['consec_hits']}")
            self.broker.sell(symbol, price, bar_ts)
            self.set_cooldown(symbol, bar_ts)
            return True

        tr["last_update_ts"] = bar_ts.astimezone(timezone.utc).isoformat()
        return False

    # ---------- Data / Signals ----------

    def fetch_symbol_df(self, symbol: str, start_dt: datetime, bar_ts: datetime) -> Optional[pd.DataFrame]:
        try:
            df = self.provider.get_data(symbol, self.interval, start_dt, bar_ts)
        except Exception as e:
            print(f"[WARN] data fetch failed for {symbol}: {e}")
            return None
        if df is None or df.empty or "close" not in df.columns:
            return None
        return df

    def latest_bar_timestamp(self, df: pd.DataFrame, bar_ts: datetime) -> datetime:
        return df.index[-1] if bar_ts not in df.index else bar_ts

    def last_signal(self, df: pd.DataFrame) -> int:
        try:
            return self.engine.last_signal(df)
        except Exception as e:
            print(f"[WARN] signal failed: {e}")
            return 0

    # ---------- Sizing / Accounting ----------

    def total_equity(self) -> float:
        equity = self.broker.cash
        for pos in self.state["positions"].values():
            equity += float(pos["qty"]) * float(pos.get("current_price", pos["entry_price"]))
        return float(equity)

    def qty_for_price(self, price: float) -> float:
        equity = self.total_equity()
        target_notional = (self.position_share_pct / 100.0) * equity
        fee_mult = 1.0 + float(self.state.get("transaction_fee", 0.0))
        budget = min(self.broker.cash, target_notional) / fee_mult
        return 0.0 if price <= 0 else budget / price

    def record_equity(self, bar_ts: datetime) -> Tuple[float, float]:
        equity = self.total_equity()
        self.state["equity_curve"].append({"ts": bar_ts.astimezone(timezone.utc).isoformat(), "equity": equity})
        ec = pd.Series({pd.Timestamp(e["ts"]): e["equity"] for e in self.state["equity_curve"]}).sort_index()
        sharpe = sharpe_from_equity(ec, periods_per_year=periods_per_year_for_4h(), rf_annual=0.01)
        return equity, sharpe

    # ---------- Per-symbol processing ----------

    def handle_open_position(self, symbol: str, close: float, bar_ts_df: datetime) -> bool:
        """Update marks and trailing stop; return True if exited (so we should skip further actions)."""
        self.broker.update_mark(symbol, close)
        return self.update_trailing_stop_and_maybe_exit(symbol, close, bar_ts_df)

    def maybe_exit_on_signal(self, symbol: str, sig: int, close: float, bar_ts_df: datetime) -> bool:
        """Process -1 exit signal; return True if exited."""
        if sig == -1 and self.broker.in_position(symbol):
            print(f"[EXIT]  {symbol} @ {close:.6f} ({bar_ts_df})")
            self.broker.sell(symbol, close, bar_ts_df)
            self.set_cooldown(symbol, bar_ts_df)
            return True
        return False

    def maybe_enter_on_signal(self, symbol: str, sig: int, close: float, bar_ts_df: datetime, top_base: List[str]) -> bool:
        """Process +1 entry; return True if entered."""
        if (
                sig == 1
                and not self.broker.in_position(symbol)
                and symbol in top_base
                and self.cooldown_ok(symbol, bar_ts_df)
        ):
            qty = self.qty_for_price(close)
            if qty > 0:
                print(f"[ENTER] {symbol} qty={qty:.6f} @ {close:.6f} ({bar_ts_df})")
                self.broker.buy(symbol, qty, close, bar_ts_df, trail_pct=self.trail_pct, hold_min=self.hold_min_bars)
                return True
        return False

    # ---------- Main ----------

    def run_once(self) -> None:
        bar_ts = nearest_4h_bar_end(datetime.now(timezone.utc))
        start_dt = bar_ts - timedelta(days=self.lookback_days)

        # Universe
        try:
            top_base, open_syms, symbols = self.build_universe()
        except Exception as e:
            print(f"[WARN] Could not build universe: {e}")
            return
        if not symbols:
            print("[WARN] No symbols to trade or manage.")
            return

        entries = exits = 0

        for sym in symbols:
            df = self.fetch_symbol_df(sym, start_dt, bar_ts)
            if df is None:
                continue

            bar_ts_df = self.latest_bar_timestamp(df, bar_ts)
            close = float(df.loc[bar_ts_df, "close"])

            # Manage open positions first
            if self.broker.in_position(sym):
                if self.handle_open_position(sym, close, bar_ts_df):
                    exits += 1
                    continue  # already exited on trailing stop

            sig = self.last_signal(df)

            # Exit on -1 signal
            if self.maybe_exit_on_signal(sym, sig, close, bar_ts_df):
                exits += 1
                continue

            # Enter on +1 signal (restricted to Top-N and cooldown OK)
            if self.maybe_enter_on_signal(sym, sig, close, bar_ts_df, top_base):
                entries += 1

        # Accounting
        equity, sharpe = self.record_equity(bar_ts)
        print(f"\nEquity=${equity:,.2f} | Open={len(self.state['positions'])} | Entries={entries} | Exits={exits} | Sharpe={sharpe:.3f}")

        # Persist
        self.store.save(self.state)


def main():
    runner = Runner(
        interval="4h",
        lookback_days=120,
        fast_window=30,
        slow_window=84,
        top_n=20,
        position_share_pct=10.0,  # % equity per new position
        trail_pct=4.5,            # trailing stop percent
        hold_min_bars=3,          # bars price must stay <= stop to trigger
        cooldown_bars=8,          # bars to wait after exit before reentry
    )
    runner.run_once()


if __name__ == "__main__":
    main()
