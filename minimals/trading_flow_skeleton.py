
#!/usr/bin/env python3
"""
Trading Flow Skeleton: Volume Screen → Signals (Kalman) → Rank → Size (PyPortfolioOpt) → ATR Trailing Stops → Execute

- Universe: Top-N by 24h USD volume from your exchange (e.g., Kraken via ccxt). Stablecoins filtered out.
- Data: OHLCV (e.g., 4h). Use your existing data layer; here we provide placeholders.
- Signals: Kalman local-level residual z-score (mean-reversion) or slope (trend). Choose one or both.
- Ranking: Strength score = |z| * confidence * liquidity weight (customizable).
- Sizing: PyPortfolioOpt (HRP or EF) on eligibles, respecting caps and leverage.
- Risk: ATR trailing stops (Wilder ATR), per-position persistence + update logic.
- Execution: Translate target weights → target quantities → orders; obey min size/lot increments & fees.

Replace TODO parts with your ggTrader data/execution code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# ----------------------------
# Config
# ----------------------------

@dataclass
class UniverseConfig:
    top_by_volume: int = 20
    exclude_tickers: List[str] = None  # e.g., stablecoins
    quote: str = "USD"                 # or "USDT" depending on venue
    min_history_bars: int = 400        # ensure enough history for ATR/zscore windows

@dataclass
class BarConfig:
    interval: str = "4h"
    lookback_bars: int = 1500          # how many past bars to fetch

@dataclass
class SignalConfig:
    resid_lb: int = 60                 # residual zscore window
    entry_z: float = 1.2
    exit_z: float = 0.2
    allow_shorts: bool = False
    # slope/speed signal toggles
    use_resid_mr: bool = True          # mean-reversion on residual z
    use_trend_slope: bool = False      # optional slope-of-KF trend signal
    slope_lb: int = 20                 # slope window (bars)

@dataclass
class SizerConfig:
    method: str = "HRP"                # HRP or EF
    max_w: float = 0.20
    leverage: float = 1.0
    ema_span: int = 60
    shrink: bool = True
    l2_reg: float = 0.001
    allow_shorts: bool = False
    top_k: int = 8                     # cap # of concurrent positions via eligibility

@dataclass
class RiskConfig:
    atr_period: int = 14
    atr_mult: float = 3.0              # e.g., 3*ATR
    max_positions: int = 8
    per_trade_risk_frac: float = 0.01  # optional: risk budgeting
    cooldown_bars: int = 6             # bars to wait after stop/exit
    slip_bp: float = 2.0               # slippage in bps for stop simulation
    fee_bp: float = 8.0

@dataclass
class EngineConfig:
    universe: UniverseConfig = UniverseConfig()
    bars: BarConfig = BarConfig()
    signals: SignalConfig = SignalConfig()
    sizer: SizerConfig = SizerConfig()
    risk: RiskConfig = RiskConfig()


# ----------------------------
# Utils: ATR, Kalman, zscore
# ----------------------------

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Wilder ATR on OHLCV DataFrame with columns: 'high','low','close'
    """
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    # Wilder's smoothing (EMA with alpha=1/n)
    atr = tr.ewm(alpha=1.0/n, adjust=False).mean()
    return atr.rename('atr')

@dataclass
class KFConfig:
    q: float = 0.0025
    r: float = 0.05
    p0: float = 1.0
    x0: Optional[float] = None

def kalman_local_level(y: pd.Series, kf: KFConfig) -> Tuple[pd.Series, pd.Series]:
    q, r = kf.q, kf.r
    x_prev = y.iloc[0] if kf.x0 is None else kf.x0
    p_prev = kf.p0
    x_hat = np.zeros(len(y))
    resid = np.zeros(len(y))
    for i, yt in enumerate(y.values):
        x_pred = x_prev
        p_pred = p_prev + q
        innov = yt - x_pred
        s = p_pred + r
        k = p_pred / s
        x_new = x_pred + k * innov
        p_new = (1 - k) * p_pred
        x_hat[i] = x_new
        resid[i] = innov
        x_prev, p_prev = x_new, p_new
    return pd.Series(x_hat, index=y.index, name='kf_est'), pd.Series(resid, index=y.index, name='resid')

def zscore(s: pd.Series, lb: int) -> pd.Series:
    m = s.rolling(lb, min_periods=lb//2).mean()
    sd = s.rolling(lb, min_periods=lb//2).std(ddof=0)
    return (s - m) / sd


# ----------------------------
# Data Interfaces (placeholders)
# ----------------------------

class DataProvider:
    """
    Implement using ccxt or your store.
    Methods should return wide DataFrames indexed by datetime:
    - get_top_volume_universe(): List[str]
    - get_ohlcv(symbols): Dict[symbol -> DataFrame with columns ohlcv]
    """
    def get_top_volume_universe(self, cfg: UniverseConfig) -> List[str]:
        # TODO: Use ccxt to fetch tickers & 24h stats; filter stablecoins
        return []

    def get_ohlcv(self, symbols: List[str], bars: BarConfig) -> Dict[str, pd.DataFrame]:
        # TODO: Return dict of DataFrames with columns: open,high,low,close,volume
        return {}


# ----------------------------
# Signal Engine
# ----------------------------

class SignalEngine:
    def __init__(self, cfg: SignalConfig, kf_cfg: KFConfig = KFConfig()):
        self.cfg = cfg
        self.kf_cfg = kf_cfg

    def compute(self, ohlcv: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Return dict of per-symbol signal frames with columns:
        - close, kf_est, resid, z_resid, slope(optional), strength
        """
        out = {}
        for sym, df in ohlcv.items():
            close = df['close'].copy()
            if close.isna().sum() > 0:
                close = close.ffill()
            kf_est, resid = kalman_local_level(close, KFConfig())
            z_resid = zscore(close - kf_est, self.cfg.resid_lb).clip(-5, 5)

            # Optional trend slope: slope of kf_est over slope_lb via linear fit
            slope = None
            if self.cfg.use_trend_slope:
                x = np.arange(len(kf_est))
                coef = pd.Series(kf_est).rolling(self.cfg.slope_lb).apply(
                    lambda w: np.polyfit(x[:len(w)], w, 1)[0], raw=False
                )
                slope = coef.rename('slope')

            sig = pd.DataFrame({'close': close, 'kf_est': kf_est, 'resid': close - kf_est, 'z_resid': z_resid})
            if slope is not None:
                sig['slope'] = slope

            out[sym] = sig
        return out


# ----------------------------
# Ranker
# ----------------------------

class Ranker:
    """
    Example strength score:
      strength = |z_resid| * liquidity_weight
    You may also weight by ADX, realized vol, or recent PnL stability.
    """
    def __init__(self, top_k: int, liq_weight: float = 0.25):
        self.top_k = top_k
        self.liq_weight = liq_weight

    def rank(self, signals: Dict[str, pd.DataFrame], volumes: Dict[str, float]) -> Dict[str, float]:
        latest_strength = {}
        for sym, sig in signals.items():
            z = sig['z_resid'].iloc[-1]
            liq = volumes.get(sym, 1.0)
            liq_w = 1.0 + self.liq_weight * np.tanh(np.log1p(liq) / 10.0)
            latest_strength[sym] = float(abs(z) * liq_w)
        # Return sorted dict of top_k
        ranked = dict(sorted(latest_strength.items(), key=lambda kv: kv[1], reverse=True)[: self.top_k])
        return ranked


# ----------------------------
# Sizer (PyPortfolioOpt)
# ----------------------------

class Sizer:
    def __init__(self, cfg: SizerConfig):
        self.cfg = cfg

    def size(self, prices: pd.DataFrame, elig_longs: List[str], elig_shorts: List[str]) -> pd.Series:
        # Lazy-import pypfopt; user must install in their env
        from pypfopt import expected_returns, risk_models
        weights = pd.Series(0.0, index=prices.columns)

        universe = list(set(elig_longs + (elig_shorts if self.cfg.allow_shorts else [])))
        if not universe:
            return weights

        sub = prices[universe].dropna()
        if sub.shape[1] == 1:
            weights.loc[sub.columns[0]] = 1.0
            return weights

        # Returns/cov
        if self.cfg.shrink:
            S = risk_models.CovarianceShrinkage(sub).ledoit_wolf()
        else:
            S = risk_models.sample_cov(sub)
        mu = expected_returns.ema_historical_return(sub, span=self.cfg.ema_span)

        if self.cfg.method.upper() == "HRP":
            from pypfopt.hrp import HRPOpt
            rets = sub.pct_change().dropna(how="any")
            hrp = HRPOpt(rets)
            raw = pd.Series(hrp.optimize()).reindex(sub.columns).fillna(0.0)
            raw[~raw.index.isin(elig_longs)] = 0.0  # long-only by default
            w = raw / raw.sum() if raw.sum() > 0 else raw
        else:
            from pypfopt.efficient_frontier import EfficientFrontier
            from pypfopt import objective_functions
            ef = EfficientFrontier(mu, S, weight_bounds=(0, self.cfg.max_w))
            ef.add_objective(objective_functions.L2_reg, gamma=self.cfg.l2_reg)
            w = pd.Series(ef.max_sharpe())
            w = w.reindex(sub.columns).fillna(0.0)
            w[~w.index.isin(elig_longs)] = 0.0
            w = w / w.sum() if w.sum() > 0 else w

        w = w.clip(upper=self.cfg.max_w)
        if w.sum() > 0:
            w = w * (self.cfg.leverage / w.sum())
        weights.update(w)
        return weights


# ----------------------------
# Risk: ATR Trailing Stop
# ----------------------------

class ATRStopManager:
    """
    Maintains per-symbol trailing stop levels and updates them with each bar.
    For a long: stop = max(prev_stop, close - k*ATR).
    For a short: stop = min(prev_stop, close + k*ATR).
    """
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.stop_levels: Dict[str, float] = {}
        self.cooldowns: Dict[str, int] = {}

    def init_or_update(self, sym: str, df: pd.DataFrame, side: int) -> float:
        a = atr(df, self.cfg.atr_period).iloc[-1]
        px = df['close'].iloc[-1]
        if side > 0:  # long
            new_stop = px - self.cfg.atr_mult * a
            if sym in self.stop_levels:
                self.stop_levels[sym] = max(self.stop_levels[sym], new_stop)
            else:
                self.stop_levels[sym] = new_stop
        elif side < 0:  # short
            new_stop = px + self.cfg.atr_mult * a
            if sym in self.stop_levels:
                self.stop_levels[sym] = min(self.stop_levels[sym], new_stop)
            else:
                self.stop_levels[sym] = new_stop
        return self.stop_levels[sym]

    def check_exit(self, sym: str, df: pd.DataFrame, side: int) -> bool:
        if sym not in self.stop_levels:
            return False
        px = df['close'].iloc[-1]
        stop = self.stop_levels[sym]
        if (side > 0 and px <= stop) or (side < 0 and px >= stop):
            self.cooldowns[sym] = self.cfg.cooldown_bars
            del self.stop_levels[sym]
            return True
        return False

    def tick_cooldowns(self):
        for sym in list(self.cooldowns.keys()):
            self.cooldowns[sym] -= 1
            if self.cooldowns[sym] <= 0:
                del self.cooldowns[sym]

    def blocked(self, sym: str) -> bool:
        return sym in self.cooldowns


# ----------------------------
# Orchestrator
# ----------------------------

class TradingEngine:
    def __init__(self, data: DataProvider, cfg: EngineConfig):
        self.data = data
        self.cfg = cfg
        self.signals = SignalEngine(cfg.signals)
        self.ranker = Ranker(top_k=cfg.sizer.top_k)
        self.sizer = Sizer(cfg.sizer)
        self.stops = ATRStopManager(cfg.risk)

    def step(self, equity_usd: float) -> Dict[str, object]:
        # 1) Universe
        universe = self.data.get_top_volume_universe(self.cfg.universe)
        universe = [s for s in universe if not self._is_excluded(s)]
        # 2) Fetch bars
        bars = self.data.get_ohlcv(universe, self.cfg.bars)
        # Filter out assets lacking history
        bars = {s: df for s, df in bars.items() if len(df) >= self.cfg.universe.min_history_bars}

        if not bars:
            return {"orders": [], "weights": pd.Series(dtype=float), "reason": "no data"}

        # 3) Signals
        sigs = self.signals.compute(bars)

        # 4) Eligibility
        volumes = {s: bars[s]['volume'].iloc[-1] for s in bars}
        ranked = self.ranker.rank(sigs, volumes)
        # Build long list using z threshold and rank
        longs = [s for s in ranked if sigs[s]['z_resid'].iloc[-1] < -self.cfg.signals.entry_z]
        longs = longs[: self.cfg.risk.max_positions]

        # 5) Sizing
        closes = pd.DataFrame({s: bars[s]['close'] for s in bars})
        weights = self.sizer.size(closes, elig_longs=longs, elig_shorts=[])

        # 6) Risk: update ATR trailing stops
        orders = []
        for sym in longs:
            self.stops.init_or_update(sym, bars[sym], side=+1)
            # Translate weight to target quantity
            px = bars[sym]['close'].iloc[-1]
            target_notional = equity_usd * float(weights.get(sym, 0.0))
            qty = target_notional / px if px > 0 else 0.0
            if qty > 0:
                orders.append({"symbol": sym, "side": "buy", "qty": qty, "price": px, "stop": self.stops.stop_levels[sym]})

        # 7) Exit checks (stop-based). In live, you'd map from positions; here just a placeholder.
        for sym in list(self.stops.stop_levels.keys()):
            # suppose we are long; check stop
            if self.stops.check_exit(sym, bars.get(sym, pd.DataFrame()), side=+1):
                orders.append({"symbol": sym, "side": "sell", "qty": "ALL", "reason": "ATR stop"})

        self.stops.tick_cooldowns()
        return {"orders": orders, "weights": weights.sort_values(ascending=False), "ranked": ranked, "longs": longs}

    def _is_excluded(self, sym: str) -> bool:
        ex = set(self.cfg.universe.exclude_tickers or [])
        # crude stablecoin screening by ticker substring
        stable_like = any(tag in sym.upper() for tag in ["USDT", "USDC", "DAI", "TUSD", "FDUSD", "USD"])
        return (sym in ex) or stable_like


if __name__ == "__main__":
    # Example: wire your real DataProvider and run engine.step(equity_usd)
    class DummyProvider(DataProvider):
        def get_top_volume_universe(self, cfg: UniverseConfig) -> List[str]:
            return ["BTCUSD", "ETHUSD", "SOLUSD", "ADAUSD", "XRPUSD", "DOGEUSD"]

        def get_ohlcv(self, symbols: List[str], bars: BarConfig) -> Dict[str, pd.DataFrame]:
            # Generate synthetic OHLCV
            n = 600
            idx = pd.date_range("2023-01-01", periods=n, freq="4h")
            out = {}
            rng = np.random.default_rng(7)
            for s in symbols:
                logret = rng.normal(0.0005, 0.02, size=n)
                px = 100 * np.exp(np.cumsum(logret))
                close = pd.Series(px, index=idx)
                high = close * (1 + rng.normal(0.002, 0.003, n))
                low  = close * (1 - rng.normal(0.002, 0.003, n))
                openp = close.shift(1).fillna(close.iloc[0])
                vol = rng.lognormal(mean=10, sigma=0.5, size=n)
                df = pd.DataFrame({"open": openp, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
                out[s] = df
            return out

    engine = TradingEngine(DummyProvider(), EngineConfig())
    res = engine.step(equity_usd=10000.0)
    print("Longs:", res["longs"])
    print("Top weights:\n", res["weights"].head())
    print("Orders:", res["orders"])
