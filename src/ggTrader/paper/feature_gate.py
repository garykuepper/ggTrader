"""ML feature gate: filter ensemble buy signals through a LightGBM classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "ensemble_gate.joblib"
DEFAULT_THRESHOLD = 0.55


def extract_features(
    close: pd.Series, volume: pd.Series, bar_date: pd.Timestamp
) -> dict[str, float]:
    """Compute 10 features for one symbol at one bar.

    Args:
        close: full close price series for the symbol (indexed by date, up to bar_date).
        volume: full volume series for the symbol.
        bar_date: the entry bar date.
    """
    c = close.loc[:bar_date]
    v = volume.loc[:bar_date]

    rets = c.pct_change()

    vol_5d = float(rets.iloc[-5:].std() * np.sqrt(252)) if len(rets) >= 5 else 0.0
    vol_20d = float(rets.iloc[-20:].std() * np.sqrt(252)) if len(rets) >= 20 else 0.0
    vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1.0

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50.0

    # BB %B
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    band_width = upper.iloc[-1] - lower.iloc[-1]
    bb_pctb = float((c.iloc[-1] - lower.iloc[-1]) / band_width) if band_width > 0 else 0.5

    # Momentum
    ret_5d = float(c.iloc[-1] / c.iloc[-6] - 1.0) if len(c) >= 6 else 0.0
    ret_20d = float(c.iloc[-1] / c.iloc[-21] - 1.0) if len(c) >= 21 else 0.0

    # Volume ratio
    vol_avg_20 = float(v.iloc[-20:].mean()) if len(v) >= 20 else 1.0
    volume_ratio = float(v.iloc[-1] / vol_avg_20) if vol_avg_20 > 0 else 1.0

    # ATR ratio
    high_proxy = c  # approximate with close when high/low unavailable
    tr = c.diff().abs()
    atr_5 = float(tr.iloc[-5:].mean()) if len(tr) >= 5 else 0.0
    atr_20 = float(tr.iloc[-20:].mean()) if len(tr) >= 20 else 0.0
    atr_ratio = atr_5 / atr_20 if atr_20 > 0 else 1.0

    dow = bar_date.dayofweek if hasattr(bar_date, "dayofweek") else 0

    return {
        "vol_5d": vol_5d,
        "vol_20d": vol_20d,
        "vol_ratio": vol_ratio,
        "rsi_14": rsi_val,
        "bb_pctb": bb_pctb,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "volume_ratio": volume_ratio,
        "atr_ratio": atr_ratio,
        "day_of_week": float(dow),
    }


FEATURE_NAMES = [
    "vol_5d",
    "vol_20d",
    "vol_ratio",
    "rsi_14",
    "bb_pctb",
    "ret_5d",
    "ret_20d",
    "volume_ratio",
    "atr_ratio",
    "day_of_week",
]


class FeatureGate:
    """Load a pre-trained LightGBM model and gate buy signals by predicted probability."""

    def __init__(
        self,
        model_path: Path | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self._threshold = threshold
        self._model: Any = None
        self._enabled = False

        path = model_path or DEFAULT_MODEL_PATH
        if path.exists():
            import joblib

            self._model = joblib.load(path)
            self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def score(self, features: dict[str, float]) -> float:
        """Return P(profitable) for a single entry signal."""
        if not self._enabled:
            return 1.0
        row = pd.DataFrame([{k: features[k] for k in FEATURE_NAMES}])
        proba = self._model.predict_proba(row)[0, 1]
        return float(proba)

    def filter_buys(
        self,
        buys: list[str],
        ohlcv: pd.DataFrame,
    ) -> tuple[list[str], dict[str, float]]:
        """Filter buy signals through the ML gate.

        Args:
            buys: list of symbols with buy signals.
            ohlcv: MultiIndex DataFrame (symbol, field) with at least 'close' and 'volume'.

        Returns:
            (kept_buys, all_scores) where all_scores maps symbol -> P(profitable).
        """
        if not self._enabled or not buys:
            return buys, {}

        bar_date = ohlcv.index[-1]
        scores: dict[str, float] = {}
        kept: list[str] = []

        sym_cols = set(ohlcv.columns.get_level_values(0).unique())
        for symbol in buys:
            if symbol not in sym_cols:
                kept.append(symbol)
                continue
            close = ohlcv[symbol]["close"].dropna()
            volume = ohlcv[symbol].get("volume", pd.Series(dtype=float))
            if volume is None or volume.empty:
                volume = pd.Series(1.0, index=close.index)
            else:
                volume = volume.dropna()

            feats = extract_features(close, volume, bar_date)
            prob = self.score(feats)
            scores[symbol] = prob
            if prob >= self._threshold:
                kept.append(symbol)

        return kept, scores
