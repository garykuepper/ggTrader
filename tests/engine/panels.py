"""Seeded synthetic panels and independent numpy reference formulas for the engine tests.

Nothing here imports vectorbt: the references are the oracles the adapter is held to.

The parameter sets in ``FROZEN_PANELS`` feed bit-exact comparisons against the frozen legacy
goldens, so their arrays are read from ``golden/panels.npz`` instead of being generated: the
generator uses ``np.exp``, whose float64 loop is CPU-dependent (AVX-512 versus AVX2 can differ in
the last bit), which made those tests fail on GitHub runners. Other parameter sets are generated.
Rewrite or verify the file with ``scripts/freeze_test_inputs.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tests.frozen_inputs import load_frozen

GOLDEN_SEED = 20261106
PANELS_FILE = Path(__file__).parent / "golden" / "panels.npz"
PANEL_ARRAYS = ("close", "high", "low", "volume")
# (seed, n_bars, n_syms, holes) of every panel a golden comparison reads (test_portfolio.py).
FROZEN_PANELS: tuple[tuple[int, int, int, bool], ...] = (
    (3, 500, 6, True),
    (4, 400, 4, False),
    (6, 500, 4, False),
    (6, 300, 5, True),
    (8, 400, 6, False),
    (8, 120, 3, False),
)
START = "2020-01-01"


def panel_key(seed: int, n_bars: int, n_syms: int, holes: bool, start: str) -> str:
    """The archive key of one parameter set."""
    return f"panel|seed={seed}|n_bars={n_bars}|n_syms={n_syms}|holes={holes}|start={start}"


def make_panel(
    seed: int = GOLDEN_SEED,
    n_bars: int = 600,
    n_syms: int = 8,
    *,
    holes: bool = False,
    start: str = START,
) -> dict[str, pd.DataFrame]:
    """Log-normal close panel with high/low/volume; optional NaN holes and a price gap.

    A frozen parameter set (see the module docstring) is read from ``golden/panels.npz``.
    """
    frozen = load_frozen(PANELS_FILE, panel_key(seed, n_bars, n_syms, holes, start), PANEL_ARRAYS)
    if frozen is None:
        frozen = generate_panel_arrays(seed, n_bars, n_syms, holes=holes)
    index = pd.bdate_range(start, periods=n_bars)
    symbols = [f"S{i:02d}" for i in range(n_syms)]
    return {k: pd.DataFrame(frozen[k], index=index, columns=symbols) for k in PANEL_ARRAYS}


def generate_panel_arrays(
    seed: int, n_bars: int, n_syms: int, *, holes: bool = False
) -> dict[str, np.ndarray]:
    """Generate the panel arrays live (CPU-dependent in the last bit; see the module docstring)."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, (n_bars, n_syms)), axis=0))
    high = close * (1.0 + rng.uniform(0.0, 0.01, close.shape))
    low = close * (1.0 - rng.uniform(0.0, 0.01, close.shape))
    volume = rng.lognormal(12.0, 0.5, close.shape)
    if holes:
        close[100:104, 1] = np.nan  # interior NaN hole
        close[:150, 2] = np.nan  # late listing
        close[400:, 3] = np.nan  # delisting
        close[300:, 4] *= 0.5  # a price gap
        high, low = np.where(np.isnan(close), np.nan, high), np.where(np.isnan(close), np.nan, low)
    return {"close": close, "high": high, "low": low, "volume": volume}


def ref_sma(x: np.ndarray, window: int) -> np.ndarray:
    """Plain loop SMA; NaN until a full window of finite values exists."""
    out = np.full(x.shape, np.nan)
    for i in range(window - 1, len(x)):
        out[i] = x[i - window + 1 : i + 1].mean()
    return out


def ref_ema(x: np.ndarray, window: int, seed_at: int = 0) -> np.ndarray:
    """Recursive EMA, alpha = 2/(window+1), seeded with x[seed_at]; no warm-up mask."""
    alpha = 2.0 / (window + 1)
    out = np.full(x.shape, np.nan)
    out[seed_at] = x[seed_at]
    for i in range(seed_at + 1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def ref_wilder_rsi(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder 1978 RSI via the recursive average with alpha = 1/n, seeded at the first diff."""
    delta = np.diff(x)
    gain, loss = np.clip(delta, 0.0, None), np.clip(-delta, 0.0, None)
    avg_gain, avg_loss = np.empty(len(delta)), np.empty(len(delta))
    avg_gain[0], avg_loss[0] = gain[0], loss[0]
    for i in range(1, len(delta)):
        avg_gain[i] = (1.0 - 1.0 / n) * avg_gain[i - 1] + gain[i] / n
        avg_loss[i] = (1.0 - 1.0 / n) * avg_loss[i - 1] + loss[i] / n
    out = np.full(x.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def ref_bbands(
    x: np.ndarray, window: int, alpha: float, ddof: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger middle/upper/lower from a plain window loop with the given ddof."""
    middle = ref_sma(x, window)
    std = np.full(x.shape, np.nan)
    for i in range(window - 1, len(x)):
        std[i] = x[i - window + 1 : i + 1].std(ddof=ddof)
    return middle, middle + alpha * std, middle - alpha * std


def ref_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True range; the first bar has no previous close so it is high - low."""
    tr = high - low
    prev = close[:-1]
    tr[1:] = np.maximum.reduce(
        [high[1:] - low[1:], np.abs(high[1:] - prev), np.abs(low[1:] - prev)]
    )
    return tr


def ref_atr_sma(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """ATR as the simple average of true range (the legacy ``ewm=False`` convention)."""
    return ref_sma(ref_true_range(high, low, close), window)


def ref_max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown of the compounded return path (negative number)."""
    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float(np.min(wealth / peak - 1.0))
