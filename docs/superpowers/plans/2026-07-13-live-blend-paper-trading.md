# Live 3-Sleeve Blend on Paper Trading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the live paper account's single-universe SP500 `ensemble` signal generation with the validated 3-sleeve blend (SP500 + MidCap400 + Nasdaq100, inverse-vol/target-vol overlay capped at 1.0x leverage), behind a dry-run burn-in mode.

**Architecture:** A new `overlay.py` module recomputes each sleeve's trailing 90-day equity curve monthly (reusing `lab.simulate.simulate_signals`) and feeds the existing, unmodified `lab/allocation.py` functions to get sleeve weights + a leverage scale. `signal_runner.py` is parameterized to generate signals for any of the three universes instead of just SP500. `risk.py` gains sleeve-aware position sizing. `trader.py` orchestrates all of this and gains a `dry_run` flag for the burn-in period.

**Tech Stack:** Python, pandas, existing `lab/` research modules (`allocation.py`, `simulate.py`, `data.py`), SQLAlchemy via `paper/persist.py`, pytest with `unittest.mock`.

## Global Constraints

- `EnsembleSignal` must be constructed identically (no extra kwargs) everywhere it's used for live signal generation AND for the overlay's trailing-curve simulation — this is Invariant 1 from the spec (`docs/superpowers/specs/2026-07-13-live-blend-paper-trading-design.md`). Concretely: never pass strategy params beyond `LabConfig`; if a call site ever needs to override a param, both the live and overlay call sites must change together.
- `max_leverage=1.0` throughout (matches the validated deployable result, not the idealized 2.0x research default).
- Sleeve universes are exactly `("sp500", "midcap400", "nasdaq100")` — the three sleeves validated in the July-13 roadmap verdict. No fourth sleeve (idio_vol was rejected).
- Global risk guards (`check_drawdown_halt`, `check_daily_loss`, overall `max_positions`, `check_concentration`) are unchanged — do not touch their logic, only feed them different notional values.
- All new DB tables/queries go through `paper/persist.py`'s existing `_get_engine()` pattern (SQLAlchemy `text()`, explicit `conn.commit()`), matching every existing function in that file.
- No code changes to `docker/`, cron, or `scripts/paper_trade.sh` in this plan — the dry-run flag defaults to `True` until a human explicitly flips it after the burn-in window (see Task 8).

---

### Task 1: Parameterize `signal_runner.py` by universe

**Files:**
- Modify: `src/ggTrader/paper/signal_runner.py`
- Test: `tests/paper/test_signal_runner.py`

**Interfaces:**
- Consumes: `ggTrader.data.core.index_constituents.universe_members_asof(universe: str, asof: pd.Timestamp) -> List[str]` (already exists, unified PIT-for-sp500/snapshot-for-others lookup), `ggTrader.lab.data.fetch_stock_ohlcv`, `ggTrader.lab.strategies.ensemble.EnsembleSignal`, `ggTrader.lab.strategy.LabConfig`.
- Produces: `generate_signals(universe: str = "sp500", lookback_days: int = 120) -> dict` with keys `buys: list[str]`, `sells: list[str]`, `as_of: str`, `universe_size: int`, `gate: dict` — same shape as today, `universe` param added with a default that preserves current SP500-only callers.

Today's `generate_signals()` hardcodes `sp500_members_asof`. Task: swap it for the already-existing, universe-agnostic `universe_members_asof(universe, asof)`, and thread a `universe` parameter through. Existing tests patch `ggTrader.paper.signal_runner.sp500_members_asof` — after this change they must patch `ggTrader.paper.signal_runner.universe_members_asof` instead, so update the existing tests' patch targets in the same commit (this is a rename, not new behavior — the current default-arg call still produces byte-identical output for `universe="sp500"`).

- [ ] **Step 1: Write the failing test for universe parameterization**

Add to `tests/paper/test_signal_runner.py`:

```python
    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_universe_param_passed_through(self, mock_fetch, mock_members):
        symbols = ["AAPL", "MSFT", "GOOG"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)

        from ggTrader.paper.signal_runner import generate_signals

        result = generate_signals(universe="midcap400", lookback_days=120)

        mock_members.assert_called_once()
        assert mock_members.call_args[0][0] == "midcap400"
        assert result["universe_size"] == 3
```

Also update every existing `@patch("ggTrader.paper.signal_runner.sp500_members_asof")` decorator in this file to `@patch("ggTrader.paper.signal_runner.universe_members_asof")`, and every `mock_members.return_value = symbols` stays the same (the mock now stands in for the more general function).

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/paper/test_signal_runner.py -v`
Expected: FAIL — `AttributeError` or `ModuleNotFoundError` on `ggTrader.paper.signal_runner.universe_members_asof` (doesn't exist yet), and `generate_signals()` doesn't accept `universe=`.

- [ ] **Step 3: Implement the parameterization**

Edit `src/ggTrader/paper/signal_runner.py`:

```python
"""Generate today's ensemble signals from a given equity universe."""

from __future__ import annotations

import pandas as pd

from ggTrader.data.core.index_constituents import normalize_yf_ticker, universe_members_asof
from ggTrader.lab.data import fetch_stock_ohlcv
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig


def generate_signals(universe: str = "sp500", lookback_days: int = 120) -> dict:
    """Fetch recent data for a PIT universe and return today's ensemble signals.

    Returns dict with keys: buys (list[str]), sells (list[str]),
    as_of (str date), universe_size (int).
    """
    today = pd.Timestamp.now(tz="UTC").normalize()
    start = today - pd.Timedelta(days=lookback_days)

    members = universe_members_asof(universe, today)
    symbols = sorted({normalize_yf_ticker(t) for t in members})

    ohlcv = fetch_stock_ohlcv(symbols, start=str(start.date()), end=str(today.date()))
    sym_cols = list(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in sym_cols}, axis=1)

    if close.empty:
        return {"buys": [], "sells": [], "as_of": str(today.date()), "universe_size": 0}

    last_bar = close.index[-1]
    cfg = LabConfig(min_history_bars=60)
    ensemble = EnsembleSignal(cfg)
    plan = ensemble.select(last_bar, ohlcv, sym_cols)
    if not plan:
        return {
            "buys": [],
            "sells": [],
            "as_of": str(last_bar.date()),
            "universe_size": len(sym_cols),
        }
    targets = ensemble.to_targets({last_bar: plan}, ohlcv)

    last_entries = targets.entries.loc[last_bar]
    last_exits = targets.exits.loc[last_bar]

    buys = sorted(last_entries[last_entries].index.tolist())
    sells = sorted(last_exits[last_exits].index.tolist())

    gate_info: dict = {}
    from ggTrader.paper.feature_gate import FeatureGate

    gate = FeatureGate()
    if gate.enabled and buys:
        raw_count = len(buys)
        buys, scores = gate.filter_buys(buys, ohlcv)
        gate_info = {
            "gate_enabled": True,
            "raw_buys": raw_count,
            "kept_buys": len(buys),
            "scores": scores,
        }
    else:
        gate_info = {"gate_enabled": gate.enabled}

    return {
        "buys": buys,
        "sells": sells,
        "as_of": str(last_bar.date()),
        "universe_size": len(sym_cols),
        "gate": gate_info,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/paper/test_signal_runner.py -v`
Expected: PASS (all tests, including the renamed patches).

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/paper/signal_runner.py tests/paper/test_signal_runner.py
git commit -m "refactor(paper): parameterize generate_signals by universe

Swaps the hardcoded sp500_members_asof for the existing universe-agnostic
universe_members_asof, threading a universe param through. Default
universe=\"sp500\" preserves current behavior exactly."
```

---

### Task 2: Sleeve trailing-curve simulation (`overlay.py`, part 1)

**Files:**
- Create: `src/ggTrader/paper/overlay.py`
- Test: `tests/paper/test_overlay.py`

**Interfaces:**
- Consumes: `generate_signals`'s internals are NOT reused directly (that function returns only today's buys/sells, not the full signal history needed for a trailing curve) — instead this task calls `EnsembleSignal.select()`/`.to_targets()` directly over the trailing window, then `ggTrader.lab.simulate.simulate_signals(targets_by_strategy, prices, base_config, ohlcv=...) -> (returns_df, equity_df, diags)`, and `ggTrader.lab.data.STOCK_BASE_CONFIG`.
- Produces: `compute_sleeve_curve(universe: str, asof: pd.Timestamp, window_days: int = 90) -> pd.Series` — a daily equity curve (pandas Series indexed by date) for that sleeve over `[asof - window_days, asof]`, using the *same* `EnsembleSignal(LabConfig(min_history_bars=60))` construction as `signal_runner.generate_signals` (Invariant 1 — no separate params).

This is the piece that satisfies Invariant 1 from the spec: it must build `EnsembleSignal` exactly like `signal_runner.py` does, so the vol estimate describes the strategy actually trading live.

- [ ] **Step 1: Write the failing test**

Create `tests/paper/test_overlay.py`:

```python
"""Tests for the live blend's trailing-curve and rebalance-weight overlay."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd


def _mock_ohlcv(symbols: list[str], n_days: int = 150) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-07-10", periods=n_days, tz="UTC")
    np.random.seed(7)
    frames = {}
    for sym in symbols:
        price = 100.0 * np.exp(np.random.randn(n_days).cumsum() * 0.015)
        frames[sym] = pd.DataFrame(
            {
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1e6,
            },
            index=dates,
        )
    df = pd.concat(frames, axis=1)
    df.columns.names = ["symbol", "field"]
    return df


class TestComputeSleeveCurve:
    @patch("ggTrader.paper.overlay.universe_members_asof")
    @patch("ggTrader.paper.overlay.fetch_stock_ohlcv")
    def test_returns_equity_series_over_window(self, mock_fetch, mock_members):
        symbols = ["AAPL", "MSFT", "GOOG"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)

        from ggTrader.paper.overlay import compute_sleeve_curve

        asof = pd.Timestamp("2026-07-10", tz="UTC")
        curve = compute_sleeve_curve("sp500", asof, window_days=90)

        assert isinstance(curve, pd.Series)
        assert len(curve) > 0
        assert curve.index.max() <= asof

    @patch("ggTrader.paper.overlay.universe_members_asof")
    @patch("ggTrader.paper.overlay.fetch_stock_ohlcv")
    def test_uses_same_ensemble_construction_as_signal_runner(self, mock_fetch, mock_members):
        """Invariant 1: overlay must build EnsembleSignal identically to
        signal_runner.py -- no separate params, or the vol estimate silently
        describes a different strategy than the one actually live-trading."""
        import inspect

        from ggTrader.paper import overlay, signal_runner

        overlay_src = inspect.getsource(overlay.compute_sleeve_curve)
        runner_src = inspect.getsource(signal_runner.generate_signals)

        assert "EnsembleSignal(cfg)" in overlay_src
        assert "EnsembleSignal(cfg)" in runner_src
        assert "LabConfig(min_history_bars=60)" in overlay_src
        assert "LabConfig(min_history_bars=60)" in runner_src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/paper/test_overlay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ggTrader.paper.overlay'`.

- [ ] **Step 3: Implement `compute_sleeve_curve`**

Create `src/ggTrader/paper/overlay.py`:

```python
"""Live overlay: recompute each sleeve's trailing equity curve and the
resulting inverse-vol / target-vol weights, reusing the exact research
mechanism from lab/allocation.py and lab/simulate.py.
"""

from __future__ import annotations

import pandas as pd

from ggTrader.data.core.index_constituents import normalize_yf_ticker, universe_members_asof
from ggTrader.lab.allocation import inverse_vol_weights, target_vol_scale, trailing_realized_vol
from ggTrader.lab.data import STOCK_BASE_CONFIG, fetch_stock_ohlcv
from ggTrader.lab.simulate import simulate_signals
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig

SLEEVE_UNIVERSES: tuple[str, ...] = ("sp500", "midcap400", "nasdaq100")


def compute_sleeve_curve(universe: str, asof: pd.Timestamp, window_days: int = 90) -> pd.Series:
    """Trailing equity curve for one sleeve, purely from OHLCV + signals.

    Uses the identical EnsembleSignal(LabConfig(min_history_bars=60))
    construction signal_runner.generate_signals uses live, so the vol
    estimate describes the strategy actually trading (Invariant 1).
    """
    start = asof - pd.Timedelta(days=window_days + 60)  # extra warmup for indicators

    members = universe_members_asof(universe, asof)
    symbols = sorted({normalize_yf_ticker(t) for t in members})

    ohlcv = fetch_stock_ohlcv(symbols, start=str(start.date()), end=str(asof.date()))
    sym_cols = list(ohlcv.columns.get_level_values(0).unique())
    if not sym_cols:
        return pd.Series(dtype=float)

    close = pd.concat({s: ohlcv[s]["close"] for s in sym_cols}, axis=1)
    if close.empty:
        return pd.Series(dtype=float)

    cfg = LabConfig(min_history_bars=60)
    ensemble = EnsembleSignal(cfg)

    plans = {}
    for bar in close.loc[str((asof - pd.Timedelta(days=window_days)).date()) :].index:
        plan = ensemble.select(bar, ohlcv, sym_cols)
        if plan:
            plans[bar] = plan
    if not plans:
        return pd.Series(dtype=float)

    targets = ensemble.to_targets(plans, ohlcv)
    returns_df, equity_df, _diags = simulate_signals(
        {"ensemble": targets}, close, dict(STOCK_BASE_CONFIG), ohlcv=ohlcv
    )
    curve = equity_df["ensemble"].dropna()
    return curve.loc[curve.index >= (asof - pd.Timedelta(days=window_days))]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/paper/test_overlay.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/paper/overlay.py tests/paper/test_overlay.py
git commit -m "feat(paper): compute_sleeve_curve -- trailing equity curve per sleeve

Reuses simulate_signals from the lab research module and builds
EnsembleSignal identically to signal_runner.py (Invariant 1: the overlay's
vol estimate must describe the strategy actually trading live)."
```

---

### Task 3: Weight/scale computation + rebalance-date logic (`overlay.py`, part 2)

**Files:**
- Modify: `src/ggTrader/paper/overlay.py`
- Test: `tests/paper/test_overlay.py`

**Interfaces:**
- Consumes: `ggTrader.lab.allocation.trailing_realized_vol(returns: pd.Series, window: int = 60) -> pd.Series`, `inverse_vol_weights(vols: dict[str, float]) -> dict[str, float]`, `target_vol_scale(blend_trailing_vol: float, target_vol: float, max_leverage: float = 2.0) -> float` (all already exist, unmodified).
- Produces: `compute_weights_and_scale(curves: dict[str, pd.Series], target_vol: float = 0.068, window: int = 60, max_leverage: float = 1.0) -> tuple[dict[str, float], float]`, and `should_rebalance(last_rebalance_date: str | None, today: pd.Timestamp) -> bool` (True on the first run of a new calendar month relative to `last_rebalance_date`, or if `last_rebalance_date` is `None`).

- [ ] **Step 1: Write the failing test**

Add to `tests/paper/test_overlay.py`:

```python
class TestComputeWeightsAndScale:
    def test_weights_sum_to_one_and_scale_capped(self):
        from ggTrader.paper.overlay import compute_weights_and_scale

        dates = pd.bdate_range("2026-01-01", periods=120, tz="UTC")
        rng = np.random.default_rng(3)
        curves = {
            "sp500": pd.Series(10000 * np.exp(rng.normal(0, 0.01, 120).cumsum()), index=dates),
            "midcap400": pd.Series(10000 * np.exp(rng.normal(0, 0.02, 120).cumsum()), index=dates),
            "nasdaq100": pd.Series(10000 * np.exp(rng.normal(0, 0.015, 120).cumsum()), index=dates),
        }

        weights, scale = compute_weights_and_scale(curves, max_leverage=1.0)

        assert set(weights) == {"sp500", "midcap400", "nasdaq100"}
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        assert 0.0 <= scale <= 1.0


class TestShouldRebalance:
    def test_none_last_rebalance_triggers(self):
        from ggTrader.paper.overlay import should_rebalance

        assert should_rebalance(None, pd.Timestamp("2026-07-13", tz="UTC")) is True

    def test_same_month_does_not_trigger(self):
        from ggTrader.paper.overlay import should_rebalance

        assert should_rebalance("2026-07-01", pd.Timestamp("2026-07-13", tz="UTC")) is False

    def test_new_month_triggers(self):
        from ggTrader.paper.overlay import should_rebalance

        assert should_rebalance("2026-06-15", pd.Timestamp("2026-07-01", tz="UTC")) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/paper/test_overlay.py -v -k "WeightsAndScale or ShouldRebalance"`
Expected: FAIL — functions don't exist yet.

- [ ] **Step 3: Implement**

Append to `src/ggTrader/paper/overlay.py`:

```python
def compute_weights_and_scale(
    curves: dict[str, pd.Series],
    target_vol: float = 0.068,
    window: int = 60,
    max_leverage: float = 1.0,
) -> tuple[dict[str, float], float]:
    """Inverse-vol weights + target-vol leverage scale for the given sleeve
    curves, reusing the unmodified lab/allocation.py functions."""
    returns = {label: curve.pct_change().dropna() for label, curve in curves.items()}
    vols = {
        label: float(trailing_realized_vol(r, window=window).iloc[-1]) if len(r) >= window else None
        for label, r in returns.items()
    }
    weights = inverse_vol_weights(vols)

    common = None
    for label, r in returns.items():
        common = r.index if common is None else common.intersection(r.index)
    blended = sum(returns[label].reindex(common) * w for label, w in weights.items())
    blend_vol = (
        float(trailing_realized_vol(blended, window=window).iloc[-1])
        if common is not None and len(common) >= window
        else float("nan")
    )
    scale = target_vol_scale(blend_vol, target_vol, max_leverage=max_leverage)
    return weights, scale


def should_rebalance(last_rebalance_date: str | None, today: pd.Timestamp) -> bool:
    """True on the first run after entering a new calendar month, or if
    there is no prior rebalance recorded."""
    if last_rebalance_date is None:
        return True
    last = pd.Timestamp(last_rebalance_date, tz="UTC")
    return (today.year, today.month) != (last.year, last.month)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/paper/test_overlay.py -v`
Expected: PASS (all of Task 2 + Task 3's tests).

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/paper/overlay.py tests/paper/test_overlay.py
git commit -m "feat(paper): compute_weights_and_scale + should_rebalance

Wraps the unmodified lab/allocation.py inverse-vol/target-vol functions
for live use, plus a monthly rebalance-date check matching the
combine_sleeves(rebalance=\"ME\") cadence validated in research."
```

---

### Task 4: Rebalance-state persistence

**Files:**
- Modify: `src/ggTrader/paper/persist.py`
- Test: `tests/paper/test_paper_persist.py`

**Interfaces:**
- Consumes: existing `_get_engine()` pattern in `persist.py`.
- Produces: `get_rebalance_state() -> dict | None` (returns `{"rebalance_date": str, "weights": dict[str, float], "scale": float}` or `None` if no row exists yet), `save_rebalance_state(rebalance_date: str, weights: dict[str, float], scale: float) -> None`.

- [ ] **Step 1: Write the failing test**

Every existing test class in `tests/paper/test_paper_persist.py` follows the same pattern: `@patch("ggTrader.paper.persist._get_engine")` on the class, a `MagicMock()` standing in for the connection (wired via `mock_engine.return_value.connect.return_value.__enter__`/`__exit__`), and assertions on `mock_conn.execute.call_args_list` — there is no real DB roundtrip anywhere in this file. Follow that exact pattern (mirroring `TestLogSnapshot`/`TestGetLatestSnapshot`), not a real database call:

```python
@patch("ggTrader.paper.persist._get_engine")
class TestRebalanceState:
    def test_save_rebalance_state_inserts_row(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        from ggTrader.paper.persist import save_rebalance_state

        save_rebalance_state(
            "2026-07-01", {"sp500": 0.5, "midcap400": 0.3, "nasdaq100": 0.2}, 0.87
        )

        mock_conn.execute.assert_called_once()
        sql_str = str(mock_conn.execute.call_args[0][0])
        assert "paper_rebalance_state" in sql_str
        params = mock_conn.execute.call_args[0][1]
        assert params["rd"] == "2026-07-01"
        assert params["s"] == 0.87

    def test_get_rebalance_state_returns_parsed_row(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = (
            "2026-07-01",
            {"sp500": 0.5, "midcap400": 0.3, "nasdaq100": 0.2},
            0.87,
        )

        from ggTrader.paper.persist import get_rebalance_state

        state = get_rebalance_state()

        assert state["rebalance_date"] == "2026-07-01"
        assert state["weights"] == {"sp500": 0.5, "midcap400": 0.3, "nasdaq100": 0.2}
        assert state["scale"] == 0.87

    def test_get_rebalance_state_returns_none_when_empty(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = None

        from ggTrader.paper.persist import get_rebalance_state

        assert get_rebalance_state() is None
```

Also add `"paper_rebalance_state"` to the existing `TestInitSchema.test_creates_tables` assertion list (that test already asserts `"paper_trades"`/`"paper_snapshots"` appear in the executed SQL — add the new table name alongside them).

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/paper/test_paper_persist.py -v -k Rebalance`
Expected: FAIL — `save_rebalance_state`/`get_rebalance_state` don't exist.

- [ ] **Step 3: Implement**

Edit `src/ggTrader/paper/persist.py` — add to `_SCHEMA`:

```python
CREATE TABLE IF NOT EXISTS paper_rebalance_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    rebalance_date DATE NOT NULL,
    weights JSONB NOT NULL,
    scale DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT single_row CHECK (id = 1)
);
```

Add functions at the end of the file:

```python
def get_rebalance_state() -> dict | None:
    """Return the current sleeve weights/scale, or None if never set."""
    with _get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT rebalance_date, weights, scale FROM paper_rebalance_state WHERE id = 1")
        ).first()
    if row is None:
        return None
    rebalance_date, weights, scale = row
    return {
        "rebalance_date": str(rebalance_date),
        "weights": weights if isinstance(weights, dict) else json.loads(weights),
        "scale": float(scale),
    }


def save_rebalance_state(rebalance_date: str, weights: dict[str, float], scale: float) -> None:
    """Upsert the single current rebalance-state row (id=1)."""
    with _get_engine().connect() as conn:
        conn.execute(
            text(
                "INSERT INTO paper_rebalance_state (id, rebalance_date, weights, scale) "
                "VALUES (1, :rd, :w, :s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "rebalance_date = EXCLUDED.rebalance_date, "
                "weights = EXCLUDED.weights, "
                "scale = EXCLUDED.scale"
            ),
            {"rd": rebalance_date, "w": json.dumps(weights), "s": scale},
        )
        conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/paper/test_paper_persist.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/paper/persist.py tests/paper/test_paper_persist.py
git commit -m "feat(paper): persist rebalance state (weights + leverage scale)

Single-row table (id=1, upserted) so the monthly overlay recomputation
has somewhere to read/write its last decision, enabling the
stale-weights fallback on a failed OHLCV fetch."
```

---

### Task 5: Multi-sleeve signal generation with rebalance orchestration + fallback

**Files:**
- Modify: `src/ggTrader/paper/signal_runner.py`
- Test: `tests/paper/test_signal_runner.py`

**Interfaces:**
- Consumes: `generate_signals(universe, lookback_days) -> dict` (Task 1), `overlay.compute_sleeve_curve`, `overlay.compute_weights_and_scale`, `overlay.should_rebalance`, `overlay.SLEEVE_UNIVERSES` (Tasks 2-3), `persist.get_rebalance_state`, `persist.save_rebalance_state` (Task 4).
- Produces: `generate_blended_signals() -> dict` with keys: `sleeves: dict[str, dict]` (universe -> that sleeve's `generate_signals()` result), `weights: dict[str, float]`, `scale: float`, `rebalanced_today: bool`, `fallback_used: bool`.

- [ ] **Step 1: Write the failing test**

Add to `tests/paper/test_signal_runner.py`:

```python
class TestGenerateBlendedSignals:
    @patch("ggTrader.paper.signal_runner.save_rebalance_state")
    @patch("ggTrader.paper.signal_runner.compute_weights_and_scale")
    @patch("ggTrader.paper.signal_runner.compute_sleeve_curve")
    @patch("ggTrader.paper.signal_runner.get_rebalance_state")
    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_first_run_rebalances_and_returns_all_sleeves(
        self, mock_fetch, mock_members, mock_get_state, mock_curve, mock_weights, mock_save
    ):
        symbols = ["AAPL", "MSFT"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)
        mock_get_state.return_value = None  # no prior rebalance
        mock_curve.return_value = pd.Series([1.0, 1.01, 1.02])
        mock_weights.return_value = ({"sp500": 0.4, "midcap400": 0.3, "nasdaq100": 0.3}, 0.9)

        from ggTrader.paper.signal_runner import generate_blended_signals

        result = generate_blended_signals()

        assert set(result["sleeves"]) == {"sp500", "midcap400", "nasdaq100"}
        assert result["weights"] == {"sp500": 0.4, "midcap400": 0.3, "nasdaq100": 0.3}
        assert result["scale"] == 0.9
        assert result["rebalanced_today"] is True
        assert result["fallback_used"] is False
        mock_save.assert_called_once()

    @patch("ggTrader.paper.signal_runner.save_rebalance_state")
    @patch("ggTrader.paper.signal_runner.compute_weights_and_scale")
    @patch("ggTrader.paper.signal_runner.compute_sleeve_curve")
    @patch("ggTrader.paper.signal_runner.get_rebalance_state")
    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_mid_month_reuses_stored_weights(
        self, mock_fetch, mock_members, mock_get_state, mock_curve, mock_weights, mock_save
    ):
        symbols = ["AAPL", "MSFT"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)
        today_str = str(pd.Timestamp.now(tz="UTC").normalize().date())
        mock_get_state.return_value = {
            "rebalance_date": today_str,
            "weights": {"sp500": 0.5, "midcap400": 0.25, "nasdaq100": 0.25},
            "scale": 0.8,
        }

        from ggTrader.paper.signal_runner import generate_blended_signals

        result = generate_blended_signals()

        assert result["weights"] == {"sp500": 0.5, "midcap400": 0.25, "nasdaq100": 0.25}
        assert result["scale"] == 0.8
        assert result["rebalanced_today"] is False
        mock_curve.assert_not_called()
        mock_save.assert_not_called()

    @patch("ggTrader.paper.signal_runner.save_rebalance_state")
    @patch("ggTrader.paper.signal_runner.compute_weights_and_scale")
    @patch("ggTrader.paper.signal_runner.compute_sleeve_curve")
    @patch("ggTrader.paper.signal_runner.get_rebalance_state")
    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_rebalance_fetch_failure_falls_back_to_stored_weights(
        self, mock_fetch, mock_members, mock_get_state, mock_curve, mock_weights, mock_save
    ):
        symbols = ["AAPL", "MSFT"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)
        mock_get_state.return_value = {
            "rebalance_date": "2026-05-01",  # stale -- would normally trigger a rebalance
            "weights": {"sp500": 0.6, "midcap400": 0.2, "nasdaq100": 0.2},
            "scale": 0.7,
        }
        mock_curve.side_effect = RuntimeError("OHLCV fetch failed")

        from ggTrader.paper.signal_runner import generate_blended_signals

        result = generate_blended_signals()

        assert result["weights"] == {"sp500": 0.6, "midcap400": 0.2, "nasdaq100": 0.2}
        assert result["scale"] == 0.7
        assert result["fallback_used"] is True
        mock_save.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/paper/test_signal_runner.py -v -k BlendedSignals`
Expected: FAIL — `generate_blended_signals` doesn't exist.

- [ ] **Step 3: Implement**

Add to `src/ggTrader/paper/signal_runner.py` (append imports at top, function at bottom):

```python
from ggTrader.paper.overlay import (
    SLEEVE_UNIVERSES,
    compute_sleeve_curve,
    compute_weights_and_scale,
    should_rebalance,
)
from ggTrader.paper.persist import get_rebalance_state, save_rebalance_state
```

```python
def generate_blended_signals() -> dict:
    """Generate today's signals for all three sleeves, recomputing the
    inverse-vol/target-vol overlay monthly. On any failure to recompute
    (e.g. an OHLCV fetch error on a rebalance date), falls back to the last
    stored weights/scale rather than raising."""
    today = pd.Timestamp.now(tz="UTC").normalize()
    sleeves = {universe: generate_signals(universe=universe) for universe in SLEEVE_UNIVERSES}

    state = get_rebalance_state()
    rebalanced_today = False
    fallback_used = False

    if should_rebalance(state["rebalance_date"] if state else None, today):
        try:
            curves = {u: compute_sleeve_curve(u, today) for u in SLEEVE_UNIVERSES}
            weights, scale = compute_weights_and_scale(curves)
            save_rebalance_state(str(today.date()), weights, scale)
            rebalanced_today = True
        except Exception:
            if state is None:
                raise  # no fallback available on the very first run
            weights, scale = state["weights"], state["scale"]
            fallback_used = True
    else:
        weights, scale = state["weights"], state["scale"]

    return {
        "sleeves": sleeves,
        "weights": weights,
        "scale": scale,
        "rebalanced_today": rebalanced_today,
        "fallback_used": fallback_used,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/paper/test_signal_runner.py -v`
Expected: PASS (all of Task 1 + Task 5's tests).

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/paper/signal_runner.py tests/paper/test_signal_runner.py
git commit -m "feat(paper): generate_blended_signals -- 3-sleeve orchestration

Monthly rebalance recomputes weights/scale via the overlay module;
mid-month runs reuse the stored state; a failed rebalance-day OHLCV
fetch falls back to the last stored weights rather than raising."
```

---

### Task 6: Sleeve-aware position sizing in `risk.py`

**Files:**
- Modify: `src/ggTrader/paper/risk.py`
- Test: `tests/paper/test_risk.py`

**Correction from the design conversation:** the spec's "split sleeve budget across that sleeve's buy signals" framing (budget ÷ signal count) would badly overconcentrate on a low-signal day — one buy signal in a 50%-weighted sleeve would put 45% of the portfolio in a single stock. That's not what the research validated: each sleeve's own backtest equity curve was generated by `simulate_signals` using a **fixed fraction of that sleeve's own capital per entry** (the same `position_pct`/`SIGNAL_POSITION_SIZE` convention the flat-3% live system already uses) — the weight×scale overlay only caps how much *total* capital a sleeve may deploy, it never resizes an individual position based on how many signals happened to fire that day. Fixing this here rather than carrying the flawed formula into `trader.py`.

**Interfaces:**
- Consumes: nothing new (pure functions on primitives already in scope).
- Produces on `RiskGuard`: `sleeve_slot_caps(self, weights: dict[str, float]) -> dict[str, int]` (per-sleeve share of `self.cfg.max_positions`, `floor(weight_i * max_positions)`, minimum 1 for any `weight_i > 0`, leftover slots from rounding assigned to the highest-weight sleeve so the total never exceeds `max_positions`) — governs how many concurrent positions a sleeve may hold, not position size. `sleeve_position_notional(self, portfolio_value: float, sleeve_weight: float, scale: float) -> float` (returns `portfolio_value * sleeve_weight * scale * self.cfg.position_pct` — a fixed per-position dollar amount for that sleeve, independent of how many signals fire; matches the flat-3%-of-allocated-capital convention `simulate_signals` used to generate each sleeve's own validated backtest curve).

**Existing `position_notional()` and `max_new_positions()` are untouched** — they remain available for any caller not yet migrated, and this task adds new methods alongside them rather than modifying their signatures.

- [ ] **Step 1: Write the failing test**

Add to `tests/paper/test_risk.py`:

```python
def test_sleeve_slot_caps_proportional(guard):
    caps = guard.sleeve_slot_caps({"sp500": 0.5, "midcap400": 0.3, "nasdaq100": 0.2})
    assert caps["sp500"] == 15
    assert caps["midcap400"] == 9
    assert caps["nasdaq100"] == 6
    assert sum(caps.values()) <= guard.cfg.max_positions


def test_sleeve_slot_caps_minimum_one_slot_for_small_weight():
    cfg = RiskConfig(max_positions=10)
    guard = RiskGuard(cfg)
    caps = guard.sleeve_slot_caps({"sp500": 0.95, "midcap400": 0.03, "nasdaq100": 0.02})
    assert caps["midcap400"] >= 1
    assert caps["nasdaq100"] >= 1
    assert sum(caps.values()) <= 10


def test_sleeve_position_notional_fixed_fraction_of_sleeve_capital(guard):
    # portfolio_value * sleeve_weight * scale * position_pct(0.033) --
    # independent of how many signals fire that day.
    notional = guard.sleeve_position_notional(portfolio_value=10000.0, sleeve_weight=0.4, scale=0.9)
    assert notional == round(10000.0 * 0.4 * 0.9 * 0.033, 2)


def test_sleeve_position_notional_zero_weight_is_zero(guard):
    notional = guard.sleeve_position_notional(portfolio_value=10000.0, sleeve_weight=0.0, scale=0.9)
    assert notional == 0.0


def test_sleeve_position_notional_matches_flat_when_full_weight_full_scale(guard):
    # Sanity check: a single sleeve at weight=1.0, scale=1.0 must reproduce
    # today's flat position_notional exactly -- this is the degenerate case
    # a 1-sleeve "blend" should collapse back to current live behavior.
    sleeve_notional = guard.sleeve_position_notional(portfolio_value=10000.0, sleeve_weight=1.0, scale=1.0)
    assert sleeve_notional == guard.position_notional(10000.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/paper/test_risk.py -v -k sleeve`
Expected: FAIL — methods don't exist.

- [ ] **Step 3: Implement**

Add to the `RiskGuard` class in `src/ggTrader/paper/risk.py`:

```python
    def sleeve_slot_caps(self, weights: dict[str, float]) -> dict[str, int]:
        """Per-sleeve share of max_positions, proportional to weight.

        floor(weight_i * max_positions), minimum 1 slot for any sleeve with
        weight_i > 0, with leftover slots from rounding assigned to the
        highest-weight sleeve so the total never exceeds max_positions.
        """
        raw = {
            label: (max(1, int(w * self.cfg.max_positions)) if w > 0 else 0)
            for label, w in weights.items()
        }
        total = sum(raw.values())
        if total > self.cfg.max_positions and raw:
            top = max(raw, key=lambda k: weights[k])
            raw[top] -= total - self.cfg.max_positions
            raw[top] = max(raw[top], 1 if weights[top] > 0 else 0)
        return raw

    def sleeve_position_notional(
        self,
        portfolio_value: float,
        sleeve_weight: float,
        scale: float,
    ) -> float:
        """Dollar amount for a single new position within one sleeve.

        A fixed fraction (position_pct) of that sleeve's allocated capital
        (portfolio_value * sleeve_weight * scale) -- independent of how many
        signals fire that day. Matches the same fixed-fraction-per-entry
        convention simulate_signals used to generate each sleeve's own
        validated backtest curve; the weight*scale overlay caps how much
        total capital a sleeve may deploy (via sleeve_slot_caps), it does
        not resize individual positions based on signal count.
        """
        return round(portfolio_value * sleeve_weight * scale * self.cfg.position_pct, 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/paper/test_risk.py -v`
Expected: PASS (all existing + new tests).

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/paper/risk.py tests/paper/test_risk.py
git commit -m "feat(paper): sleeve-aware position sizing (RiskGuard)

Adds sleeve_slot_caps + sleeve_position_notional alongside the existing
flat position_notional (untouched). Implements the per-sleeve dollar
budget from the design spec: portfolio_value x weight x scale, split
across that sleeve's signals."
```

---

### Task 7: Wire `trader.py` to the blend + add dry-run mode

**Files:**
- Modify: `src/ggTrader/paper/trader.py`
- Test: `tests/paper/test_trader.py`

**Interfaces:**
- Consumes: `signal_runner.generate_blended_signals()` (Task 5), `RiskGuard.sleeve_slot_caps` / `sleeve_position_notional` (Task 6).
- Produces: `PaperTrader.__init__(self, broker, notifier, risk_cfg=None, dry_run: bool = True)`; `PaperTrader.run()` unchanged return shape (`{"buys": [...], "sells": [...], "errors": [...]}`) but now sources signals from all three sleeves and, when `dry_run=True`, logs/notifies intended trades without calling `broker.submit_buy`/`submit_sell`.

Read the full current `run()` method (already shown above) before editing — this task replaces the single `generate_signals()` call and the flat-notional buy loop; the sell loop, halt checks, order-polling/reconciliation, and snapshot logging are unchanged.

**Migration cost, checked directly against the current file:** every one of the ~15 existing tests in `tests/paper/test_trader.py` patches `ggTrader.paper.trader.generate_signals` (not the new `generate_blended_signals`) and constructs its trader via the shared `_make_trader()` helper (no `dry_run` argument, so it will pick up the new `dry_run=True` default and silently stop asserting real `submit_buy`/`submit_sell` calls). Two changes make all of them pass again with no per-test logic rewrite:

1. **`_make_trader()` passes `dry_run=False` explicitly** — preserves every existing test's assumption that `broker.submit_buy`/`submit_sell` are actually called.
2. **A new helper `_blend(buys, sells, as_of, universe="sp500")`** wraps the old flat shape into the new `generate_blended_signals()` return shape, with the full weight (1.0) and scale (1.0) on one sleeve and the other two empty — this exactly reproduces today's flat-3% single-universe behavior (per the `test_sleeve_position_notional_matches_flat_when_full_weight_full_scale` invariant from Task 6), so every existing dollar-amount assertion (e.g. `broker.submit_buy.assert_called_once_with("MSFT", 3300.0)`) keeps passing unchanged. Every `@patch("ggTrader.paper.trader.generate_signals")` / `mock_signals.return_value = {...}` pair in the file becomes `@patch("ggTrader.paper.trader.generate_blended_signals")` / `mock_signals.return_value = _blend(buys=[...], sells=[...], as_of="...")`.

- [ ] **Step 1: Update `_make_trader()` and add the `_blend()` helper, migrate existing tests**

At the top of `tests/paper/test_trader.py`, change `_make_trader`'s `PaperTrader(broker, notifier)` call to `PaperTrader(broker, notifier, dry_run=False)`, and add:

```python
def _blend(buys, sells, as_of, universe="sp500"):
    """Wrap a flat buys/sells list into generate_blended_signals()'s shape,
    with full weight+scale on one sleeve -- reproduces today's flat-3%
    single-universe behavior exactly (see Task 6's collapse-to-flat test)."""
    all_universes = ("sp500", "midcap400", "nasdaq100")
    sleeves = {
        u: {"buys": buys if u == universe else [], "sells": sells if u == universe else [],
            "as_of": as_of, "universe_size": 100 if u == universe else 0, "gate": {}}
        for u in all_universes
    }
    return {
        "sleeves": sleeves,
        "weights": {u: (1.0 if u == universe else 0.0) for u in all_universes},
        "scale": 1.0,
        "rebalanced_today": False,
        "fallback_used": False,
    }
```

Then, throughout the file, replace every `@patch("ggTrader.paper.trader.generate_signals")` with `@patch("ggTrader.paper.trader.generate_blended_signals")`, and every
```python
mock_signals.return_value = {
    "buys": [...], "sells": [...], "as_of": "...", "universe_size": 100,
}
```
with
```python
mock_signals.return_value = _blend(buys=[...], sells=[...], as_of="...")
```
(same buys/sells/as_of values as before, per test). This is a mechanical rename across the existing ~15 test methods — no assertions on dollar amounts, call counts, or error handling change.

- [ ] **Step 2: Write the new dry-run test**

Add:

```python
@patch("ggTrader.paper.trader.get_latest_snapshot", return_value=None)
@patch("ggTrader.paper.trader.log_snapshot")
@patch("ggTrader.paper.trader.log_trade")
@patch("ggTrader.paper.trader.init_paper_schema")
class TestDryRun:
    @patch("ggTrader.paper.trader.generate_blended_signals")
    def test_dry_run_does_not_submit_orders(self, mock_signals, *_):
        """In dry_run mode (the new default), buys/sells are computed and
        reported but no real order is submitted."""
        from ggTrader.paper.trader import PaperTrader

        mock_signals.return_value = _blend(buys=["AAPL"], sells=[], as_of="2026-07-13")

        broker = MagicMock()
        broker.get_account.return_value = {
            "cash": 10000.0, "portfolio_value": 10000.0, "buying_power": 10000.0
        }
        broker.get_positions.return_value = {}
        notifier = MagicMock()
        notifier.trade_alert.return_value = True
        notifier.daily_summary.return_value = True

        trader = PaperTrader(broker, notifier)  # dry_run=True default

        result = trader.run()

        broker.submit_buy.assert_not_called()
        broker.submit_sell.assert_not_called()
        assert result["buys"] == ["AAPL"]
        notifier.send.assert_any_call(
            f"<b>🔍 DRY RUN buy:</b> AAPL (${round(10000.0 * 1.0 * 1.0 * 0.033, 0):.0f}, sleeve=sp500)"
        )
```

- [ ] **Step 3: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/paper/test_trader.py -v -k DryRun`
Expected: FAIL — `PaperTrader` doesn't accept `dry_run`, and still calls `generate_signals` not `generate_blended_signals`. (Every other existing test in this file will also currently fail after Step 1's migration, until Step 4 implements the real code — that's expected mid-task.)

- [ ] **Step 4: Implement**

Edit `src/ggTrader/paper/trader.py`:

1. Change the import: `from ggTrader.paper.signal_runner import generate_blended_signals`
2. Change `__init__`:

```python
    def __init__(
        self,
        broker: AlpacaBroker,
        notifier: TelegramNotifier,
        risk_cfg: RiskConfig | None = None,
        dry_run: bool = True,
    ) -> None:
        self._broker = broker
        self._notifier = notifier
        self._risk = RiskGuard(risk_cfg)
        self._dry_run = dry_run
```

3. In `run()`, replace `signals = generate_signals()` with:

```python
        try:
            blend = generate_blended_signals()
        except Exception as exc:
            self._notifier.send(f"Paper trading failed: signal generation error\n{exc}")
            raise

        if blend["fallback_used"]:
            self._notifier.send(
                "<b>⚠️ Overlay fallback:</b> rebalance-day data fetch failed; "
                "reusing last month's sleeve weights."
            )

        weights, scale = blend["weights"], blend["scale"]
        all_buys: list[tuple[str, str]] = []  # (symbol, sleeve)
        all_sells: list[str] = []
        gate_infos: dict[str, dict] = {}
        for universe, sleeve_signals in blend["sleeves"].items():
            for sym in sleeve_signals["buys"]:
                all_buys.append((sym, universe))
            all_sells.extend(sleeve_signals["sells"])
            gate_infos[universe] = sleeve_signals.get("gate", {})

        signals = {
            "buys": [sym for sym, _u in all_buys],
            "sells": sorted(set(all_sells)),
            "as_of": next(iter(blend["sleeves"].values()))["as_of"],
        }
```

4. Replace the gate-notification block's `gate = signals.get("gate", {})` with iteration over `gate_infos` (one notification line per sleeve with `gate_enabled`).

5. Replace the flat buy-sizing loop. Where the current code computes `notional = self._risk.position_notional(portfolio_value)` once and reuses it for every buy, instead compute per-sleeve position size and a per-sleeve slot cap (bounding how many concurrent positions that sleeve may hold — sizing itself does not depend on signal count, per Task 6's correction):

```python
        slot_caps = self._risk.sleeve_slot_caps(weights)
        # Tracks only positions opened THIS run -- pre-existing positions
        # from before this feature shipped aren't attributed to a sleeve
        # historically, so the sleeve cap is a soft governor on new buys
        # per run, not a hard total-per-sleeve limit. check_concentration
        # still applies per-symbol regardless of sleeve accounting.
        sleeve_open_count = {u: 0 for u in weights}

        buys_by_sleeve: dict[str, list[str]] = {}
        for sym, universe in all_buys:
            buys_by_sleeve.setdefault(universe, []).append(sym)

        slots_available = self._risk.max_new_positions(len(positions) - len(executed_sells))
        buys_attempted = 0
        for universe, syms in buys_by_sleeve.items():
            sleeve_notional = self._risk.sleeve_position_notional(
                portfolio_value, weights.get(universe, 0.0), scale
            )
            sleeve_cap = slot_caps.get(universe, 0)
            for symbol in syms:
                if symbol in positions:
                    continue
                if buys_attempted >= slots_available:
                    break
                if sleeve_open_count[universe] >= sleeve_cap:
                    break
                if self._risk.check_concentration(
                    symbol, positions, portfolio_value, prospective_notional=sleeve_notional
                ):
                    continue
                if self._dry_run:
                    executed_buys.append(symbol)
                    buys_attempted += 1
                    sleeve_open_count[universe] += 1
                    self._notifier.send(
                        f"<b>🔍 DRY RUN buy:</b> {symbol} (${sleeve_notional:.0f}, sleeve={universe})"
                    )
                    continue
                try:
                    oid = self._broker.submit_buy(symbol, sleeve_notional)
                    executed_buys.append(symbol)
                    buys_attempted += 1
                    sleeve_open_count[universe] += 1
                    pending_orders.append((oid, "BUY", symbol, sleeve_notional))
                except Exception as exc:
                    errors.append(f"BUY {symbol}: {exc}")
```

6. Guard the sell loop and the order-submission calls the same way: when `self._dry_run` is `True`, skip `self._broker.submit_sell(...)` and instead append to `executed_sells` with a dry-run notification, mirroring the buy branch above.

- [ ] **Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/paper/test_trader.py -v`
Expected: PASS — every existing test now uses `_blend(...)` + `dry_run=False` (Step 1) and reproduces its original flat-3% assertions exactly (per Task 6's collapse-to-flat invariant), plus the new `TestDryRun` test from Step 2.

- [ ] **Step 6: Commit**

```bash
git add src/ggTrader/paper/trader.py tests/paper/test_trader.py
git commit -m "feat(paper): wire blended signals + sleeve sizing into PaperTrader

PaperTrader now sources signals from all three sleeves via
generate_blended_signals(). Each position is sized at a fixed fraction
(position_pct) of its sleeve's allocated capital (weight x scale x
portfolio_value), independent of signal count -- matching the
fixed-fraction-per-entry convention each sleeve's own backtest curve
used, not a budget-divided-by-signal-count split. New dry_run flag
(default True) skips real order submission and Telegram-alerts intended
trades instead -- the burn-in mode from the spec."
```

---

### Task 8: Account leverage pre-flight check + CLI flag

**Files:**
- Modify: `src/ggTrader/paper/alpaca_broker.py`
- Modify: `src/ggTrader/cli/main.py`
- Test: `tests/paper/test_alpaca_broker.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `AlpacaBroker.get_account()` gains a `"multiplier"` key (float, from Alpaca's raw account `multiplier` field). `ggt paper` CLI gains a `--live` flag (default off, i.e. dry-run) that maps to `PaperTrader(..., dry_run=not args.live)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/paper/test_alpaca_broker.py` (match existing mocking conventions in that file):

```python
def test_get_account_includes_multiplier(mock_trading_client):
    mock_trading_client.get_account.return_value.multiplier = "1"
    mock_trading_client.get_account.return_value.cash = "10000.0"
    mock_trading_client.get_account.return_value.portfolio_value = "10000.0"
    mock_trading_client.get_account.return_value.buying_power = "10000.0"

    broker = AlpacaBroker()
    account = broker.get_account()

    assert account["multiplier"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/paper/test_alpaca_broker.py -v -k multiplier`
Expected: FAIL — `"multiplier"` key missing.

- [ ] **Step 3: Implement**

Edit `src/ggTrader/paper/alpaca_broker.py`'s `get_account`:

```python
    def get_account(self) -> dict:
        acct = self._client.get_account()
        return {
            "cash": float(acct.cash),
            "portfolio_value": float(acct.portfolio_value),
            "buying_power": float(acct.buying_power),
            "multiplier": float(acct.multiplier),
        }
```

Edit `src/ggTrader/cli/main.py`'s `paper` command to add a `--live` flag defaulting to dry-run, and have `run_paper_trading` (in `trader.py`) accept and forward a `dry_run` argument:

```python
def run_paper_trading(dry_run: bool = True) -> dict:
    """Convenience entry point: wire up broker + notifier and run.

    Also checks the account isn't margin-enabled, since the blend's
    target-vol overlay assumes max_leverage=1.0 (unlevered)."""
    broker = AlpacaBroker()
    account = broker.get_account()
    if account["multiplier"] > 1.0:
        raise RuntimeError(
            f"Account multiplier is {account['multiplier']}x (margin-enabled); "
            "the blend overlay assumes an unlevered (1.0x) account."
        )
    notifier = TelegramNotifier()
    trader = PaperTrader(broker, notifier, dry_run=dry_run)
    return trader.run()
```

(This replaces the existing `run_paper_trading()` in `trader.py` — add the `dry_run` param there, not in `alpaca_broker.py`.) Then in `cli/main.py`'s `paper` Typer command, add a `live: bool = typer.Option(False, "--live", help="Submit real paper orders instead of dry-run logging.")` parameter and call `run_paper_trading(dry_run=not live)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/paper/ -v`
Expected: PASS (full paper test suite).

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/paper/alpaca_broker.py src/ggTrader/paper/trader.py src/ggTrader/cli/main.py tests/paper/test_alpaca_broker.py
git commit -m "feat(paper): margin pre-flight check + --live CLI flag

get_account() now reports the account multiplier; run_paper_trading
refuses to proceed on a margin-enabled account (overlay assumes 1.0x).
ggt paper defaults to dry-run; --live must be passed explicitly to
submit real orders -- the cron job stays on dry-run until a human flips
this after the burn-in window (not done in this plan)."
```

---

## Post-plan: burn-in and cutover (not automated, not part of this plan)

This plan ships the blend live-trading capability entirely in dry-run mode
(`--live` not passed, cron untouched). Per the spec, run dry-run for a
couple of weeks comparing logged intended trades against expectations, then
manually edit `scripts/paper_trade.sh` to add `--live` once satisfied — a
deliberate, separate, human decision point, not something this plan
automates.

## Verification (after all tasks)

- Full test suite: `source .venv/bin/activate && pytest tests/paper/ -v` — expect all green, including every existing test updated for the new `universe_members_asof` patch target and `dry_run=True` default.
- Manual dry-run smoke test against real data (safe — no orders submitted): `source .venv/bin/activate && python ggt.py paper` (without `--live`) and confirm the Telegram output shows sleeve-tagged `DRY RUN buy` messages for whichever sleeves have signals today, and that `paper_rebalance_state` has a row (`SELECT * FROM paper_rebalance_state` via the `postgres` MCP or `mcp__postgres__query`).
- Confirm Invariant 1 holds by construction: `overlay.compute_sleeve_curve` and `signal_runner.generate_signals` both construct `EnsembleSignal(cfg)` with `cfg = LabConfig(min_history_bars=60)` and nothing else — grep both files for `EnsembleSignal(` to eyeball this after implementation, in addition to the Task 2 test that already asserts it via source inspection.
- Confirm Invariant 2 holds by construction: `sleeve_position_notional` (Task 6) and `sleeve_slot_caps` (Task 6) both take `portfolio_value` as a parameter passed fresh from `self._broker.get_account()` on every `run()` call (Task 7) — there is no persisted/cached portfolio value anywhere in this design (only `weights`/`scale` are persisted, in `paper_rebalance_state`), so a stale-portfolio-value bug isn't reachable by construction. Grep `trader.py` for `portfolio_value` after implementation to confirm it's only ever read from the current `get_account()` call, never from `persist`.
