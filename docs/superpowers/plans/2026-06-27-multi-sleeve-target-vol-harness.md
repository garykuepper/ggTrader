# Multi-Sleeve Target-Vol Research Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a gate-honest research harness that runs three reversion sleeves (sp500/midcap400/nasdaq100) through their own gated WFO, combines them with rolling inverse-vol weighting scaled to a target volatility (leverage capped), and reports vs the gated SP500 core.

**Architecture:** A pure, unit-tested `allocation.py` module holds the overlay math (rolling vol, inverse-vol weights, target-vol scaling, sleeve combination). `run_wfo` is extended to return a structured `WfoResult` exposing the gated OOS equity curve. A thin orchestration script wires sleeves → gated WFO → allocation → report.

**Tech Stack:** Python 3.12, pandas, numpy, vectorbt (via existing lab), pytest, ruff. Native `.venv` for research (per project convention).

## Global Constraints

- Strict ruff linting; absolute imports from `src` (`from ggTrader.lab... import`).
- Research runs natively via `.venv`, not Docker (Docker is live-only).
- Vol estimation must be out-of-sample: value at date *t* uses only data strictly before the rebalance — no look-ahead.
- Default target vol = `0.068` annualized (SP500 core's observed realized vol, fixed a-priori constant). Default `max_leverage = 2.0`. Default `window = 60`. Rebalance monthly (`"ME"`).
- Three sleeves are fixed parameters (`sp500`, `midcap400`, `nasdaq100`); no generalized N-sleeve framework. Russell 2000 excluded (backfill prerequisite).
- No borrow/transaction-cost modelling on leverage — report must print this caveat.

---

### Task 1: `trailing_realized_vol` in new `allocation.py`

**Files:**
- Create: `src/ggTrader/lab/allocation.py`
- Test: `tests/lab/test_allocation.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `trailing_realized_vol(returns: pd.Series, window: int = 60) -> pd.Series` — rolling annualized vol; `NaN` for the first `window-1` observations.

- [ ] **Step 1: Write the failing test**

```python
# tests/lab/test_allocation.py
import numpy as np
import pandas as pd
import pytest


def test_trailing_realized_vol_annualizes_and_warms_up():
    from ggTrader.lab.allocation import trailing_realized_vol

    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    rets = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01], index=idx)
    vol = trailing_realized_vol(rets, window=3)

    # First two are warmup (NaN), third onward defined
    assert vol.iloc[:2].isna().all()
    expected = rets.iloc[:3].std() * np.sqrt(252)
    assert vol.iloc[2] == pytest.approx(expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_allocation.py::test_trailing_realized_vol_annualizes_and_warms_up -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` (allocation not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# src/ggTrader/lab/allocation.py
"""Out-of-sample portfolio overlay math for the multi-sleeve research harness.

Pure functions only — no I/O, no DB, no WFO. All volatility estimates use
trailing data so a value at date t never depends on t's own future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def trailing_realized_vol(returns: pd.Series, window: int = 60) -> pd.Series:
    """Rolling annualized realized volatility from daily returns.

    Returns NaN for the first ``window - 1`` observations (warmup).
    """
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/lab/test_allocation.py::test_trailing_realized_vol_annualizes_and_warms_up -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/allocation.py tests/lab/test_allocation.py
git commit -m "feat(lab): trailing_realized_vol for multi-sleeve allocation"
```

---

### Task 2: `inverse_vol_weights`

**Files:**
- Modify: `src/ggTrader/lab/allocation.py`
- Test: `tests/lab/test_allocation.py`

**Interfaces:**
- Produces: `inverse_vol_weights(vols: dict[str, float]) -> dict[str, float]` — risk-parity weights summing to 1.0; drops sleeves with non-positive or NaN vol; returns equal weights if all invalid.

- [ ] **Step 1: Write the failing test**

```python
def test_inverse_vol_weights_favor_low_vol_and_sum_to_one():
    from ggTrader.lab.allocation import inverse_vol_weights

    w = inverse_vol_weights({"a": 0.04, "b": 0.08})
    assert w["a"] > w["b"]                 # lower vol -> higher weight
    assert w["a"] + w["b"] == pytest.approx(1.0)
    # a has half the vol of b -> twice the weight: 2/3 vs 1/3
    assert w["a"] == pytest.approx(2 / 3)


def test_inverse_vol_weights_drop_invalid_and_fallback_equal():
    from ggTrader.lab.allocation import inverse_vol_weights

    # NaN/zero dropped
    w = inverse_vol_weights({"a": 0.05, "b": float("nan"), "c": 0.0})
    assert set(w) == {"a"}
    assert w["a"] == pytest.approx(1.0)

    # all invalid -> equal weights across original keys
    w2 = inverse_vol_weights({"a": 0.0, "b": float("nan")})
    assert w2 == pytest.approx({"a": 0.5, "b": 0.5})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/lab/test_allocation.py::test_inverse_vol_weights_favor_low_vol_and_sum_to_one -v`
Expected: FAIL with `AttributeError`/`ImportError` (function not defined).

- [ ] **Step 3: Write minimal implementation**

```python
def inverse_vol_weights(vols: dict[str, float]) -> dict[str, float]:
    """Risk-parity weights: w_i = (1/vol_i) / sum(1/vol_j), summing to 1.0.

    Sleeves with non-positive or NaN vol are dropped. If none are valid,
    fall back to equal weights across the original keys.
    """
    valid = {k: v for k, v in vols.items() if v is not None and np.isfinite(v) and v > 0}
    if not valid:
        n = len(vols)
        return {k: 1.0 / n for k in vols}
    inv = {k: 1.0 / v for k, v in valid.items()}
    total = sum(inv.values())
    return {k: x / total for k, x in inv.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/lab/test_allocation.py -k inverse_vol -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/allocation.py tests/lab/test_allocation.py
git commit -m "feat(lab): inverse_vol_weights risk-parity weighting"
```

---

### Task 3: `target_vol_scale`

**Files:**
- Modify: `src/ggTrader/lab/allocation.py`
- Test: `tests/lab/test_allocation.py`

**Interfaces:**
- Produces: `target_vol_scale(blend_trailing_vol: float, target_vol: float, max_leverage: float = 2.0) -> float` — exposure multiplier `clip(target/blend, 0.0, max_leverage)`; returns `0.0` if blend vol is non-positive/NaN.

- [ ] **Step 1: Write the failing test**

```python
def test_target_vol_scale_levers_up_and_caps():
    from ggTrader.lab.allocation import target_vol_scale

    # blend vol 0.04, target 0.068 -> scale 1.7
    assert target_vol_scale(0.04, 0.068) == pytest.approx(1.7)
    # would be 3.4x but capped at 2.0
    assert target_vol_scale(0.02, 0.068, max_leverage=2.0) == pytest.approx(2.0)
    # already above target -> scale down below 1.0
    assert target_vol_scale(0.10, 0.068) == pytest.approx(0.68)
    # degenerate blend vol -> no exposure
    assert target_vol_scale(0.0, 0.068) == 0.0
    assert target_vol_scale(float("nan"), 0.068) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/lab/test_allocation.py::test_target_vol_scale_levers_up_and_caps -v`
Expected: FAIL (function not defined).

- [ ] **Step 3: Write minimal implementation**

```python
def target_vol_scale(
    blend_trailing_vol: float, target_vol: float, max_leverage: float = 2.0
) -> float:
    """Exposure multiplier to bring a blend's trailing vol to target_vol.

    clip(target_vol / blend_trailing_vol, 0.0, max_leverage). Returns 0.0 when
    blend vol is non-positive or NaN (cannot size safely).
    """
    if not np.isfinite(blend_trailing_vol) or blend_trailing_vol <= 0:
        return 0.0
    return float(np.clip(target_vol / blend_trailing_vol, 0.0, max_leverage))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/lab/test_allocation.py::test_target_vol_scale_levers_up_and_caps -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/allocation.py tests/lab/test_allocation.py
git commit -m "feat(lab): target_vol_scale exposure multiplier"
```

---

### Task 4: `combine_sleeves` (the overlay, with look-ahead guard)

**Files:**
- Modify: `src/ggTrader/lab/allocation.py`
- Test: `tests/lab/test_allocation.py`

**Interfaces:**
- Consumes: `trailing_realized_vol`, `inverse_vol_weights`, `target_vol_scale`.
- Produces: `combine_sleeves(sleeve_returns: pd.DataFrame, target_vol: float = 0.068, window: int = 60, max_leverage: float = 2.0, rebalance: str = "ME") -> tuple[pd.Series, pd.DataFrame]` — `(blended_daily_returns, diagnostics)`. Diagnostics is indexed by rebalance date with columns: one weight column per sleeve (`w_<sleeve>`), `blend_vol`, `scale`.

**Method:** At each rebalance date, using only returns strictly before that date, compute per-sleeve trailing vol (last value of `trailing_realized_vol`), inverse-vol weights, the provisional equal-overlay blend's trailing vol, and the target-vol scale. Apply those held weights × scale to each forward day's returns until the next rebalance. During warmup (fewer than `window` rows available before the first rebalance), use equal weights at scale 1.0.

- [ ] **Step 1: Write the failing test**

```python
def test_combine_sleeves_no_lookahead_and_diag_shape():
    from ggTrader.lab.allocation import combine_sleeves

    idx = pd.date_range("2021-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "sp500": rng.normal(0.0005, 0.01, 120),
            "midcap": rng.normal(0.0005, 0.012, 120),
            "nasdaq": rng.normal(0.0005, 0.009, 120),
        },
        index=idx,
    )
    blended, diag = combine_sleeves(df, target_vol=0.068, window=60, max_leverage=2.0)

    # blended is a daily series aligned to the input index
    assert isinstance(blended, pd.Series)
    assert blended.index.equals(df.index)

    # diagnostics carries a weight column per sleeve + blend_vol + scale
    for col in ("w_sp500", "w_midcap", "w_nasdaq", "blend_vol", "scale"):
        assert col in diag.columns

    # LOOK-AHEAD GUARD: mutating returns AFTER the last rebalance date must not
    # change any weight/scale decided at-or-before that date.
    last_reb = diag.index[-1]
    df2 = df.copy()
    df2.loc[df2.index > last_reb] += 5.0  # perturb only the future
    _, diag2 = combine_sleeves(df2, target_vol=0.068, window=60, max_leverage=2.0)
    pd.testing.assert_frame_equal(diag, diag2)


def test_combine_sleeves_warmup_is_equal_weight_scale_one():
    from ggTrader.lab.allocation import combine_sleeves

    idx = pd.date_range("2021-01-01", periods=40, freq="D")  # < window
    df = pd.DataFrame(
        {"sp500": 0.001, "midcap": 0.001, "nasdaq": 0.001}, index=idx
    )
    blended, diag = combine_sleeves(df, target_vol=0.068, window=60)

    # All-warmup: equal weights, scale 1.0 -> blended equals the equal-weight mean
    expected = df.mean(axis=1)
    pd.testing.assert_series_equal(blended, expected, check_names=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/lab/test_allocation.py -k combine_sleeves -v`
Expected: FAIL (function not defined).

- [ ] **Step 3: Write minimal implementation**

```python
def combine_sleeves(
    sleeve_returns: pd.DataFrame,
    target_vol: float = 0.068,
    window: int = 60,
    max_leverage: float = 2.0,
    rebalance: str = "ME",
) -> tuple[pd.Series, pd.DataFrame]:
    """Blend sleeve return streams with rolling inverse-vol weights scaled to
    a target volatility. Returns (blended_daily_returns, diagnostics).

    Out-of-sample: at each rebalance date, only returns strictly BEFORE that
    date inform the weights/scale that are then applied to forward days.
    Warmup (insufficient history before the first rebalance) uses equal weights
    at scale 1.0.
    """
    df = sleeve_returns.sort_index()
    sleeves = list(df.columns)
    equal = {s: 1.0 / len(sleeves) for s in sleeves}

    # Rebalance dates that fall within the sample.
    reb_dates = df.resample(rebalance).last().index
    reb_dates = [d for d in reb_dates if d <= df.index[-1]]

    weights = pd.Series(equal)
    scale = 1.0
    blended = pd.Series(0.0, index=df.index)
    diag_rows: list[dict] = []

    # Build held (weights, scale) decisions per rebalance using only past data.
    decisions: list[tuple[pd.Timestamp, pd.Series, float]] = []
    for d in reb_dates:
        past = df.loc[df.index < d]
        if len(past) < window:
            w, sc = pd.Series(equal), 1.0
        else:
            vols = {s: float(trailing_realized_vol(past[s], window).iloc[-1]) for s in sleeves}
            wd = inverse_vol_weights(vols)
            w = pd.Series({s: wd.get(s, 0.0) for s in sleeves})
            prov = (past[sleeves] * w).sum(axis=1)
            blend_vol = float(trailing_realized_vol(prov, window).iloc[-1])
            sc = target_vol_scale(blend_vol, target_vol, max_leverage)
        decisions.append((d, w, sc))
        row = {f"w_{s}": w[s] for s in sleeves}
        row["blend_vol"] = (
            float(trailing_realized_vol((past[sleeves] * w).sum(axis=1), window).iloc[-1])
            if len(past) >= window
            else float("nan")
        )
        row["scale"] = sc
        diag_rows.append(row)

    # Apply each decision forward until the next rebalance.
    for i, (d, w, sc) in enumerate(decisions):
        nxt = decisions[i + 1][0] if i + 1 < len(decisions) else None
        mask = df.index >= d if nxt is None else (df.index >= d) & (df.index < nxt)
        blended.loc[mask] = (df.loc[mask, sleeves] * w).sum(axis=1) * sc

    # Days before the first rebalance: equal weight, scale 1.0.
    if decisions:
        pre = df.index < decisions[0][0]
        blended.loc[pre] = (df.loc[pre, sleeves] * pd.Series(equal)).sum(axis=1)
    else:
        blended = (df[sleeves] * pd.Series(equal)).sum(axis=1)

    diag = pd.DataFrame(diag_rows, index=pd.Index([d for d, _, _ in decisions], name="rebalance"))
    return blended, diag
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/lab/test_allocation.py -k combine_sleeves -v`
Expected: PASS (both tests)

> Note: if `test_combine_sleeves_warmup_is_equal_weight_scale_one` fails because all 40 rows precede the first month-end rebalance, confirm the `pre`-rebalance branch covers them (40 daily rows starting 2021-01-01 means the first `"ME"` date is 2021-01-31, so rows before it use equal weight). The assertion compares against `df.mean(axis=1)`.

- [ ] **Step 5: Run the full allocation suite + ruff**

Run: `python -m pytest tests/lab/test_allocation.py -v && ruff check src/ggTrader/lab/allocation.py tests/lab/test_allocation.py`
Expected: all PASS, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/ggTrader/lab/allocation.py tests/lab/test_allocation.py
git commit -m "feat(lab): combine_sleeves rolling inverse-vol + target-vol overlay"
```

---

### Task 5: `run_wfo` returns `WfoResult`

**Files:**
- Modify: `src/ggTrader/lab/wfo.py` (add `WfoResult`; change `run_wfo` return ~line 653)
- Modify: `scripts/midcap_research.py:202-217` (read `result.table`)
- Test: `tests/lab/test_wfo.py`

**Interfaces:**
- Produces: `WfoResult(oos_equity: pd.Series, fold_results: list[dict], live_params: dict, table: str)`. `run_wfo(...)` now returns `WfoResult` (still prints the table). `oos_equity` is the continuous gated OOS curve (empty `pd.Series(dtype=float)` if no folds produced curves).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/lab/test_wfo.py
def test_run_wfo_returns_wforesult_namedtuple():
    from ggTrader.lab.wfo import WfoResult

    r = WfoResult(
        oos_equity=__import__("pandas").Series(dtype=float),
        fold_results=[],
        live_params={},
        table="x",
    )
    assert r.table == "x"
    assert list(r._fields) == ["oos_equity", "fold_results", "live_params", "table"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/lab/test_wfo.py::test_run_wfo_returns_wforesult_namedtuple -v`
Expected: FAIL with `ImportError: cannot import name 'WfoResult'`.

- [ ] **Step 3: Add `WfoResult` and change the return**

In `src/ggTrader/lab/wfo.py`, add near the other `NamedTuple` definitions (e.g. after `AnchorSet`):

```python
class WfoResult(NamedTuple):
    """Structured result of a walk-forward run (the table is also printed)."""

    oos_equity: pd.Series
    fold_results: List[Dict[str, Any]]
    live_params: Dict[str, Any]
    table: str
```

In `run_wfo`, ensure `oos_equity` exists in the empty-curve branch. The block starting `if oos_curves:` (~line 607) already assigns `oos_equity` when curves exist; add an `else` default. Then change the tail:

```python
    if oos_curves:
        oos_equity = pd.concat(oos_curves)
        oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")]
        oos_metrics = curve_stats(oos_equity)
        # ... existing spy_metrics logic unchanged ...
    else:
        oos_equity = pd.Series(dtype=float)
        oos_metrics = {
            "sharpe": float("nan"),
            "cagr_pct": float("nan"),
            "max_drawdown_pct": float("nan"),
        }
        spy_metrics = oos_metrics.copy()
```

Replace the final two lines (`print(table)` / `return table`) with:

```python
    print(table)
    return WfoResult(
        oos_equity=oos_equity,
        fold_results=fold_results,
        live_params=live,
        table=table,
    )
```

- [ ] **Step 4: Update the midcap caller**

In `scripts/midcap_research.py`, change line 202 `table = run_wfo(` to `result = run_wfo(` and line 217 `mid = parse_table(table)` to `mid = parse_table(result.table)`.

- [ ] **Step 5: Run tests to verify pass**

Run: `python -m pytest tests/lab/test_wfo.py -v`
Expected: PASS (new test + all 35 existing).

- [ ] **Step 6: Commit**

```bash
git add src/ggTrader/lab/wfo.py tests/lab/test_wfo.py scripts/midcap_research.py
git commit -m "feat(lab): run_wfo returns WfoResult exposing gated OOS equity curve"
```

---

### Task 6: `multi_sleeve_research.py` orchestration + end-to-end run

**Files:**
- Create: `scripts/multi_sleeve_research.py`

**Interfaces:**
- Consumes: `combine_sleeves` (Task 4), `run_wfo` → `WfoResult` (Task 5), `equity_universe_between`, `load_ohlcv`, `STOCK_BASE_CONFIG` (`ggTrader.lab.data`), `curve_stats` (`ggTrader.lab.metrics`), `build_grid` (`ggTrader.lab.sweep`), `EnsembleSignal`, `LabConfig`.
- Produces: a CLI script (`python scripts/multi_sleeve_research.py [--target-vol F] [--max-leverage F] [--window N]`) that prints the comparison report.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python
"""Multi-sleeve target-vol research harness.

Runs sp500 / midcap400 / nasdaq100 through their own gated WFO, blends the
gate-honest OOS curves with rolling inverse-vol weighting scaled to a target
volatility (leverage capped), and reports vs the gated SP500 core.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from ggTrader.lab.allocation import combine_sleeves
from ggTrader.lab.data import STOCK_BASE_CONFIG, equity_universe_between, load_ohlcv
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.wfo import run_wfo

EVAL_START = "2021-01-31"
SLEEVES = ("sp500", "midcap400", "nasdaq100")


def _row(label: str, s: dict) -> str:
    return (
        f"| {label} | {s['cagr_pct']:.2f}% | {s['sharpe']:.2f} | {s['sortino']:.2f} "
        f"| {s['ann_vol_pct']:.2f}% | {s['max_drawdown_pct']:.2f}% |"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-sleeve target-vol research harness")
    ap.add_argument("--target-vol", type=float, default=0.068)
    ap.add_argument("--max-leverage", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=60)
    args = ap.parse_args()

    cfg = LabConfig()
    eval_start = pd.Timestamp(EVAL_START, tz="UTC")
    eval_end = pd.Timestamp.now(tz="UTC").normalize()
    eval_start_str, eval_end_str = str(eval_start.date()), str(eval_end.date())

    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start_str = str((eval_start - pd.Timedelta(days=warmup_days)).date())

    members: dict[str, list[str]] = {
        s: equity_universe_between(eval_start, eval_end, universe=s) for s in SLEEVES
    }
    all_symbols = sorted({sym for ms in members.values() for sym in ms} | {"SPY"})
    print(f"[load] {len(all_symbols)} symbols across {len(SLEEVES)} sleeves", flush=True)
    ohlcv = load_ohlcv(all_symbols, data_start_str, eval_end_str, use_negative_cache=True)
    available = set(ohlcv.columns.get_level_values(0))
    grid = build_grid(EnsembleSignal, cfg)

    curves: dict[str, pd.Series] = {}
    for s in SLEEVES:
        syms = [x for x in members[s] if x in available]
        print(f"[wfo] {s}: {len(syms)} symbols", flush=True)
        t0 = time.time()
        result = run_wfo(
            "ensemble",
            EnsembleSignal,
            cfg,
            ohlcv[syms],
            ohlcv["SPY"]["close"].dropna(),
            eval_start=eval_start_str,
            eval_end=eval_end_str,
            market="equity",
            base_config=dict(STOCK_BASE_CONFIG),
            grid=grid,
        )
        curves[s] = result.oos_equity
        print(f"[wfo] {s} done in {time.time() - t0:.0f}s", flush=True)

    # Align gated OOS curves on common dates -> daily returns.
    common = curves["sp500"].index
    for s in SLEEVES:
        common = common.intersection(curves[s].index)
    returns_df = pd.DataFrame(
        {s: curves[s].reindex(common).pct_change() for s in SLEEVES}
    ).dropna()

    blended, diag = combine_sleeves(
        returns_df,
        target_vol=args.target_vol,
        window=args.window,
        max_leverage=args.max_leverage,
    )
    blend_eq = (1.0 + blended).cumprod() * float(STOCK_BASE_CONFIG["START_CASH"])

    # ── Report ──
    print("\n" + "=" * 80)
    print("MULTI-SLEEVE TARGET-VOL RESEARCH REPORT (gate-honest)")
    print("=" * 80)
    print(f"Eval: {eval_start_str} to {eval_end_str}")
    print(f"target_vol={args.target_vol}  max_leverage={args.max_leverage}x  window={args.window}d")
    print("\nGated sleeve OOS correlation matrix:")
    print(returns_df.corr().round(4).to_string())
    print(
        f"\nRealized leverage: avg {diag['scale'].mean():.2f}x  max {diag['scale'].max():.2f}x"
    )

    print("\n| Strategy | CAGR | Sharpe | Sortino | Vol | Max DD |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    print(_row("S&P 500 (gated core)", curve_stats(curves["sp500"].reindex(common))))
    print(_row("MidCap 400 (gated)", curve_stats(curves["midcap400"].reindex(common))))
    print(_row("Nasdaq-100 (gated)", curve_stats(curves["nasdaq100"].reindex(common))))
    print(_row("Inverse-vol + target-vol blend", curve_stats(blend_eq)))

    print(
        "\nCaveats: (1) sleeve curves are gate-honest (anchor/halt applied) but the "
        "leverage carries NO borrow or transaction cost in this model. (2) target_vol "
        "is a fixed a-priori constant, not re-fit per period."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint**

Run: `ruff check scripts/multi_sleeve_research.py`
Expected: All checks passed.

- [ ] **Step 3: End-to-end run (smoke)**

Run: `source .venv/bin/activate && python scripts/multi_sleeve_research.py 2>&1 | tail -25`
Expected: prints the correlation matrix, realized-leverage line, the performance table with four rows, and the caveats. (Runtime is several minutes — three gated WFO legs.)

- [ ] **Step 4: Commit**

```bash
git add scripts/multi_sleeve_research.py
git commit -m "feat(scripts): multi-sleeve target-vol research harness orchestration"
```

---

## Self-Review

**Spec coverage:**
- `allocation.py` four functions → Tasks 1–4. ✓
- OOS-honest rolling vol + look-ahead guard → Task 4 test. ✓
- `run_wfo` → `WfoResult` with gated OOS curve + caller update → Task 5. ✓
- Orchestration script (load → gated WFO per sleeve → combine → report) → Task 6. ✓
- Report contents (core baseline, sleeves, blend, correlation matrix, realized leverage, caveats) → Task 6 Step 1. ✓
- Defaults (target 0.068, leverage 2.0, window 60, monthly) → Global Constraints + Tasks 3/4/6. ✓
- Testing incl. look-ahead guard → Task 4. ✓
- Russell 2000 exclusion / fixed 3 sleeves / no cost model → Global Constraints + report caveat. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `combine_sleeves` returns `(pd.Series, pd.DataFrame)` consistently (Tasks 4, 6). `WfoResult` fields `oos_equity / fold_results / live_params / table` used identically in Tasks 5 and 6 (`result.oos_equity`, `result.table`). Diagnostics columns `w_<sleeve>`, `blend_vol`, `scale` consistent (Task 4 test ↔ Task 6 `diag['scale']`). ✓
