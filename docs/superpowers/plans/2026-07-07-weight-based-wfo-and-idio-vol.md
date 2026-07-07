# Weight-Based WFO Gating + Cross-Sectional Idiosyncratic Volatility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (Phase A) Generalize the gated `ggt lab --wfo` harness (`wfo.py`) so
weight-based strategies (`target_kind="weights"`) get real NDH/DSR-gated
walk-forward verdicts, validated against the existing `xs_momentum`/
`dual_momentum` strategies. (Phase B) Build Candidate B from
`docs/2026-07-06_strategy_recommendations.md` — Cross-Sectional Idiosyncratic
Volatility (`idio_vol`) — as a new weight-based lab strategy, and get its
first honest full-SP500 WFO verdict now that the harness supports it.

**Architecture:** `wfo.py` currently hardcodes `simulate_signals`/
`sweep_signals` inside `_sweep_fold` and the fold loop's OOS-curve rebuild —
weight strategies (`xs_momentum`, `dual_momentum`) can only run through the
ungated `--sweep` path or the plain `walkforward()` (both already dispatch
correctly on `target_kind`/`hasattr(strat, "sweep_signals")` — see
`sweep.py:218-263` and `harness.py:96-118`). Phase A adds a `_sweep_fold_weights`
function that mirrors `sweep.py`'s weight branch (per-combo `LabConfig`,
`select`/`to_targets`/`simulate_weights`) but scoped to a WFO fold window, then
a thin dispatcher used at every `_sweep_fold` call site in `wfo.py`. This also
lets the OOS-curve rebuild block reuse the equity curve `_sweep_fold`/
`_sweep_fold_weights` already computed instead of re-simulating from scratch
(a pre-existing duplication in the signal path, fixed as a byproduct). Phase B
adds `idio_vol.py` following the `momentum.py` `Strategy` protocol pattern,
using the eligible universe's own equal-weighted return as the market factor
(the same convention `simulate.py::compute_vol_scalar` already uses) instead
of wiring in an external SPY series — avoiding a second, unrelated interface
change to `Strategy.select()`.

**Tech Stack:** Python, pandas, numpy, pytest, vectorbt 0.28.5 (via the
existing `simulate_weights`/`Portfolio.from_orders` path — untouched by this
plan).

## Global Constraints

- Follow `ggTrader/agents.md` coding standards: strict ruff linting,
  vectorization-first (no per-row Python loops over time; per-symbol loops
  with vectorized per-symbol rolling ops are the established convention —
  see `simulate.py::compute_atr_stop`), absolute imports from `src`.
- Long-only: `idio_vol` selects the bottom-quintile (lowest idiosyncratic
  variance) and equal-weights them; it does not short the top quintile. The
  recommendations doc mentions a long/short variant, but the project's
  existing weight strategies (`xs_momentum`, `dual_momentum`) and live
  infrastructure are long-only with no margin/short plumbing — shorting is
  out of scope for this plan.
- Run `pytest tests/lab/ -q` and `ruff check src/ggTrader/lab/` before every
  commit.
- No changes to `gates.py`, `simulate.py`, or `harness.py` — Phase A only
  touches `wfo.py` and `cli.py`.

---

## Phase A: Weight-Based Strategies Through the Gated WFO Harness

### File Structure

- **Modify `src/ggTrader/lab/wfo.py`**: add `_sweep_fold_weights` (weight-strategy
  analog of `_sweep_fold`) and `_sweep_fold_dispatch` (routes by
  `target_kind`); thread a `universe_fn` parameter through `run_wfo`,
  `compute_anchor_set`, and `select_live_params`; replace the OOS-curve
  rebuild block's hardcoded `sweep_signals`/`simulate_signals` call with the
  equity curve already returned by the dispatcher.
- **Modify `src/ggTrader/lab/cli.py`**: drop the `--wfo` signal-strategies-only
  restriction; pass `universe_fn` into `run_wfo`.
- **Modify `tests/lab/test_wfo.py`**: add a `_TinyWeight` synthetic strategy
  (mirrors the existing `_TinySignal`) and tests proving weight combos flow
  through NDH/DSR gates end-to-end.

---

### Task 1: `_sweep_fold_weights` — weight-strategy analog of `_sweep_fold`

**Files:**
- Modify: `src/ggTrader/lab/wfo.py`
- Test: `tests/lab/test_wfo.py`

**Interfaces:**
- Consumes: `rebalance_dates` (`ggTrader.lab.data`), `simulate_weights`
  (`ggTrader.lab.simulate`), `extract_close`
  (`ggTrader.lab.strategies.indicators`), `LabConfig` (`ggTrader.lab.strategy`),
  `build_combo_lookup`/`combo_name` (`ggTrader.lab.sweep`, already imported in
  `wfo.py`), `curve_stats` (`ggTrader.lab.metrics`, already imported).
- Produces: `UniverseFn = Callable[[pd.Timestamp, pd.DataFrame], List[str]]`
  (same protocol as `ggTrader.lab.harness.UniverseFn`) and
  `_sweep_fold_weights(strategy_name: str, strategy_cls: Type, cfg: LabConfig,
  ohlcv: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp,
  base_config: Dict[str, Any], grid: List[Dict[str, Any]], universe_fn:
  UniverseFn) -> tuple[List[Dict[str, Any]], Dict[str, pd.Series]]` — same
  return contract as `_sweep_fold`.

- [ ] **Step 1: Write the failing test**

Add to `tests/lab/test_wfo.py`, near the top (after `_TinySignal`):

```python
class _TinyWeight:
    """Minimal weight strategy for testing: equal-weight top param_a symbols."""

    name = "tinyweight"
    target_kind = "weights"

    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def sweep_params(cls):
        return {"top_n": [1, 2]}

    def select(self, asof, data, eligible):
        data = data.loc[:asof]
        chosen = sorted(eligible)[: self.cfg.top_n]
        if not chosen:
            return []
        w = 1.0 / len(chosen)
        return [{"symbol": s, "weight": w} for s in chosen]

    def to_targets(self, plans, data):
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0
            for sel in plans[asof]:
                targets.loc[bar, sel["symbol"]] = float(sel["weight"])
        return targets


def _tiny_weight_universe_fn(asof, past):
    return sorted(past.columns.get_level_values(0).unique())


def test_sweep_fold_weights_basic():
    from ggTrader.lab.wfo import _sweep_fold_weights

    symbols = ["X", "Y", "Z"]
    n = 300
    ohlcv = _ohlcv(symbols, n)
    cfg = LabConfig(top_n=2, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
    }
    window_start = ohlcv.index[100]
    window_end = ohlcv.index[250]
    grid = [{"top_n": 1}, {"top_n": 2}]

    results, all_eq = _sweep_fold_weights(
        "tinyweight",
        _TinyWeight,
        cfg,
        ohlcv,
        window_start,
        window_end,
        base_config,
        grid,
        _tiny_weight_universe_fn,
    )

    assert len(results) == 2
    for r in results:
        assert "combo" in r and "params" in r and "sharpe" in r
    assert set(all_eq.keys()) == {r["combo"] for r in results}
    for eq in all_eq.values():
        assert eq.notna().sum() > 0


def test_sweep_fold_weights_no_rebalance_dates_returns_empty():
    from ggTrader.lab.wfo import _sweep_fold_weights

    symbols = ["X", "Y"]
    ohlcv = _ohlcv(symbols, 50)
    cfg = LabConfig(top_n=1, min_history_bars=5)
    base_config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1d"}
    # A window with no full month inside it produces no rebalance dates.
    window_start = ohlcv.index[0]
    window_end = ohlcv.index[1]
    grid = [{"top_n": 1}]

    results, all_eq = _sweep_fold_weights(
        "tinyweight", _TinyWeight, cfg, ohlcv, window_start, window_end,
        base_config, grid, _tiny_weight_universe_fn,
    )
    assert results == []
    assert all_eq == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lab/test_wfo.py -k sweep_fold_weights -v`
Expected: FAIL with `ImportError: cannot import name '_sweep_fold_weights'
from ggTrader.lab.wfo`.

- [ ] **Step 3: Implement `_sweep_fold_weights`**

Add to `src/ggTrader/lab/wfo.py`, after `_sweep_fold` (before
`compute_anchor_set`):

```python
from typing import Callable

UniverseFn = Callable[[pd.Timestamp, pd.DataFrame], List[str]]


def _sweep_fold_weights(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    universe_fn: UniverseFn,
) -> tuple[List[Dict[str, Any]], Dict[str, pd.Series]]:
    """Weight-strategy analog of _sweep_fold: plan + simulate every combo on one window.

    Mirrors sweep.py's weight branch (run_sweep), scoped to [window_start,
    window_end] instead of the full eval span, using a caller-supplied
    universe_fn (the same UniverseFn protocol harness.py's walkforward() uses)
    instead of a hardcoded universe string.
    """
    from ggTrader.lab.data import rebalance_dates
    from ggTrader.lab.simulate import simulate_weights
    from ggTrader.lab.strategies.indicators import extract_close

    ohlcv_window = ohlcv.loc[:window_end]
    symbols = sorted(ohlcv_window.columns.get_level_values(0).unique())
    prices = extract_close(ohlcv_window, symbols)

    dates = rebalance_dates(ohlcv_window.index, window_start, window_end)
    if not dates:
        return [], {}

    all_targets: Dict[str, pd.DataFrame] = {}
    for combo_params in grid:
        merged = {
            "top_n": cfg.top_n,
            "lookback": cfg.lookback,
            "skip": cfg.skip,
            **combo_params,
        }
        combo_cfg = LabConfig(
            top_n=int(merged.get("top_n", cfg.top_n)),
            lookback=int(merged.get("lookback", cfg.lookback)),
            skip=int(merged.get("skip", cfg.skip)),
            min_history_bars=cfg.min_history_bars,
            max_stocks=cfg.max_stocks,
            max_sector_count=cfg.max_sector_count,
        )
        strat = strategy_cls(combo_cfg)
        plans: Dict[pd.Timestamp, Any] = {}
        for asof in dates:
            past = ohlcv_window.loc[:asof]
            eligible = universe_fn(asof, past)
            plans[asof] = strat.select(asof, past, eligible)
        target = strat.to_targets(plans, ohlcv_window)
        key = combo_name(strategy_name, combo_params)
        all_targets[key] = target

    start_cash = float(base_config["START_CASH"])
    _rets, eq_df, _diags = simulate_weights(all_targets, prices, base_config)

    results: List[Dict[str, Any]] = []
    all_eq: Dict[str, pd.Series] = {}
    combo_lookup = build_combo_lookup(strategy_name, grid)
    for key in all_targets:
        eq_series = eq_df[key]
        all_eq[key] = eq_series
        eq_window = eq_series.loc[window_start:window_end].dropna()
        if len(eq_window) < 2:
            continue
        eq_scaled = start_cash * (eq_window / eq_window.iloc[0])
        metrics = curve_stats(eq_scaled)
        results.append({"combo": key, "params": combo_lookup[key], **metrics})
    return results, all_eq
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/lab/test_wfo.py -k sweep_fold_weights -v`
Expected: PASS on both new tests.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/wfo.py tests/lab/test_wfo.py
git commit -m "feat(lab): add _sweep_fold_weights, the weight-strategy WFO fold path"
```

---

### Task 2: Dispatch by `target_kind` in `run_wfo`, `compute_anchor_set`, `select_live_params`

**Files:**
- Modify: `src/ggTrader/lab/wfo.py`
- Test: `tests/lab/test_wfo.py`

**Interfaces:**
- Consumes: `_sweep_fold_weights` (Task 1), `_sweep_fold` (existing),
  `UniverseFn` (Task 1).
- Produces: `_sweep_fold_dispatch(strategy_name: str, strat_instance: Any,
  strategy_cls: Type, cfg: LabConfig, ohlcv: pd.DataFrame, window_start:
  pd.Timestamp, window_end: pd.Timestamp, base_config: Dict[str, Any], grid:
  List[Dict[str, Any]], universe_fn: UniverseFn | None) -> tuple[List[Dict[str,
  Any]], Dict[str, pd.Series]]`; `run_wfo(..., universe_fn: UniverseFn | None
  = None)`; `compute_anchor_set(..., universe_fn: UniverseFn | None = None)`;
  `select_live_params(..., universe_fn: UniverseFn | None = None)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/lab/test_wfo.py`:

```python
def test_run_wfo_weight_strategy_integration():
    """Weight strategies flow through the full gated WFO: folds, gates, table."""
    symbols = ["X", "Y", "Z"]
    n = 252 * 7
    ohlcv = _ohlcv(symbols, n)
    spy_close = ohlcv["X"]["close"].copy()
    cfg = LabConfig(top_n=2, lookback=20, skip=5, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
    }
    eval_start = ohlcv.index[0]
    eval_end = ohlcv.index[-1]
    grid = [{"top_n": 1}, {"top_n": 2}]

    output = run_wfo(
        "tinyweight",
        _TinyWeight,
        cfg,
        ohlcv,
        spy_close,
        str(eval_start.date()),
        str(eval_end.date()),
        "test",
        base_config,
        grid,
        universe_fn=_tiny_weight_universe_fn,
    )
    assert "WFO:" in output.table
    assert "OOS Aggregate:" in output.table
    assert "Recommended Live Params" in output.table


def test_run_wfo_weight_strategy_without_universe_fn_raises():
    """A weight strategy with no universe_fn is a caller bug, not a silent no-op."""
    symbols = ["X", "Y"]
    ohlcv = _ohlcv(symbols, 252 * 2)
    spy_close = ohlcv["X"]["close"].copy()
    cfg = LabConfig(top_n=1, min_history_bars=10)
    base_config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1d"}
    grid = [{"top_n": 1}]

    with pytest.raises(ValueError, match="universe_fn"):
        run_wfo(
            "tinyweight", _TinyWeight, cfg, ohlcv, spy_close,
            str(ohlcv.index[0].date()), str(ohlcv.index[-1].date()),
            "test", base_config, grid,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lab/test_wfo.py -k weight_strategy -v`
Expected: FAIL — `_TinyWeight` has no `sweep_signals`, so the current
hardcoded `_sweep_fold` call raises `AttributeError` inside `run_wfo`.

- [ ] **Step 3: Implement the dispatcher and thread `universe_fn` through**

Add the dispatcher to `src/ggTrader/lab/wfo.py`, directly after
`_sweep_fold_weights`:

```python
def _sweep_fold_dispatch(
    strategy_name: str,
    strat_instance: Any,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    universe_fn: "UniverseFn | None",
) -> tuple[List[Dict[str, Any]], Dict[str, pd.Series]]:
    """Route to the signal or weight fold-sweep path by strategy target_kind."""
    if getattr(strat_instance, "target_kind", "signals") == "weights":
        if universe_fn is None:
            raise ValueError(
                f"{strategy_name}: weight strategies require universe_fn "
                "(e.g. lambda asof, past: eligible_at(asof, past, cfg, universe=...)[0])"
            )
        return _sweep_fold_weights(
            strategy_name, strategy_cls, cfg, ohlcv, window_start, window_end,
            base_config, grid, universe_fn,
        )
    return _sweep_fold(
        strategy_name, strat_instance, ohlcv, window_start, window_end, base_config, grid
    )
```

Now update every call site to go through the dispatcher instead of
`_sweep_fold` directly, and thread `universe_fn` through the three public
functions.

In `compute_anchor_set` — change the signature and its one `_sweep_fold` call:

```python
def compute_anchor_set(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    risk_free_rate: float = 4.0,
    universe_fn: "UniverseFn | None" = None,
) -> AnchorSet:
    """Derive the anchor set: minimize max drawdown subject to CAGR > risk-free.

    Runs all grid combos over the full available history and picks the combo
    with the smallest absolute drawdown among those with CAGR > risk_free_rate.
    Falls back to the least-drawdown combo if none clears the CAGR constraint.
    """
    strat_instance = strategy_cls(cfg)
    full_start = ohlcv.index[0]
    full_end = ohlcv.index[-1]

    metrics_list, _eq = _sweep_fold_dispatch(
        strategy_name,
        strat_instance,
        strategy_cls,
        cfg,
        ohlcv,
        full_start,
        full_end,
        base_config,
        grid,
        universe_fn,
    )
```

(Leave the rest of `compute_anchor_set` unchanged — only the call and
signature change.)

In `run_wfo`, update the signature:

```python
def run_wfo(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    spy_close: pd.Series,
    eval_start: str,
    eval_end: str,
    market: str,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    ndh_threshold: float = 0.85,
    dsr_threshold: float = 0.80,
    universe_fn: "UniverseFn | None" = None,
) -> str:
```

Update the `compute_anchor_set` call inside `run_wfo` to pass `universe_fn`:

```python
    anchor = compute_anchor_set(
        strategy_name,
        strategy_cls,
        cfg,
        ohlcv,
        base_config,
        grid,
        universe_fn=universe_fn,
    )
```

Update the train-window `_sweep_fold` call in the fold loop to go through the
dispatcher:

```python
        # Train: sweep all combos on train window
        train_metrics, train_eq = _sweep_fold_dispatch(
            strategy_name,
            strat_instance,
            strategy_cls,
            cfg,
            ohlcv,
            fold.train_start,
            fold.train_end,
            base_config,
            grid,
            universe_fn,
        )
```

Replace the entire OOS test-window block (the existing `_sweep_fold` call
plus the manual `sweep_signals`/`simulate_signals` equity rebuild right after
it) with a single dispatcher call that reuses its own returned equity curve —
this deletes the pre-existing signal-only duplication (the old code computed
`test_metrics` via `_sweep_fold`, then separately re-simulated the same combo
a second time to build the OOS curve):

```python
        # Test: simulate deploy params on data up to test_end, score test window
        winner_grid = [deploy_params]
        test_metrics, test_eq = _sweep_fold_dispatch(
            strategy_name,
            strat_instance,
            strategy_cls,
            cfg,
            ohlcv,
            fold.test_start,
            fold.test_end,
            base_config,
            winner_grid,
            universe_fn,
        )
        oos_score = 0.0
        if test_metrics:
            oos_score = composite_score(test_metrics)[0]
            full_key = combo_name(strategy_name, deploy_params)
            eq_test = test_eq.get(full_key, pd.Series(dtype=float))
            eq_test = eq_test.loc[fold.test_start : fold.test_end].dropna()
            if len(eq_test) > 0:
                normalized = oos_running_value * (eq_test / eq_test.iloc[0])
                oos_curves.append(normalized)
                oos_running_value = float(normalized.iloc[-1])
```

This removes the need for the `split_params`/`stop_p`/`ohlcv_arg`/
`simulate_signals` import-and-call block that previously followed — delete
those now-unused lines (the `_r, eq, _d = simulate_signals(...)` call and its
`test_ohlcv`/`symbols`/`prices`/`signal_combos`/`key` setup). `split_params`
and `simulate_signals` may become unused imports in `wfo.py` at this point —
check with `ruff check` in Task 4 and remove them from the import block if so
(the signal path's own `sweep_signal_group` inside `_sweep_fold` already
imports `simulate_signals` locally, so the module-level import is likely
dead after this change).

Finally, update `select_live_params`:

```python
def select_live_params(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    eval_end: str,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    fold_winners: List[Dict[str, Any]],
    universe_fn: "UniverseFn | None" = None,
) -> Dict[str, Any]:
    """Train on the most recent TRAIN_MONTHS window and pick the durable winner.

    Prefers combos proven across walk-forward folds over those that merely score
    best on the most recent window — see :func:`_pick_live_winner`.
    """
    eval_end_ts = pd.Timestamp(eval_end, tz="UTC")
    live_train_start = eval_end_ts - pd.DateOffset(months=TRAIN_MONTHS)
    strat_instance = strategy_cls(cfg)

    train_metrics, _train_eq = _sweep_fold_dispatch(
        strategy_name,
        strat_instance,
        strategy_cls,
        cfg,
        ohlcv,
        live_train_start,
        eval_end_ts,
        base_config,
        grid,
        universe_fn,
    )
```

And update `run_wfo`'s call to `select_live_params` to pass `universe_fn`:

```python
    live = select_live_params(
        strategy_name,
        strategy_cls,
        cfg,
        ohlcv,
        eval_end,
        base_config,
        grid,
        fold_winners,
        universe_fn=universe_fn,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/lab/test_wfo.py -v`
Expected: PASS on all tests, including the two new ones and every
pre-existing signal-strategy test (`test_run_wfo_integration`,
`test_run_wfo_reports_gate_status`, `test_run_wfo_anchor_fallback_on_gate_failure`,
etc.) — the signal path's behavior must be byte-for-byte unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/wfo.py tests/lab/test_wfo.py
git commit -m "feat(lab): dispatch WFO fold sweeps by target_kind, support weight strategies"
```

---

### Task 3: `cli.py` — allow weight strategies through `--wfo`

**Files:**
- Modify: `src/ggTrader/lab/cli.py`
- Test: `tests/lab/test_wfo.py`

**Interfaces:**
- Consumes: `run_wfo(..., universe_fn=...)` (Task 2), `eligible_at`
  (`ggTrader.lab.data`, already imported in `cli.py` for the plain
  `walkforward()` call).
- Produces: `--wfo` now accepts any strategy name in `STRATEGY_REGISTRY`, not
  just `SIGNAL_STRATEGY_NAMES`.

- [ ] **Step 1: Write the failing test**

Append to `tests/lab/test_wfo.py`:

```python
def test_cli_wfo_accepts_weight_strategy():
    """--wfo must accept a weight strategy name (xs_momentum), not just signals."""
    parser = build_arg_parser()
    args = parser.parse_args(["--strategy", "xs_momentum", "--wfo"])
    assert args.strategy == "xs_momentum"
    assert args.wfo is True
```

(This test only exercises argument parsing, which already accepts any
registered strategy name — the restriction lives in `cli.py`'s `main()`
function body, not the parser, so this test documents intent and is a
regression guard for Step 3 below; the real proof is Task 5's smoke run.)

- [ ] **Step 2: Run tests to verify current behavior**

Run: `pytest tests/lab/test_wfo.py -k cli_wfo_accepts_weight_strategy -v`
Expected: PASS already (parser accepts it) — this step confirms there is no
parser-level change needed; the fix in Step 3 is in `main()`'s runtime guard.

- [ ] **Step 3: Remove the signal-only `--wfo` restriction and pass `universe_fn`**

In `src/ggTrader/lab/cli.py`, find:

```python
    if args.wfo:
        from ggTrader.lab.wfo import run_wfo

        if args.strategy not in SIGNAL_STRATEGY_NAMES:
            raise SystemExit(f"--wfo only supports signal strategies: {SIGNAL_STRATEGY_NAMES}")
        print(f"WFO: {args.strategy} | {len(grid)} param combos", flush=True)
        result = run_wfo(
            args.strategy,
            strategy_cls,
            cfg,
            ohlcv,
            spy_close,
            eval_start=str(eval_start.date()),
            eval_end=str(eval_end.date()),
            market=args.market,
            base_config=dict(base_config),
            grid=grid,
        )
        return result.table
```

Replace with:

```python
    if args.wfo:
        from ggTrader.lab.wfo import run_wfo

        print(f"WFO: {args.strategy} | {len(grid)} param combos", flush=True)
        result = run_wfo(
            args.strategy,
            strategy_cls,
            cfg,
            ohlcv,
            spy_close,
            eval_start=str(eval_start.date()),
            eval_end=str(eval_end.date()),
            market=args.market,
            base_config=dict(base_config),
            grid=grid,
            universe_fn=lambda asof, past: eligible_at(asof, past, cfg, universe=univ)[0],
        )
        return result.table
```

`SIGNAL_STRATEGY_NAMES` may now be unused in `cli.py` if it was only
referenced at this one call site — check with `ruff check` in Task 4 and
remove the import/derivation if it's dead (it's still needed a few lines
above at `if args.strategy in SIGNAL_STRATEGY_NAMES: strat =
build_signal_strategy(...)`, so it almost certainly stays; verify rather than
assume).

- [ ] **Step 4: Run the full lab test suite**

Run: `pytest tests/lab/ -q`
Expected: All tests pass — no regressions in the signal-strategy `--wfo` path
or the `--sweep`/plain-walkforward paths.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/cli.py tests/lab/test_wfo.py
git commit -m "feat(lab): allow weight strategies through --wfo"
```

---

### Task 4: Lint cleanup

**Files:** `src/ggTrader/lab/wfo.py`, `src/ggTrader/lab/cli.py` — verification only.

- [ ] **Step 1: Run ruff**

Run: `ruff check src/ggTrader/lab/`
Expected: Report any unused imports left over from Task 2's deletion of the
duplicate OOS-rebuild block (`split_params`, `simulate_signals` in `wfo.py`)
or Task 3's restriction removal (`SIGNAL_STRATEGY_NAMES` in `cli.py`, if truly
unused). Remove any it flags.

- [ ] **Step 2: Re-run the full lab suite**

Run: `pytest tests/lab/ -q`
Expected: All tests still pass after the import cleanup.

- [ ] **Step 3: Commit (only if ruff required fixes)**

```bash
git add -u
git commit -m "chore(lab): remove dead imports from weight-based WFO dispatch"
```

(Skip this step entirely if Step 1 reported no issues.)

---

### Task 5: WFO smoke run — validate the new harness path against `xs_momentum`

**Files:** none — this is a research run, not a code change.

**Purpose:** Tasks 1–4 prove the dispatch is correctly wired with unit tests
on synthetic data. This task runs the *existing, already-researched*
`xs_momentum` strategy through the real `ggt lab --wfo` harness for the first
time, to confirm the new weight path produces sane output against real data
before Phase B builds a brand-new strategy on top of it.

- [ ] **Step 1: Run a smoke WFO pass on xs_momentum**

Run:
```bash
.venv/bin/python -m ggTrader.lab.cli --strategy xs_momentum --universe sp500 --wfo
```
Expected: Completes without raising an exception, prints per-fold WFO output
(train/test windows, gate pass/fail, chosen params) for at least one fold —
matching the shape of `overnight_gap`'s and every other `--wfo` strategy's
console output.

- [ ] **Step 2: Record the smoke-run result**

If the command raises, it's a Task 1–3 bug (fix before proceeding — a crash
means the dispatch is broken, not that xs_momentum lacks edge, since
xs_momentum's cross-sectional momentum premise on large-cap already has a
prior NO-GO verdict in `roadmap.md`'s gap-analysis list — a low/negative
Sharpe here is expected and not itself a new finding). If it completes, no
roadmap entry is needed — `xs_momentum`'s NO-GO status doesn't change; this
run only proves the harness, not a new research question.

- [ ] **Step 3: Commit**

No code changes in this task — nothing to commit unless Step 1 required a
fix, in which case that fix was already committed in Task 1–4's flow.

---

## Phase B: Cross-Sectional Idiosyncratic Volatility (`idio_vol`)

### File Structure

- **Create `src/ggTrader/lab/strategies/idio_vol.py`**: `idiosyncratic_variance`
  indicator function (rolling residual variance vs. the eligible universe's
  own equal-weighted return) and `IdioVolStrategy` (weights, long-only bottom
  quintile).
- **Modify `src/ggTrader/lab/strategies/__init__.py`**: import and register
  `IdioVolStrategy` in `STRATEGY_REGISTRY` and `__all__`.
- **Create `tests/lab/test_idio_vol.py`**: indicator-function tests,
  strategy-class tests, registry/CLI wiring tests.

---

### Task 6: `idiosyncratic_variance` indicator function

**Files:**
- Create: `src/ggTrader/lab/strategies/idio_vol.py`
- Test: `tests/lab/test_idio_vol.py` (new file)

**Interfaces:**
- Produces: `idiosyncratic_variance(returns: pd.DataFrame, market_returns:
  pd.Series, window: int) -> pd.DataFrame` — (time x symbol) rolling residual
  variance of each symbol's return against a single market-factor return
  series, using rolling covariance/variance (`beta_t = Cov(r_i, r_m)_t /
  Var(r_m)_t`, `resid_t = r_i,t - beta_t * r_m,t`, output = rolling variance
  of `resid`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/lab/test_idio_vol.py
"""Tests for cross-sectional idiosyncratic-volatility indicator and strategy."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.idio_vol import idiosyncratic_variance


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _returns(symbols, n=300, seed=42):
    rng = np.random.default_rng(seed)
    idx = _idx(n)
    market = rng.normal(0.0005, 0.01, n)
    data = {}
    for i, s in enumerate(symbols):
        # Each symbol = beta*market + idiosyncratic noise of varying scale.
        idio_scale = 0.005 * (i + 1)
        data[s] = 0.8 * market + rng.normal(0, idio_scale, n)
    return pd.DataFrame(data, index=idx), pd.Series(market, index=idx)


class TestIdiosyncraticVariance:
    def test_output_shape(self):
        returns, market = _returns(["A", "B", "C"], n=200)
        resid_var = idiosyncratic_variance(returns, market, window=20)
        assert resid_var.shape == returns.shape
        assert list(resid_var.columns) == ["A", "B", "C"]

    def test_warmup_is_nan(self):
        returns, market = _returns(["A"], n=100)
        resid_var = idiosyncratic_variance(returns, market, window=20)
        assert resid_var["A"].iloc[:19].isna().all()
        assert resid_var["A"].iloc[25:].notna().all()

    def test_higher_idio_noise_gives_higher_residual_variance(self):
        """Symbol C has 3x the idiosyncratic noise scale of A by construction."""
        returns, market = _returns(["A", "B", "C"], n=300, seed=7)
        resid_var = idiosyncratic_variance(returns, market, window=60)
        last = resid_var.iloc[-1]
        assert last["C"] > last["A"]

    def test_zero_market_variance_does_not_raise(self):
        idx = _idx(50)
        returns = pd.DataFrame({"A": np.full(50, 0.001)}, index=idx)
        market = pd.Series(np.zeros(50), index=idx)  # constant market -> Var=0
        resid_var = idiosyncratic_variance(returns, market, window=10)
        assert resid_var.shape == (50, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lab/test_idio_vol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named
'ggTrader.lab.strategies.idio_vol'`.

- [ ] **Step 3: Implement the indicator function**

```python
# src/ggTrader/lab/strategies/idio_vol.py
"""Cross-sectional idiosyncratic-volatility strategy (weight-based)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import eligible_symbols, extract_close
from ggTrader.lab.strategy import LabConfig, Plan


def idiosyncratic_variance(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int,
) -> pd.DataFrame:
    """Rolling residual variance of each symbol vs. a single market-factor return.

    beta_t = Cov(r_i, r_m)_t / Var(r_m)_t (rolling, causal as of bar t);
    resid_t = r_i,t - beta_t * r_m,t; output = rolling variance of resid.
    A near-zero rolling market variance (e.g. a flat/constant market factor)
    produces beta = inf/NaN by division, which is expected and handled by the
    caller (NaN residual variance sorts last / is dropped, never selected).
    """
    market_returns = market_returns.reindex(returns.index)
    market_var = market_returns.rolling(window=window, min_periods=window).var()

    resid_var = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for col in returns.columns:
        cov = returns[col].rolling(window=window, min_periods=window).cov(market_returns)
        beta = cov / market_var
        resid = returns[col] - beta * market_returns
        resid_var[col] = resid.rolling(window=window, min_periods=window).var()
    return resid_var
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/lab/test_idio_vol.py -v`
Expected: PASS on all 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/idio_vol.py tests/lab/test_idio_vol.py
git commit -m "feat(lab): add idiosyncratic_variance indicator function"
```

---

### Task 7: `IdioVolStrategy` (select/to_targets, long-only bottom quintile)

**Files:**
- Modify: `src/ggTrader/lab/strategies/idio_vol.py`
- Test: `tests/lab/test_idio_vol.py` (append)

**Interfaces:**
- Consumes: `idiosyncratic_variance` (Task 6), `extract_close`,
  `eligible_symbols` (`ggTrader.lab.strategies.indicators`), `LabConfig`,
  `Plan` (`ggTrader.lab.strategy`).
- Produces: `IdioVolStrategy` class, `name = "idio_vol"`, `target_kind =
  "weights"`, constructor `(cfg, reg_window=20, quintile=5)`, `select()`,
  `to_targets()`, `sweep_params()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/lab/test_idio_vol.py`:

```python
def _ohlcv_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    frames = {}
    for col in returns.columns:
        close = 100.0 * (1.0 + returns[col]).cumprod()
        frames[col] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": np.full(len(close), 1e6),
            },
            index=returns.index,
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


from ggTrader.lab.strategies.idio_vol import IdioVolStrategy


class TestIdioVolStrategy:
    def test_select_returns_bottom_quintile_only(self):
        returns, _market = _returns(["A", "B", "C", "D", "E"], n=300, seed=3)
        ohlcv = _ohlcv_from_returns(returns)
        strat = IdioVolStrategy(LabConfig(min_history_bars=100), reg_window=20, quintile=5)
        sels = strat.select(ohlcv.index[-1], ohlcv, ["A", "B", "C", "D", "E"])
        # 5 symbols / quintile=5 -> bucket size 1: only the single lowest-idio-var symbol.
        assert len(sels) == 1
        assert all("weight" in s for s in sels)
        assert abs(sum(s["weight"] for s in sels) - 1.0) < 1e-9

    def test_select_respects_min_history(self):
        returns, _market = _returns(["A", "B"], n=50, seed=1)
        ohlcv = _ohlcv_from_returns(returns)
        strat = IdioVolStrategy(LabConfig(min_history_bars=400))
        sels = strat.select(ohlcv.index[-1], ohlcv, ["A", "B"])
        assert sels == []

    def test_select_prefers_low_idio_variance_symbol(self):
        """Symbol A has the lowest idiosyncratic noise scale by construction (i=0)."""
        returns, _market = _returns(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"], n=400, seed=11)
        ohlcv = _ohlcv_from_returns(returns)
        strat = IdioVolStrategy(LabConfig(min_history_bars=100), reg_window=30, quintile=5)
        sels = strat.select(ohlcv.index[-1], ohlcv, list(returns.columns))
        assert "A" in [s["symbol"] for s in sels]

    def test_to_targets_returns_weight_dataframe(self):
        returns, _market = _returns(["A", "B", "C"], n=300, seed=5)
        ohlcv = _ohlcv_from_returns(returns)
        strat = IdioVolStrategy(LabConfig(min_history_bars=100))
        plans = {
            ohlcv.index[250]: [{"symbol": "A", "weight": 1.0}],
        }
        targets = strat.to_targets(plans, ohlcv)
        assert isinstance(targets, pd.DataFrame)
        assert set(targets.columns) == {"A"}
        assert (targets.dropna() == 1.0).all().all() or targets.dropna().empty is False

    def test_sweep_params_has_reg_window_and_quintile(self):
        params = IdioVolStrategy.sweep_params()
        assert "reg_window" in params
        assert "quintile" in params
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lab/test_idio_vol.py -v`
Expected: FAIL with `ImportError: cannot import name 'IdioVolStrategy'`.

- [ ] **Step 3: Implement the strategy class**

Append to `src/ggTrader/lab/strategies/idio_vol.py`:

```python
class IdioVolStrategy:
    """Long-only defensive-premium sleeve: equal-weight the lowest-idiosyncratic-
    variance quintile of the eligible universe, rebalanced monthly.

    Market factor is the eligible universe's own equal-weighted return (the
    same convention ggTrader.lab.simulate.compute_vol_scalar already uses for
    vol targeting) rather than an external benchmark series, since the
    Strategy protocol's select()/to_targets() only receive the strategy's own
    OHLCV frame.
    """

    name = "idio_vol"
    target_kind = "weights"

    def __init__(self, cfg: LabConfig, reg_window: int = 20, quintile: int = 5) -> None:
        self.cfg = cfg
        self.reg_window = reg_window
        self.quintile = quintile

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "reg_window": [20, 40, 60],
            "quintile": [4, 5],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        elig = eligible_symbols(data, eligible, self.cfg.min_history_bars)
        if len(elig) < self.quintile:
            return []

        close = extract_close(data, elig)
        returns = close.pct_change()
        market_returns = returns.mean(axis=1)
        resid_var = idiosyncratic_variance(returns, market_returns, self.reg_window)

        latest = resid_var.iloc[-1].dropna()
        if len(latest) < self.quintile:
            return []

        ranked = latest.sort_values()  # ascending: lowest residual variance first
        bucket_size = max(1, len(ranked) // self.quintile)
        bottom = ranked.index[:bucket_size].tolist()
        if not bottom:
            return []

        weight = 1.0 / len(bottom)
        return [{"symbol": s, "weight": weight} for s in bottom]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0  # default: exit anything not re-selected
            for sel in plans[asof]:
                targets.loc[bar, sel["symbol"]] = float(sel["weight"])
        return targets
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/lab/test_idio_vol.py -v`
Expected: PASS on all 9 tests so far.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/idio_vol.py tests/lab/test_idio_vol.py
git commit -m "feat(lab): add IdioVolStrategy select/to_targets"
```

---

### Task 8: Registry + CLI wiring

**Files:**
- Modify: `src/ggTrader/lab/strategies/__init__.py`
- Test: `tests/lab/test_idio_vol.py` (append)

**Interfaces:**
- Consumes: `IdioVolStrategy` (Task 7).
- Produces: `STRATEGY_REGISTRY["idio_vol"] = IdioVolStrategy`; `--strategy
  idio_vol` accepted by the CLI (weight strategies' choices are also derived
  from `STRATEGY_REGISTRY`, same as signal strategies — no `cli.py` edit
  needed beyond Phase A's Task 3, already merged).

- [ ] **Step 1: Write the failing tests**

Append to `tests/lab/test_idio_vol.py`:

```python
def test_idio_vol_registered():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY

    assert "idio_vol" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["idio_vol"] is IdioVolStrategy


def test_cli_accepts_idio_vol():
    from ggTrader.lab.cli import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--strategy", "idio_vol"])
    assert args.strategy == "idio_vol"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lab/test_idio_vol.py -k registered -v`
Expected: FAIL — `"idio_vol" in STRATEGY_REGISTRY` is `False`.

- [ ] **Step 3: Register the strategy**

In `src/ggTrader/lab/strategies/__init__.py`, add the import next to
`CrossSectionalMomentum`/`DualMomentum`:

```python
from .idio_vol import IdioVolStrategy
```

Add to `STRATEGY_REGISTRY`:

```python
STRATEGY_REGISTRY: dict[str, Any] = {
    ...
    "xs_momentum": CrossSectionalMomentum,
    "dual_momentum": DualMomentum,
    "idio_vol": IdioVolStrategy,
}
```

Add `"IdioVolStrategy"` to `__all__` (next to `"CrossSectionalMomentum"`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/lab/test_idio_vol.py -v`
Expected: PASS on all 11 tests.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/__init__.py tests/lab/test_idio_vol.py
git commit -m "feat(lab): register idio_vol strategy in STRATEGY_REGISTRY"
```

---

### Task 9: Full-suite verification and lint

**Files:** none new — verification only.

- [ ] **Step 1: Run the full lab test suite**

Run: `pytest tests/lab/ -q`
Expected: All tests pass, zero failures (pre-existing count + Phase A's ~5
new tests + Phase B's 11 new tests).

- [ ] **Step 2: Run ruff**

Run: `ruff check src/ggTrader/lab/`
Expected: No new lint errors. Fix any reported issues (import ordering,
unused imports) before proceeding.

- [ ] **Step 3: Commit (only if ruff required fixes)**

```bash
git add -u
git commit -m "chore(lab): fix lint issues from idio_vol addition"
```

(Skip this step entirely if Step 2 reported no issues.)

---

### Task 10: WFO smoke run + full research pass on `idio_vol`

**Files:** none — this is a research run, not a code change.

**Purpose:** Confirm `idio_vol` is genuinely wired into the now-generalized
gated WFO harness end-to-end, then get its first honest GO/NO-GO verdict —
combining the "smoke run first" caution from the `overnight_gap` plan with
the fact that Phase A already validated the harness path itself (Task 5), so
a single full-scale run is sufficient here rather than a separate small-subset
smoke step.

- [ ] **Step 1: Run the full WFO pass**

Run:
```bash
.venv/bin/python -m ggTrader.lab.cli --strategy idio_vol --universe sp500 --wfo
```
Expected: Completes without raising an exception and prints per-fold WFO
output (fold train/test date ranges, gate pass/fail, chosen params) for at
least one fold, an OOS Aggregate line, and a SPY baseline comparison line —
matching the console output shape of the `overnight_gap` full run.

- [ ] **Step 2: Record the result in `docs/roadmap.md`**

Add or update the Cross-Sectional Idiosyncratic Volatility entry in the
research-directions list (`docs/roadmap.md`, same section as the
`overnight_gap` and Kelly-sizing entries) with the actual OOS Sharpe, CAGR,
MaxDD, and fold-stability numbers from Step 1's console output, following the
established format: `✅ Deployed` / `❌ Rejected` / `🧪 Researching` prefix,
the honest WFO numbers, and — if rejected — which specific failure mode
(fold-instability, chronic gate failure, negative OOS Sharpe) closed it, so
the entry is falsifiable evidence rather than a bare verdict.

- [ ] **Step 3: Commit the roadmap note**

```bash
git add docs/roadmap.md
git commit -m "docs(roadmap): record idio_vol full SP500 WFO verdict"
```

---

## Verification (this plan)

- `pytest tests/lab/test_wfo.py -v` — all new Phase A tests pass, no
  regressions in existing WFO tests (Tasks 1–3).
- `pytest tests/lab/test_idio_vol.py -v` — all new Phase B tests pass (Tasks
  6–8).
- `pytest tests/lab/ -q` — full lab suite passes, no regressions (Tasks 4, 9).
- `ruff check src/ggTrader/lab/` — clean (Tasks 4, 9).
- `ggt lab --wfo --strategy xs_momentum --universe sp500` completes without
  exceptions (Task 5) — proves the harness generalization against a
  known-quantity strategy before Phase B builds on it.
- `ggt lab --wfo --strategy idio_vol --universe sp500` completes and produces
  a documented GO/NO-GO verdict in `docs/roadmap.md` (Task 10).
