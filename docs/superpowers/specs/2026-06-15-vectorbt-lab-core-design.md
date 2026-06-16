# vectorbt lab core — design spec

**Date:** 2026-06-15
**Status:** Plan 1 executed 2026-06-15 (momentum bench shipped on branch `lab-core-plan1`; validation gate passes, selections bit-identical). Plans 2 (wfo_tournament/signal family) and 3 (equity backfill + old-code deletion) pending.
**Supersedes (on cutover):** the `research/` monthly walk-forward harness, the old WFO/orchestrator/backtest stack

## 1. Purpose

Start fresh on a **simple, vectorbt-centric research/backtest core** that is also the **fast research bench**: testing a new strategy idea and getting an honest out-of-sample (OOS — performance on data the strategy never saw during selection) verdict should take minutes and minimal code. The new core is a thin wrapper around vectorbt, stores all data in TimescaleDB, and serves crypto and equities through one unified abstraction.

This replaces ~26k lines of accumulated research/backtest machinery (orchestrator, dual backtest paths, file-based result managers) with a small, legible package. It builds directly on the proven monthly walk-forward harness shipped 2026-06-10/12.

**Out of scope (deferred to a future spec):** live execution (to be rebuilt wrapping Kraken CLI + Alpaca CLI), the existing CCXT broker bridges, the live crypto engine, and the daily-PnL path. Data *ingestion* is also out of scope — we build on the OHLCV already in the DB.

## 2. Decisions (from brainstorm)

| Decision | Choice |
|---|---|
| Primary goal | Simple core that is *also* the fast research bench (both, equally) |
| Scope now | Research/backtest core; live is a separate later spec |
| Markets | Crypto + equities, one unified core |
| Strategy model | Generalized `select` (data ≤ T) + vectorized simulate |
| Persistence | TimescaleDB is the **only** store; no `results/` files for research |
| Migration | New `src/ggTrader/lab/` package, coexist with old code, cut over after a validation gate, then delete old research/backtest code |
| Data sources | OHLCV stays in the DB as-is. Equity deep history via yfinance (best free source), recent equity bars via Alpaca CLI; crypto Kraken data unchanged. Not part of this rewrite. |

## 3. Architecture

New package `src/ggTrader/lab/` — small modules, each one responsibility:

| Module | Responsibility |
|---|---|
| `strategy.py` | `Strategy` protocol, `Plan` type, `build_strategy` factory + registry |
| `strategies/` | One file per strategy: `wfo_tournament.py`, `xs_momentum.py`, `dual_momentum.py`, future ideas |
| `harness.py` | Walk-forward loop: plan phase → vectorized simulate → persist → score |
| `simulate.py` | Matrix stacking + the single grouped `vbt.Portfolio` call |
| `data.py` | DB OHLCV reader → `(symbol, field)` frame; universe providers (PIT S&P 500 for equity, listed coins for crypto) |
| `metrics.py` | returns → Sharpe/Sortino/CAGR/maxDD/hit-rate via vbt; benchmark compare |
| `persist.py` | All DB reads/writes; resume logic |
| `cli.py` | `ggt lab run --strategy … --market … --freq …` |

### 3.1 Strategy protocol

```python
class Strategy(Protocol):
    name: str
    def select(self, asof, data, eligible) -> Plan: ...
        # Pure function of data <= asof. Returns a JSON-able Plan
        # (list of {symbol, weight, ...params}). MUST self-truncate to asof.
    def to_targets(self, plans: dict[Timestamp, Plan], data) -> pd.DataFrame: ...
        # Whole-window (time x symbols) target matrix — weights for hold
        # strategies, or entry/exit signals for entry/exit strategies.
        # No simulation happens here.
```

`Plan` is the frozen, JSON-able output of one rebalance (selections with weights and/or per-asset params). `select` is the only point-in-time-sequential step; it is either cheap (momentum ranking) or already internally vectorized (the WFO tournament's grid is one vbt call over all param-combos × symbols).

### 3.2 Harness — two phases

**Phase 1 — plan (point-in-time).** For each strategy and rebalance date T, run `select` on `data[:T]`, producing per-strategy target matrices.

**Phase 2 — vectorized simulate (all strategies at once).** Stack every strategy's target matrix into columns indexed by a `(strategy, symbol)` MultiIndex and run **one** `vbt.Portfolio.from_orders` (weights) or `from_signals` (entry/exit) call with `group_by="strategy"` + `cash_sharing=True`. vbt simulates all strategies, all symbols, the entire multi-rebalance path simultaneously; metrics come back as a Series indexed by strategy.

```python
def walkforward(strategies, market, freq, eval_window, benchmark) -> run_id:
    data = load_ohlcv(market, eval_window)                      # from DB
    dates = rebalance_dates(freq, eval_window)
    targets = {}
    for s in strategies:
        plans = {}
        for T in dates:
            if persist.done(run_id, s.name, T):                 # resume from DB
                plans[T] = persist.read_plan(run_id, s.name, T)
                continue
            plan = s.select(T, data.loc[:T], universe(market, T))
            persist.write_plan(run_id, s.name, T, plan)
            plans[T] = plan
        targets[s.name] = s.to_targets(plans, data)
    panel = concat(targets, axis=1, keys=[s.name for s in strategies])
    pf = vbt.Portfolio.from_orders(prices_panel(data, panel), panel,
                                   group_by="strategy", cash_sharing=True,
                                   fees=..., slippage=...)        # ONE call
    persist.write_returns_equity(run_id, pf)                     # per strategy
    persist.write_summary(run_id, metrics(pf, benchmark))
    return run_id
```

This is faster *and* more correct than per-period stitching: vbt handles cash and compounding natively across the whole window, where the old harness multiplied stitched period returns (an approximation).

Frequency-agnostic (`freq` = monthly for equity; monthly-recalibration or N-bar for crypto) and market-agnostic (only `universe()` and the chosen strategy differ).

### 3.3 Reuse

Ported into `lab/` or imported: vbt metric extraction (clean bits of `core/metrics.py`), `indicators/indicator_precompute.py`, `data/core/index_constituents.py` (PIT membership), the OHLCV read in `data/historical/timescaledb_loader.py`, the entry/exit classes in `indicators/strategies.py` (used inside `wfo_tournament.select`), the WFO grid cache (`data/cache/wfo_cache.py`), and `utils/notifier.py`.

## 4. Persistence — TimescaleDB only

New `lab_*` tables; `lab_returns` and `lab_equity` are hypertables (daily rows are exactly TimescaleDB's purpose).

| Table | Columns |
|---|---|
| `lab_runs` | run_id PK, strategy_set JSONB, market, freq, eval_start, eval_end, params JSONB, status, created_at |
| `lab_plans` | run_id, strategy, asof, plan JSONB, eligible_count, coverage JSONB — PK (run_id, strategy, asof) |
| `lab_returns` | run_id, strategy, date, ret — hypertable |
| `lab_equity` | run_id, strategy, date, strategy_equity, benchmark_equity — hypertable |
| `lab_summary` | run_id, strategy, metrics JSONB, benchmark_metrics JSONB, diagnostics JSONB — PK (run_id, strategy) |

- **Resume** reads `lab_plans` to skip completed (run_id, strategy, asof) rebalances.
- **Reports/plots** are a separate renderer that reads these tables on demand — never files.
- Per-rebalance plan writes are transactional; an interrupted run resumes from the last committed rebalance.

## 5. Leak safety

First-class, as in the current harness (it already caught a real positional-indexing lookahead bug):
- harness passes `data.loc[:T]` to `select`;
- every strategy self-truncates (`data.loc[:asof]`) as defense in depth;
- `leak_check` compares plans built from full / truncated-copy / unmasked data and requires all three identical.

## 6. Error handling

| Condition | Behavior |
|---|---|
| Empty plan / all-cash period | Flat — zero-weight columns, handled natively by `from_orders` |
| Selected symbol missing price at rebalance bar | Dropped; recorded in `coverage` JSONB |
| `select` raises | Fail fast with a clear error; plans already persisted, so resume continues |
| DB write failure mid-run | Transactional per rebalance; resume re-runs the incomplete one |

## 7. Testing

- **Per-strategy units** (synthetic, no network): `select` plans correct; `to_targets` matrices correct. Port the existing momentum unit tests.
- **Leak checks per strategy:** full / truncated / unmasked plans identical.
- **Vectorization-equivalence test:** N strategies in the grouped `from_orders` call yield the same per-strategy equity as each run alone — guards the "all strats simultaneously" claim.
- **Harness integration** (synthetic): full plan → vectorized-sim → persist; assert DB rows land and resume skips completed rebalances.
- **Metrics test:** known returns → known Sharpe/Sortino.

## 8. Validation gate (acceptance test for the rewrite)

The new core must reproduce the known-good equity runs before any old code is deleted:
- **Hard gate:** per-rebalance *selections* match the old `selections.json` **exactly** for `sp500_xs_momentum`, `sp500_dual_momentum`, `sp500_monthly_v1` (select logic is ported unchanged → must be bit-identical).
- **Soft gate:** equity curves / Sharpe within a documented tolerance; the small difference vs the old numbers is attributed to single-pass `from_orders` compounding (the new value is the more-correct one) and written up.
- **Crypto smoke run:** one strategy, small universe, proving the unified core serves both markets.

## 9. Cutover and deletion

Build `lab/` alongside the old code. **Cutover = validation gate passes.** Then delete (this spec):
- `research/` (old harness), `backtest/vectorized.py`, `backtesting/wfo.py`,
- `core/orchestrator.py`, `core/wfo.py`, `core/sensitivity.py`, `core/fast_backtest.py`,
- `pipeline/`, old `cli/cmd_research|cmd_backtest|cmd_production`,
- file-based `utils/results_manager.py` and `results/` usage for research.

**Explicitly deferred to the future live spec** (deleting these now would break the cron PnL reports before their replacement exists): `core/crypto_execution_engine.py`, `core/base_execution_engine.py`, the CCXT broker bridges in `execution/`, `core/trade_tracker.py`, and the daily-PnL path.

## 10. Open questions / risks

- **WFO tournament inside `to_targets`:** the entry/exit strategy emits per-bar signals over each frozen forward period; confirm these compose cleanly into a single whole-window `from_signals` panel alongside the weight-based strategies (may require running weight strategies and signal strategies as two grouped calls, then merging metrics). Resolve in the implementation plan.
- **Package name:** `lab/` is a placeholder; confirm before scaffolding.
- **Tolerance band** for the soft validation gate to be fixed in the plan (proposed: equity endpoint within 1% relative, Sharpe within 0.05 absolute).
