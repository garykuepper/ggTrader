# ggtrader — Algorithmic Strategy Implementation
## Coding Agent Specification & Software Prompt

**Strategies:** Cross-Sectional Momentum + Liquidity Tournament · HMM Regime Filter Overlay  
**Venues:** Alpaca (US Equities) · Binance US (Crypto)  
**Engine:** VectorBT · **Framework:** garykuepper/ggtrader  
**Date:** June 2026

---

## Table of Contents

1. [Coding Agent System Prompt](#1-coding-agent-system-prompt)
2. [Implementation Overview](#2-implementation-overview)
3. [Strategy 1 — Cross-Sectional Momentum & Liquidity Tournament](#3-strategy-1--cross-sectional-momentum--liquidity-tournament)
   - [Task 1A — Numba Factor Functions](#task-1a--numba-factor-functions)
   - [Task 1B — Rank Synthesis & Entry Signals](#task-1b--rank-synthesis--entry-signals)
   - [Task 1C — Asset-Class Config & Tuning](#task-1c--asset-class-config--tuning)
   - [Task 1D — VectorBT Portfolio Integration](#task-1d--vectorbt-portfolio-integration)
4. [Strategy 2 — HMM Regime Filter Overlay](#4-strategy-2--hmm-regime-filter-overlay)
   - [Task 2A — Offline HMM Training Script](#task-2a--offline-hmm-training-script)
   - [Task 2B — Runtime Regime Loader & Gate](#task-2b--runtime-regime-loader--gate)
   - [Task 2C — Gate Integration into Portfolio Simulation](#task-2c--gate-integration-into-portfolio-simulation)
5. [Walk-Forward Optimization Harness](#5-walk-forward-optimization-harness)
6. [Testing Requirements](#6-testing-requirements)
7. [Coding Agent Task Prompts](#7-coding-agent-task-prompts)
8. [Quick Reference — DO / DO NOT](#8-quick-reference--do--do-not)

---

## 1. Coding Agent System Prompt

> **How to use:** Paste the block below verbatim as the system prompt to your coding agent (Claude Code, Cursor, GPT-4o, etc.) before issuing any implementation tasks. It encodes the full architectural context, constraints, and conventions so the agent does not need to re-derive them from scratch.

```
You are a senior quantitative engineer working on garykuepper/ggtrader, a
modular algorithmic trading framework. Your role is to implement, extend,
and refactor production-grade Python code that conforms to the existing
codebase conventions and the architectural decisions documented below.

REPOSITORY CONTEXT
  - Language: Python 3.11+
  - Backtesting engine: VectorBT (vectorized 2D NumPy/Pandas arrays)
  - Brokers: Alpaca (US Equities), Binance US (Crypto)
  - Data store: TimescaleDB (PostgreSQL extension)
  - Execution model: async event loop (asyncio)
  - Package manager: Poetry; all deps declared in pyproject.toml
  - Testing: pytest + hypothesis for property-based edge cases
  - Code style: black formatter, ruff linter, type hints required (mypy strict)

STRATEGY SELECTION (from research phase — treat as FIXED decisions)
  Primary: Strategy 1 — Cross-Sectional Multi-Factor Momentum &
           Liquidity Tournament
  Overlay:  Strategy 2 — Unsupervised HMM State Regime Filter
  Deferred: Strategy 3 (Kalman pairs) — equities only, Phase B

HARD CONSTRAINTS (never violate)
  1. All signal arrays must be 2D: shape (n_timestamps, n_assets)
  2. No uncompiled Python loops in hot path — use Numba @njit
  3. Max position size = 10% of account equity (notional basis)
  4. Cash-shared portfolio simulation only (no per-asset sub-accounts)
  5. No look-ahead bias — all rolling windows use .shift(1) offsets
  6. HMM must be trained offline; load pre-computed regime series
  7. Equity signals rank within GICS sector (not cross-sector raw)
  8. Crypto signals strip BTC beta before computing momentum residual

FILE & MODULE LAYOUT (follow exactly)
  src/ggtrader/
  ├── strategies/
  │   ├── __init__.py
  │   ├── base.py              # StrategyBase ABC
  │   ├── momentum/
  │   │   ├── __init__.py
  │   │   ├── cross_sectional.py  # Strategy 1 core
  │   │   ├── factors.py           # Numba-compiled factor fns
  │   │   └── signals.py           # Rank synthesis + entry logic
  │   └── regime/
  │       ├── __init__.py
  │       ├── hmm_filter.py        # Strategy 2 HMM overlay
  │       ├── train_hmm.py         # Offline training script
  │       └── emission_vectors.py  # Equity vs. crypto features
  ├── execution/
  │   ├── alpaca.py               # Existing (extend, do not break)
  │   └── binance_us.py            # Existing (extend, do not break)
  ├── portfolio/
  │   ├── sizing.py               # Position sizing + 10% cap
  │   └── risk.py                 # Drawdown / exposure guards
  └── data/
      ├── timescale.py            # DB I/O
      └── universe.py             # Asset universe loader

CODING CONVENTIONS
  - Docstrings: Google style, required on all public functions/classes
  - Type annotations: full (no 'Any' without explicit justification)
  - Numba functions: suffix _nb (e.g., compute_momentum_nb)
  - Config: pydantic BaseSettings; no hardcoded magic numbers
  - Logging: structlog; never print()
  - Error handling: raise domain-specific exceptions from exceptions.py
  - Tests: one test file per module, prefix test_

ASSET-CLASS TUNING RULES (implement as config flags)
  Equities:  rank_mode='sector'  vol_mode='overnight_intraday_split'
             liquidity_smooth='simple_rolling'
  Crypto:    rank_mode='residual_btc'  vol_mode='continuous_ewm'
             liquidity_smooth='ewm'  btc_beta_window=60

Do not implement features not listed in the task. Ask before adding
dependencies. Prefer stdlib and existing repo deps. When in doubt
about architectural intent, ask rather than assume.
```

---

## 2. Implementation Overview

| Phase | Deliverable | Strategy | Priority |
|-------|-------------|----------|----------|
| 1A | Numba factor functions (momentum + liquidity) | Strategy 1 | P0 |
| 1B | Cross-sectional rank synthesis + entry signals | Strategy 1 | P0 |
| 1C | Asset-class tuning config + BTC-beta stripping | Strategy 1 | P0 |
| 1D | VectorBT portfolio simulation integration | Strategy 1 | P0 |
| 2A | HMM offline training script (equities + crypto) | Strategy 2 | P1 |
| 2B | Regime filter loader + boolean gate overlay | Strategy 2 | P1 |
| 2C | Gate integration into portfolio simulation | Strategy 2 | P1 |
| 3 | Walk-forward optimization (WFO) harness | Both | P2 |
| 4 | Unit + integration tests | Both | P2 |

> **Sequencing rule:** Complete all Phase 1 tasks and validate backtests before starting Phase 2. The HMM overlay gate must multiply against a working signal matrix — building it against a stub will cause integration failures.

---

## 3. Strategy 1 — Cross-Sectional Momentum & Liquidity Tournament

This is the primary alpha-generating strategy. It ranks assets on volatility-adjusted idiosyncratic momentum, hybridized with a transient liquidity shock factor. The top decile of the composite score generates long entry signals.

---

### Task 1A — Numba Factor Functions

**File:** `src/ggtrader/strategies/momentum/factors.py`

Implement the following Numba-compiled functions. All functions must be decorated `@njit` and accept strictly 2D NumPy float64 arrays.

#### Function 1: `compute_momentum_nb`

Computes the volatility-adjusted idiosyncratic momentum factor for each asset at each timestamp.

- **Inputs:** `close_arr` (n_t, n_assets), `window: int`, `gap: int`
- For each `(i, col)`: slice `close_arr[i-window : i-gap, col]` as `hist_slice`
- Compute log return: `log(hist_slice[-1]) - log(hist_slice[0])`
- Compute realized vol: `std` of log returns over `hist_slice`
- Output: `mom_out[i, col] = log_return / (vol + 1e-8)`
- Fill with `np.nan` for rows where `i < window`
- **CRITICAL:** Use log returns internally, not simple price ratios — avoids compounding distortion

#### Function 2: `compute_liquidity_shock_nb`

Computes the Amihud-proxy liquidity shock factor (absolute price impact per dollar volume).

- **Inputs:** `close_arr` (n_t, n_assets), `vol_arr` (n_t, n_assets) — dollar volume
- `liq_out[i, col] = |log(close[i]/close[i-1])| / (vol_arr[i, col] + 1e-8)`
- For crypto (`ewm_smooth=True`): caller pre-smooths `vol_arr` with EWM before passing in
- Output dtype must be float64; shape identical to inputs

#### Function 3: `strip_btc_beta_nb`

Strips systematic BTC beta from altcoin log returns via rolling OLS. Equity universe does **not** call this function.

- **Inputs:** `alt_log_ret` (n_t, n_alts), `btc_log_ret` (n_t,) 1D, `window: int`
- For each col, rolling window OLS: `alt ~ btc`; extract residual
- Store `beta[i, col]` and `residual[i, col]`; return residual array
- Use `np.linalg.lstsq`-equivalent inside `@njit` (manual: `beta = cov(x,y)/var(x)`)
- Minimum window: 30 bars — fill with `np.nan` below threshold

> ⚠️ **Common bug:** The original research document showed a subtle indexing error: `hist_slice[-1] / hist_slice` (scalar/array) rather than `log(hist_slice[-1]) - log(hist_slice[0])`. Implement the corrected log-return formulation above.

---

### Task 1B — Rank Synthesis & Entry Signals

**File:** `src/ggtrader/strategies/momentum/signals.py`

Implements the composite factor score and translates it into a boolean entry signal matrix. All operations must be standard Pandas/NumPy — no Python loops.

- Load pre-computed `mom_df` and `liq_df` DataFrames (output of Task 1A wrapped in `IndicatorFactory`)
- Cross-sectional rank both factors independently: `df.rank(axis=1, pct=True)`
- Composite score: `composite = w_mom * mom_rank + w_liq * liq_rank`
  - Default weights: `w_mom=0.6`, `w_liq=0.4` — expose as Pydantic config field
- Entry signal: `entries = composite >= composite.quantile(0.90, axis=1)`
  - This selects top decile per timestamp row — do **NOT** use a fixed threshold
- Shift signals: `entries = entries.shift(1)` — prevents look-ahead on close price
- Return boolean DataFrame (n_t, n_assets); dtype `bool`

> **Sector ranking (equities):** When `asset_class='equity'`, rank within GICS sector groups using `groupby` before computing the composite. Pass `sector_map: Dict[str, str]` as a constructor argument. This prevents a dominant sector (e.g., tech) from sweeping the top decile.

> **Residual ranking (crypto):** When `asset_class='crypto'`, `mom_df` must already be the BTC-residual output from `strip_btc_beta_nb`. The rank synthesis function itself is asset-class agnostic — the pre-processing step is what changes.

---

### Task 1C — Asset-Class Config & Tuning

**File:** `src/ggtrader/strategies/momentum/cross_sectional.py` + config schema

All tunable parameters must live in a Pydantic `BaseSettings` subclass — no hardcoded numbers anywhere in signal code.

| Parameter | Equity Default | Crypto Default | Type | Description |
|-----------|---------------|----------------|------|-------------|
| `formation_window` | `40` | `30` | `int` | Lookback bars for momentum |
| `exclusion_gap` | `5` | `3` | `int` | Short-term reversal skip |
| `liq_smooth_mode` | `'rolling'` | `'ewm'` | `Literal` | Volume smoothing method |
| `liq_smooth_window` | `20` | `14` | `int` | Smoothing window |
| `rank_mode` | `'sector'` | `'residual_btc'` | `Literal` | Ranking universe scope |
| `btc_beta_window` | N/A | `60` | `int` | BTC beta estimation window |
| `w_momentum` | `0.6` | `0.6` | `float` | Composite weight (momentum) |
| `w_liquidity` | `0.4` | `0.4` | `float` | Composite weight (liquidity) |
| `entry_percentile` | `0.90` | `0.90` | `float` | Top-decile threshold |
| `vol_mode` | `'split'` | `'continuous'` | `Literal` | Overnight vs. full-session vol |

> ✅ **Validation rule:** Assert `w_momentum + w_liquidity == 1.0` in the Pydantic model validator. Raise `ValueError` if `entry_percentile` is not in `(0.5, 1.0)`.

---

### Task 1D — VectorBT Portfolio Integration

**File:** `src/ggtrader/strategies/momentum/cross_sectional.py`

Wires the signal DataFrame into VectorBT's portfolio simulation with cash sharing and position sizing.

- Use `vbt.Portfolio.from_signals(entries=entries_df, exits=exits_df, ...)`
- Cash-sharing: `group_by=True` (shared portfolio cash pool across all columns)
  - Do **NOT** use `group_by=False` — this would simulate independent sub-accounts
- Position sizing: `size=0.10`, `size_type='valuepercent'` — 10% of equity per signal
  - If top decile yields >10 concurrent signals, VectorBT cash-sharing auto-caps
- Direction: long-only (`direction='longonly'`)
- Fees: pass `fee=config.fee_rate` — equities `0.0005`, crypto `0.001` (configurable)
- Slippage: `slippage=config.slippage` — equities `0.0005`, crypto `0.002`
- Exit logic: `hold_period` exit OR momentum rank exits top decile — use `vbt.Portfolio` exits param
  - Default `hold_bars=5` for equities, `hold_bars=3` for crypto
- Expose `.stats()` and `.plot()` passthroughs on the strategy class for rapid inspection

> ⚠️ **Memory warning:** Running WFO across large parameter grids on a universe of 50+ assets can exhaust RAM. Implement chunked parameter sweeps: sweep no more than 500 combinations per batch, save intermediate results to TimescaleDB, and garbage collect between batches.

---

## 4. Strategy 2 — HMM Regime Filter Overlay

The Hidden Markov Model regime filter is a **risk-management overlay** — it does NOT generate alpha on its own. It produces a boolean execution gate that suppresses Strategy 1 entry signals during systemic lapse states. It must be trained offline to prevent look-ahead bias.

---

### Task 2A — Offline HMM Training Script

**File:** `src/ggtrader/strategies/regime/train_hmm.py`

Standalone script (not imported at runtime) that trains the HMM and persists the decoded regime series. Must be re-run when extending the historical window.

#### Dependencies

- `hmmlearn >= 0.3.0` (`GaussianHMM` class)
- `scikit-learn` (`StandardScaler` for emission feature normalization)
- `pandas`, `numpy`, `sqlalchemy` (TimescaleDB I/O)

#### Emission feature vectors by asset class

| Feature | US Equities | Crypto (Binance US) |
|---------|-------------|---------------------|
| Feature 1 | SPY 20-day rolling log return | BTC 20-day rolling log return |
| Feature 2 | VIX level (close) | BTC 10-day realized volatility (std of log returns) |
| Feature 3 | VIX term structure: VIX3M/VIX ratio (contango proxy) | Aggregate perp funding rate (8-hour, cross-exchange avg) |
| Feature 4 | SPY 5-day realized vol | Stablecoin net exchange inflows (USDT/USDC combined, 24h) |

#### HMM state topology

| Hidden State | Market Characteristics | Optimal Algorithmic Behavior |
|-------------|----------------------|------------------------------|
| **State 0: Engaged Trend** | Low-variance, persistent directional drift, high serial correlation | Aggressive capital allocation — gate OPEN |
| **State 1: Mean Reversion** | High-variance sideways consolidation, expanding ranges | Reduced allocation — gate OPEN (partial) |
| **State 2: Systemic Lapse** | Extreme volatility shocks, correlation convergence to 1.0 | Cease all new entries — gate CLOSED |

#### Training procedure

- `n_components=3`
- `covariance_type='full'` — captures cross-feature correlation within each state
- `n_iter=1000` — ensure Baum-Welch convergence; log final log-likelihood
- `StandardScaler`: fit **only** on training window, apply to test window (never fit on full history)
- Rolling window training to prevent look-ahead:
  - Split history into blocks: train on `T-lookback:T-1`, decode regime at `T`
  - Slide forward by `step_size` (e.g., 21 days = monthly refit)
  - Concatenate all decoded regime probabilities into a single time-indexed Series
- Save output to TimescaleDB table: `regime_states(timestamp, asset_class, state_0_prob, state_1_prob, state_2_prob, dominant_state)`
- Also pickle the final trained model to `models/hmm_{asset_class}_{date}.pkl` for audit

#### State labeling

After training, label states by inspecting mean emission vectors. The state with lowest VIX mean and highest SPY return mean = Engaged Trend. Highest VIX mean = Systemic Lapse. Middle = Mean Reversion. Store the label mapping in the pickle alongside the model object.

> ⚠️ **Convergence warning:** `hmmlearn`'s `GaussianHMM` uses random initialization and may converge to local optima. Run `n_init=10` parallel initializations and retain the model with the highest log-likelihood score. Log which init won for each rolling window.

---

### Task 2B — Runtime Regime Loader & Gate

**File:** `src/ggtrader/strategies/regime/hmm_filter.py`

Loads pre-computed regime probabilities from TimescaleDB and produces a boolean execution gate DataFrame for use at runtime.

```python
def load_regime_gate(
    asset_class: Literal['equity', 'crypto'],
    start: pd.Timestamp,
    end: pd.Timestamp,
    engaged_threshold: float = 0.65
) -> pd.Series:  # dtype bool, index = timestamps
```

- Returns `True` where `state_0_prob` (Engaged Trend) `>= engaged_threshold`
- Returns `False` (gate closed) for all rows where `dominant_state == 2` (Systemic Lapse)
- If a timestamp has no regime record, default to `False` (conservative fallback — do **not** forward-fill)
- Gate must be 1D Series (index = timestamps) — NOT 2D
- VectorBT broadcasts 1D Series across the asset axis automatically when used as a mask
- Expose `engaged_threshold` as a config parameter (default `0.65` for equities, `0.60` for crypto)

---

### Task 2C — Gate Integration into Portfolio Simulation

**File:** modify `src/ggtrader/strategies/momentum/cross_sectional.py`

Multiplies the Strategy 1 entry signal matrix by the HMM boolean gate to produce a `filtered_entries` matrix.

```python
# In CrossSectionalMomentum.run():
regime_gate = load_regime_gate(self.asset_class, start, end)

# Broadcast 1D gate across all asset columns
filtered_entries = entries_df & regime_gate.reindex(entries_df.index, fill_value=False)

# Pass filtered_entries to vbt.Portfolio.from_signals(...)
```

- The gate must `reindex` to the `entries_df` index using `fill_value=False` (not forward-fill)
- Log how many signals were suppressed per run: structlog info with count and percentage
- When `hmm_filter_enabled=False` in config, skip gate and pass raw `entries_df` unchanged
- Include a backtest comparison method: `run_with_filter()` vs. `run_without_filter()` for diagnostics

> **Key design principle:** The HMM gate is a risk-suppression mechanism, not an alpha source. If the gate suppresses >80% of signals over a 6-month period, this indicates regime mis-labeling — trigger an automatic retraining alert via the logging system.

---

## 5. Walk-Forward Optimization Harness

**File:** `src/ggtrader/backtesting/wfo.py`

Implements the institutional-grade WFO pipeline. Never optimize parameters on the full history — always use sequential in-sample / out-of-sample blocks.

- `WFOConfig`: `in_sample_bars=252`, `out_of_sample_bars=63`, `step_bars=63` (quarterly roll)
- Parameter grid: expose as `Dict[str, List[Any]]`; generate all combinations via `itertools.product`
- For each window split:
  - **Fit:** run Strategy 1 across all param combos on in-sample slice
  - **Select:** best combo by Sharpe ratio on in-sample
  - **Evaluate:** apply best combo to out-of-sample slice, record metrics
  - **Record:** save `(window_id, params, in_sample_sharpe, oos_sharpe, oos_max_dd)` to TimescaleDB
- Robustness check: flag any window where `oos_sharpe < 0.5 * in_sample_sharpe` as `'degraded'`
- Memory: process param grid in chunks of 200 combos max; `gc.collect()` between chunks
- Final output: `pd.DataFrame` of all OOS windows concatenated; compute aggregate OOS Sharpe
- **Gate to live deployment:** mean OOS Sharpe > 1.0 AND max OOS drawdown < 15%

> ✅ **Param stability test:** Plot OOS Sharpe vs. in-sample Sharpe across all windows. If the correlation is < 0.3, the strategy is curve-fit and should **NOT** be deployed. Include this as a diagnostic chart in the WFO report output.

---

## 6. Testing Requirements

Every module must have a corresponding test file. The following are mandatory test cases — the agent must implement all of them.

### Unit Tests — `factors.py`

- `test_momentum_no_lookahead`: verify that momentum at bar `i` never uses data from bar `i+1`
- `test_momentum_output_shape`: assert output shape == input shape
- `test_momentum_nan_warmup`: assert first `(window)` rows are NaN
- `test_liquidity_shock_positive`: assert all non-NaN values >= 0
- `test_btc_beta_residual_uncorrelated`: verify `corr(residual, btc_ret) < 0.05` on synthetic data

### Unit Tests — `signals.py`

- `test_top_decile_count`: in a 100-asset universe, entry signals on any row <= 10
- `test_no_signal_on_warmup`: no `True` signals in first `(window + gap)` bars
- `test_shift_applied`: signal at bar `i` is based on data through bar `i-1` only
- `test_sector_rank_isolation`: equity rank signals do not cross sector boundaries

### Unit Tests — `hmm_filter.py`

- `test_gate_default_false`: missing timestamps default to `False` (not forward-fill)
- `test_gate_suppresses_lapse`: `state_2` dominant rows always return `False` gate
- `test_filtered_entries_subset`: `filtered_entries` never has `True` where `raw_entries` is `False`

### Integration Tests

- `test_strategy1_equity_e2e`: run full backtest on 1 year of synthetic OHLCV data (5 equities, 1 sector); assert non-empty trades, Sharpe computable
- `test_strategy1_crypto_e2e`: same with crypto config (3 altcoins + BTC); assert BTC-beta stripped
- `test_hmm_gate_integration`: combine Strategy 1 + gate; assert filtered run has <= raw signal count
- `test_10pct_cap_respected`: in no bar does any single position exceed 10% of equity (check via `portfolio.asset_value.div(portfolio.value)`)

---

## 7. Coding Agent Task Prompts

Issue these prompts to the coding agent **in sequence** after loading the System Prompt from Section 1. Complete each task and pass all tests before moving to the next.

---

### TASK 1A — Numba Factor Functions

```
Implement the file src/ggtrader/strategies/momentum/factors.py.

Create three Numba @njit functions as specified:
  1. compute_momentum_nb(close_arr, window, gap) -> mom_out
  2. compute_liquidity_shock_nb(close_arr, vol_arr) -> liq_out
  3. strip_btc_beta_nb(alt_log_ret, btc_log_ret, window) -> residual

Requirements:
  - All inputs/outputs are 2D float64 NumPy arrays except btc_log_ret (1D)
  - Use LOG returns (log(p[t]) - log(p[t-1])) not simple returns
  - Fill warmup rows with np.nan
  - No Python objects inside @njit — only NumPy primitives
  - Add Google-style docstrings to each function (outside @njit scope)

Also implement the corresponding test file:
  tests/strategies/momentum/test_factors.py
  - All 5 unit tests from the spec (Section 6) must pass
  - Use synthetic deterministic data (np.random.seed(42)) for reproducibility
```

---

### TASK 1B — Rank Synthesis & Signals

```
Implement src/ggtrader/strategies/momentum/signals.py.

Class: CrossSectionalSignalGenerator
  __init__(self, config: MomentumConfig)
  generate(self, mom_df: pd.DataFrame, liq_df: pd.DataFrame,
           sector_map: dict | None = None) -> pd.DataFrame[bool]

Logic:
  1. Rank mom_df and liq_df cross-sectionally (axis=1, pct=True)
  2. If rank_mode='sector', groupby sector_map before ranking
  3. Composite = w_mom * mom_rank + w_liq * liq_rank
  4. Entry = composite >= composite.quantile(config.entry_percentile, axis=1)
  5. Shift entries by 1 bar (.shift(1))
  6. Return boolean DataFrame

Implement tests/strategies/momentum/test_signals.py — all 4 unit tests
from Section 6 must pass.
```

---

### TASK 1C — Config Schema

```
Implement src/ggtrader/strategies/momentum/config.py.

class MomentumConfig(BaseSettings):
  # All parameters from the table in Section 3 (Task 1C)
  # Use Literal types for mode fields
  # Pydantic validator: assert w_momentum + w_liquidity == 1.0
  # Pydantic validator: assert 0.5 < entry_percentile < 1.0

Provide two pre-built factory classmethods:
  MomentumConfig.for_equities() -> MomentumConfig
  MomentumConfig.for_crypto() -> MomentumConfig

These return instances pre-populated with the default values from the
spec table.
```

---

### TASK 1D — VectorBT Integration

```
Implement the class CrossSectionalMomentum in
src/ggtrader/strategies/momentum/cross_sectional.py.

class CrossSectionalMomentum:
  def __init__(self, config: MomentumConfig): ...
  def run(self, close_df, volume_df, sector_map=None,
          btc_close=None) -> vbt.Portfolio: ...
  def stats(self) -> pd.Series: ...   # passthrough
  def plot(self): ...                  # passthrough

run() must:
  1. Call factor functions (Task 1A) via vbt.IndicatorFactory wrapper
  2. Call signal generator (Task 1B)
  3. If asset_class='crypto': call strip_btc_beta_nb first, pass residual
     as close_arr to compute_momentum_nb
  4. Build vbt.Portfolio.from_signals with cash sharing, 10% size,
     correct fees/slippage from config
  5. Return portfolio object

Run both integration tests from Section 6 (equity + crypto e2e).
```

---

### TASK 2A — HMM Offline Training

```
Implement src/ggtrader/strategies/regime/train_hmm.py as a runnable script.

python -m ggtrader.strategies.regime.train_hmm \
  --asset-class [equity|crypto] \
  --start 2018-01-01 --end 2026-01-01 \
  --train-window 504 --step 21 \
  --n-inits 10

Script must:
  1. Load emission features from TimescaleDB (or accept CSV path for testing)
  2. Normalize with StandardScaler (fit on train window only)
  3. Train GaussianHMM(n_components=3, covariance_type='full', n_iter=1000)
  4. Run n_inits=10 random inits, keep highest log-likelihood model
  5. Decode regime probabilities via model.predict_proba()
  6. Label states by inspecting emission means (log procedure)
  7. Write regime_states table to TimescaleDB
  8. Pickle final model to models/ directory
  9. Print final log-likelihood and convergence status
```

---

### TASK 2B + 2C — Runtime Gate & Integration

```
Implement src/ggtrader/strategies/regime/hmm_filter.py.

def load_regime_gate(
  asset_class: Literal['equity', 'crypto'],
  start: pd.Timestamp,
  end: pd.Timestamp,
  engaged_threshold: float = 0.65
) -> pd.Series:  # dtype bool, index = timestamps

Then modify CrossSectionalMomentum.run() to:
  - Accept hmm_filter_enabled: bool = True parameter
  - If enabled: load gate, compute filtered_entries = entries & gate
  - If disabled: use raw entries
  - Log suppression count/percentage via structlog
  - Add run_with_filter() and run_without_filter() convenience methods

Implement all tests from Section 6 (gate unit tests + gate integration test).
```

---

### TASK 3 — Walk-Forward Optimization

```
Implement src/ggtrader/backtesting/wfo.py.

class WalkForwardOptimizer:
  def __init__(self, strategy_cls, param_grid: dict,
               wfo_config: WFOConfig): ...
  def run(self, close_df, volume_df, **kwargs) -> pd.DataFrame: ...
  def summary(self) -> dict: ...   # aggregate OOS stats
  def plot_robustness(self): ...   # IS vs OOS Sharpe scatter

WFOConfig fields: in_sample_bars, out_of_sample_bars, step_bars,
chunk_size (default 200), deployment_sharpe_threshold (default 1.0),
deployment_max_dd_threshold (default 0.15)

summary() must return: mean_oos_sharpe, std_oos_sharpe, max_oos_dd,
n_degraded_windows, deploy_ready (bool).

deploy_ready = mean_oos_sharpe > threshold AND max_oos_dd < dd_threshold
```

---

## 8. Quick Reference — DO / DO NOT

### DO

- Use `@njit` for all rolling factor computations
- Shift signals by 1 bar before portfolio simulation
- Train HMM offline; load decoded series at runtime
- Use `group_by=True` for cash-shared portfolio
- Strip BTC beta before ranking crypto momentum
- Rank equities within GICS sectors
- Use log returns internally in all factor math
- Chunk WFO parameter grids (max 200/batch)
- Run `n_init=10` for HMM to avoid local optima
- Gate deployment: OOS Sharpe > 1.0, max DD < 15%
- Default missing regime timestamps to `False` (not forward-fill)
- Use `structlog`; never `print()`

### DO NOT

- Use uncompiled Python loops in the signal hot path
- Optimize parameters on the full historical dataset
- Use simple price ratios (use log returns)
- Train HMM inside the backtesting loop
- Use `group_by=False` (per-asset sub-accounts)
- Apply raw altcoin returns without BTC-beta stripping
- Hardcode magic numbers — use Pydantic config
- Forward-fill missing regime timestamps
- Add dependencies without asking first
- Deploy without passing WFO gate thresholds

---

*This document is derived from the ggtrader Algorithmic Trading Strategy Research and Asset-Class Strategy reports. It is a technical specification for implementation only and does not constitute investment advice.*
