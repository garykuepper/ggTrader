# Live Trading Gap Analysis

**Date:** 2026-03-28
**Purpose:** Precise specification of divergences between the research pipeline and the current live execution code. This document is the implementation spec for live development.

---

## TL;DR

The live execution code (`execution_engine.py`) correctly loads WFO-optimised per-coin parameters and connects to Kraken, but is missing two critical research-pipeline safeguards:

1. **BTC regime filter** — prevents new entries when BTC is in a downtrend; applied in Phase 2/3 but absent in live.
2. **Strategy exit signal execution** — the WFO selects a best exit per coin; live computes exit signals but ignores them, relying on exchange TSL only.

These two gaps mean live trading is systematically more aggressive (enters during bear markets) and less precise (ignores profit-taking/exit signals) than the backtested strategy.

---

## 1. Signal-Generation Gaps

### 1.1 BTC Regime Filter — **CRITICAL**

| | Research | Live |
|---|---|---|
| **Applies?** | Yes — Phase 2 & 3 via `_compute_btc_regime_mask()` + `_apply_tiered_regime_mask()` in `orchestrator.py:448-458` | **No** — `_compute_latest_signals()` in `execution_engine.py:142-196` has no regime check |
| **Logic** | Three-tier: BTC-corr ≥ 0.5 → BTC EMA filter; BTC-corr in [0.3, 0.5) → altcoin index EMA; below 0.3 → no filter | Not implemented |
| **Config keys** | `BTC_REGIME_FILTER=True`, `BTC_REGIME_FILTER_SHORT_EMA=50`, `BTC_REGIME_FILTER_MIN_CORRELATION=0.5`, `ALTCOIN_REGIME_FILTER=True`, `ALTCOIN_REGIME_FILTER_CORR_MIN=0.3` | Not read by live code |

**What to build:**
Port `_compute_btc_regime_mask()` and `_apply_tiered_regime_mask()` (both in `orchestrator.py`) into `_compute_latest_signals()`. The live data fetch already loads `LOOKBACK_LIMIT=200` bars per coin; add BTC to the fetch so EMA(50)/EMA(200) can be computed. Regime check blocks new entries — open positions / TSL already placed are unaffected.

**EMA warmup note:** Research pre-loads 200 extra bars *before* the backtest start date to warm the EMA. In live trading there is no fixed start date — fetching the last 200 bars is sufficient to compute a warm EMA(200) on the current bar. No special warmup loading is needed in live.

---

### 1.2 Strategy Exit Signal Execution — **HIGH**

| | Research | Live |
|---|---|---|
| **Applies?** | WFO selects best exit per coin (atr_trailing / fixed_sl_tp / trailing_stop); exit signals close the position | **Computed but ignored** — `execution_engine.py:296-302` logs "STRATEGY EXIT SIGNAL" but takes no action |
| **Current reliance** | Both exit signal + stop trigger the close | TSL placed at order time; strategy exit is dead code |

**What to build:**
For coins whose `best_exit` is `fixed_sl_tp`, the exit signal from `compute_exits()` is the primary close mechanism (fixed SL/TP levels hit inside the indicator logic). For `atr_trailing` and `trailing_stop`, the exchange TSL is the right mechanism. The logic should be:

```
if sig["exit"] is True:
    if best_exit == "fixed_sl_tp":
        # cancel TSL, place market sell
    else:
        # TSL is already handling it; log and do nothing
```

---

### 1.3 EMA Warmup (Not a gap for live)

Research pre-loads `EMA_WARMUP_BARS` before `START_DATE` so the regime EMA is warm from bar 0 of the historical window. In live trading, each loop iteration fetches the last `LOOKBACK_LIMIT` bars (`cached_loader.py:137`), which naturally provides enough history to compute a warm EMA. **No action needed.**

---

## 2. State Management Gaps

### 2.1 No Exchange Reconciliation on Startup — **HIGH**

| | Research | Live |
|---|---|---|
| **Position state** | Stateless (each backtest run is fresh) | Loaded from `active_positions.json` (`execution_engine.py:122-125`) |
| **Reconciliation** | N/A | **None** — no exchange query on startup |

**Risk:** Bot crashes after placing a buy order but before saving state. On restart, the position exists on Kraken but not in the JSON file. Next entry signal buys again (doubled position). Alternatively, Kraken closes a TSL while the bot is down; the JSON still shows an open position; on restart the bot ignores the coin thinking it's already held.

**What to build:**
On startup (before the event loop), query `exchange.fetch_open_orders()` and `exchange.fetch_balance()` to build ground-truth position state. Reconcile against `active_positions.json`: add missing positions, remove stale ones, log any discrepancies.

---

### 2.2 Orphaned Positions on Partial Order Failure — **MEDIUM**

`execution_engine.py:265-293`: if the buy order succeeds but the TSL order fails, the position is saved with `tsl_order_id=None`. The position is then held indefinitely with no stop protection.

**What to build:**
After a TSL placement failure, attempt a retry (1–2 times). If retries fail, immediately place a market sell to close the unprotected position. Log an alert.

---

### 2.3 Incomplete Trade Journal — **LOW**

Only TSL order ID is persisted per position. No entry timestamp, no strategy exit signal timestamp. Makes post-mortem analysis and audit difficult.

**What to build:**
Add `entry_time`, `exit_time`, `exit_reason` (tsl / strategy_signal / manual) to the position record.

---

## 3. Risk Control Gaps

| Control | Research Config | Live Status | Action |
|---|---|---|---|
| Max allocation per coin | `MAX_COIN_ALLOCATION=0.25` | Applied only if `portfolio_weights.json` present | Enforce cap as hard limit regardless of weights file |
| Max coins per strategy | `MAX_COINS_PER_STRATEGY=10` | Not checked | Filter per-coin results at load time |
| Min robustness gate | `MIN_ROBUSTNESS_SCORE=0.1` | Not checked — all coins in `per_coin_results` used | Drop coins below threshold at load time |
| Daily loss circuit breaker | n/a | Not implemented | Halt new entries if portfolio drops >X% intraday |
| Minimum order size | n/a | Not validated | Check `exchange.markets[symbol]["limits"]["amount"]["min"]` before placing |
| Free balance check | n/a | Not validated — uses fixed `CAPITAL_PER_TRADE` | Verify `free_balance >= capital_per_trade` before entry |

---

## 4. Config Key Mapping

Research config keys (`full_pipeline_config()` in `run_config.py`) that live must also honour:

| Research Key | Default | Must live use it? | Current Live Status |
|---|---|---|---|
| `BTC_REGIME_FILTER` | `True` | **Yes** | Not read |
| `BTC_REGIME_FILTER_SHORT_EMA` | `50` | **Yes** | Not read |
| `BTC_REGIME_FILTER_MIN_CORRELATION` | `0.5` | **Yes** | Not read |
| `ALTCOIN_REGIME_FILTER` | `True` | **Yes** | Not read |
| `ALTCOIN_REGIME_FILTER_CORR_MIN` | `0.3` | **Yes** | Not read |
| `EMA_WARMUP_BARS` | `200` | No (equivalent to `LOOKBACK_LIMIT`) | Read as `LOOKBACK_LIMIT` ✓ |
| `MAX_COIN_ALLOCATION` | `0.25` | Yes — enforce as hard cap | Applied only when weights present |
| `MAX_COINS_PER_STRATEGY` | `10` | Yes — filter at load time | Not enforced |
| `MIN_ROBUSTNESS_SCORE` | `0.1` | Yes — filter at load time | Not enforced |
| `FEES` | `0.004` | No — live uses actual exchange fees | n/a |
| `SLIPPAGE` | `0.003` | No — live uses actual fill prices | n/a |
| `INTERVAL` | `"4h"` | Yes — already used | ✓ |

---

## 5. Data Loading Differences

| Aspect | Research | Live | Gap? |
|---|---|---|---|
| Source | TimescaleDB + CCXT tail (`load_hybrid_validation_ohlcv`) | `CachedExchangeLoader` (DB → CCXT fallback) | No — same hybrid pattern |
| Stale data | N/A (fixed historical) | Falls back silently if CCXT fails; threshold is >1.5× interval | Low — log a warning when falling back |
| CCXT rate limits | N/A | Returns empty DataFrame on `NetworkError`; no retry | Medium — add 3-retry with backoff |
| Multi-symbol BTC fetch | Always includes BTC for regime filter | BTC not always fetched if not in coin universe | **Must fix** — always fetch BTC for regime check |

---

## 6. Recommended Implementation Order

Priority is based on impact to research–live consistency:

1. **BTC regime filter** (`execution_engine.py:_compute_latest_signals()`)
   — Port `_compute_btc_regime_mask()` and `_apply_tiered_regime_mask()` from `orchestrator.py`.
   — Always fetch BTC OHLCV even if BTC is not in the coin universe.
   — Config keys: `BTC_REGIME_FILTER`, `BTC_REGIME_FILTER_SHORT_EMA`, `BTC_REGIME_FILTER_MIN_CORRELATION`, `ALTCOIN_REGIME_FILTER`, `ALTCOIN_REGIME_FILTER_CORR_MIN`.

2. **Strategy exit execution for `fixed_sl_tp` coins** (`execution_engine.py:296-302`)
   — If `best_exit == "fixed_sl_tp"` and `sig["exit"]` is True: cancel TSL, place market sell.
   — For `atr_trailing` / `trailing_stop`: TSL is correct, no change.

3. **Exchange position reconciliation on startup** (`execution_engine.py:__init__` or new `_reconcile_state()`)
   — Fetch open orders + balance from Kraken.
   — Merge with JSON state; log discrepancies.

4. **Risk control gates at load time** (`execution_engine.py:load_optimized_params()`)
   — Drop coins with `robustness_score < MIN_ROBUSTNESS_SCORE`.
   — Enforce `MAX_COINS_PER_STRATEGY`.
   — Cap individual weights at `MAX_COIN_ALLOCATION`.

5. **Free balance + min order size validation** (before `_execute_market_buy_order()`)
   — Verify free balance ≥ capital requested.
   — Check exchange minimum order amount.

6. **CCXT retry with backoff** (`exchange_loader.py:fetch_ohlcv()`)
   — 3 retries with exponential backoff on `NetworkError` / rate-limit response.

7. **Daily loss circuit breaker** (new, in the main event loop)
   — Track portfolio value at loop start.
   — Skip all new entries if intraday drawdown exceeds configurable threshold.

---

## Key Source Files

| File | Relevance |
|---|---|
| `src/ggTrader/core/execution_engine.py` | Main live execution loop |
| `src/ggTrader/core/orchestrator.py:448-458` | Regime filter logic to port |
| `src/ggTrader/data/live/cached_loader.py` | Live data fetching |
| `src/ggTrader/data/live/exchange_loader.py` | CCXT wrapper |
| `src/ggTrader/utils/run_config.py:97-206` | Full research config (source of truth for defaults) |
| `scripts/auto_trader.py` | Automated recalibration loop |
