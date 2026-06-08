# Edge-Search Report — Gate / Universe / Fee Sweep (2026-06-08)

**Question:** Is there a deployable systematic edge for the per-coin WFO pipeline,
and if the default research run returns 0 coins, *why*? We swept the cheap levers
in order — gates → universe size → venue/fees — to find out.

**TL;DR:** No robust, deployable edge exists with the current strategy library on
any universe/venue tested. Fees dominate the outcome; Binance.US's 0.02%/side is
the only thing that makes anything marginally viable. Widening the universe surfaces
*more marginal* coins, never a strong one. Every "passing" portfolio loses to BTC
buy-and-hold. The cheap levers are now exhausted.

---

## Results across the three runs

| # | Universe | Venue / fee | Aggregate-gate passers | Full-cascade passers | Combined CAGR | Sharpe | Max DD | vs BTC buy-hold |
|---|----------|-------------|------------------------|----------------------|--------------:|-------:|-------:|-----------------|
| 1 | top-11 (min-vol $50k) | Binance.US 0.02% | 13 combos (DOGE, PEPE) | **0** | — | — | — | — |
| 2 | all-51 (min-vol $0) | Binance.US 0.02% | 22 combos | **2** (DOGE, TRX) | **+13.8%** | 0.54 | −30.4% | loses (BTC +38.8%) |
| 3 | top-100 (min-vol $0) | Kraken **maker 0.25%** | 13 combos | **2** (DOGE, SUI) | **−10.0%** | −0.07 | −55.0% | loses badly |

Result dirs (snapshots + per-worker `[Gates]` logs preserved):
- Run 1: `results/research/research_20260607_215008`
- Run 2: `results/research/research_20260608_083103`
- Run 3: `results/research/research_20260608_093714`

### Per-coin holdout (true OOS) for the passers

| Coin | Run | Strategy / exit | Fold consistency | Holdout |
|------|-----|-----------------|------------------|---------|
| TRX  | 2 (BUS) | bbands_mean_reversion / atr_trailing | 80% | +2.22%, Sortino 1.07 — but **1 trade** |
| DOGE | 2 (BUS) | rsi_reversal / atr_trailing | 60% | **−1.29%** (negative) |
| DOGE | 3 (Kraken) | ema_cross / trailing_stop | 60% | +0.17% (flat), 17 trades |
| SUI  | 3 (Kraken) | keltner_breakout / atr_trailing | 60% | **−8.06%** (negative) |

Both DOGE and SUI are Binance.US-listed (deployable); TRX too. None has a clean,
trade-rich, positive holdout.

---

## Key findings

1. **Fees dominate.** The same class of marginal strategies pass in both venues, but
   0.25% maker fees flip a +13.8% portfolio (Binance.US) into −10% (Kraken). This is
   hard confirmation that the Kraken→Binance.US migration was correct, and that the
   shelved `docs/superpowers/specs/2026-05-12-post-only-limit-entry.md` thesis (no
   edge survives Kraken-level fees) holds on a 100-coin universe.
2. **Widening the universe doesn't find an edge — it finds more marginal coins.**
   11 → 51 → 100 coins surfaced the same borderline names. **DOGE appears in every
   run** but its holdouts are flat/negative — persistently borderline, never real.
3. **Mean-reversion was the only flicker of life.** In the Binance.US all-51 run the
   only full-cascade survivors used MR entries (bbands_mean_reversion, rsi_reversal),
   while every momentum/trend entry (psar_adx, ema_cross) died. The (weak) edge on
   this universe, if any, is reversion — not trend. MR entries also suit limit buys
   (you buy into weakness, low miss-up), unlike momentum.
4. **The gates are working, not misconfigured.** Loosening them only deploys noise
   (negative-Sortino / fluke coins). See the gate cascade below.

---

## Why the default run returns 0 — the gate cascade (reference)

The pipeline applies three stages; a coin must clear **all** to enter `per_coin`:

1. **Aggregate gates** (`core/wfo_aggregate.py`, thresholds in `utils/run_config.py`):
   - `WFO_GATE_WFE_MIN` 0.5 — walk-forward efficiency = mean(test ann-ret)/mean(train ann-ret)
   - `WFO_GATE_PROFITABLE_FOLDS_MIN` 0.6 — fraction of OOS folds profitable
   - `WFO_GATE_PARAM_CV_MAX` 0.3 — max per-axis param CV across the 10 fold-winners
   - `WFO_GATE_DD_RATIO_MAX` 2.0 — mean(|test dd|)/mean(|train dd|)
2. **Per-coin selection** — among gate-passing (entry,exit) combos, pick highest mean Sortino.
3. **Selection gates** (`orchestrator._apply_wfo_selection_gates`):
   - `MIN_FOLD_CONSISTENCY` 0.38
   - `MIN_VALID_TRAIN_FOLDS` 3 — a fold is "valid" if ≥1 param combo passed the training
     gate (finite `is_sharpe`). **This is what killed the top-11 run's only survivors**
     (DOGE had 2 valid folds, PEPE 1): they won on lucky OOS folds while training was
     rejected on most folds — exactly the fluke this gate targets.

### Offline gate-replay tool
`scripts/gate_replay.py` re-applies the 4 aggregate gates to a saved
`wfo_stats_snapshot.json` in seconds (imports the real `wfo_aggregate.py` functions;
validated to match the live run's `[Gates]` verdicts exactly). Use it to sweep
thresholds without a 18–90 min re-run. It does NOT model the stage-3 selection gates.

### Fee override for experiments
`utils/run_config.py:fees_for_exchange()` now honors `GGTRADER_FEES=<rate>`
(no-op when unset). Run a venue's universe at a different fee tier with, e.g.:
```
docker compose run --rm -v $PWD/src:/app/src \
  -e EXCHANGE=kraken -e GGTRADER_FEES=0.0025 \
  ggtrader_live python -u ggt.py research --top 100 --min-volume 0
```

---

## Exhausted vs remaining levers

**Exhausted (no edge found):** gate thresholds, universe size (11→51→100), venue/fee
tier. More runs along these axes will not change the conclusion.

**Remaining real lever — the strategy library.** The entries are standard TA; only the
MR ones showed any life. The high-upside path is a **reversion-focused / regime-aware
redesign**, not adding more standard-TA entries. NOTE (per Flynn): strategy-library
expansion has already been explored once before — so the next attempt needs a
*genuinely different* approach (e.g. regime conditioning, a real edge hypothesis),
not another batch of indicators.

**Baseline to beat:** a low-fee buy-and-hold / DCA on BTC has out-returned every
strategy variant tested here. Any new work should be measured against that, not zero.
