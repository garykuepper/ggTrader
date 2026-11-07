# S4 Strategy Port Inventory (facts and classification)

Scope: every entry of `STRATEGY_REGISTRY` (`src/ggTrader/lab/strategies/__init__.py:56`). Read-only; no repo file was edited. Line numbers are of the working tree on `main` as read on 2026-11-06. Claims are from code opened in this pass unless marked **UNVERIFIED** or attributed to the earlier vbt notes.

Registry count: **38 entries** (counted from the dict, and confirmed by importing the registry; see `scratchpad/vbt-spec/s4_grid_count.py`).

## 0. Definitions used

- **A**: expressible with vectorbt built-ins and signal accessors only (candidate names: `MA`, `MSTD`, `BBANDS`, `RSI`, `MACD`, `ATR`, `OBV`, `STOCH`, `.vbt` crossed/boolean accessors).
- **B**: vectorbt built-ins plus thin glue (vote/threshold combination, time stop, point-in-time (PIT) mask, calendar/event mask, cross-sectional ranking, weekly resample, hysteresis helper).
- **C**: needs custom indicator or data logic with no vectorbt equivalent (rolling beta/residual variance, rank-IC weighting, Kelly sizing, pair selection, macro real-FX index, event-detection loops, LLM-scored text). Note an `IndicatorFactory.from_apply_func` wrapper could still run each of these.
- **D**: closed NO-GO research (per `research/RESEARCH_SNAPSHOT.md` §registry table); port may be deferred and reproduced from a tagged legacy snapshot.
- **Primary class** = D when the snapshot verdict is a NO-GO (of any kind), otherwise the technical class. **Alt class** = the technical class for the strategy, so every row keeps its A/B/C view. Where the two differ, the reason is in the last column.

Two vectorbt caveats that matter for any A/B claim (from the earlier note `scratchpad/vbt-research/B-vectorbt-1x-and-rust.md`, not re-measured here):
- `RSI.run` default is a simple-MA RSI (`ewm=False`), not the Wilder EWM used by `rsi_raw`/`rsi_signals` (`indicators.py:118-139`). The ewm=True variant was not measured there: **UNVERIFIED**.
- `MSTD.run` default `ddof=0`; pandas `.rolling().std()` uses `ddof=1`. `MACD.run` default is simple-MA (`macd_ewm=False`); the hand-rolled MACD uses `ewm(span, adjust=False)`. The B note reports 0 difference for `macd_ewm=True, signal_ewm=True`.

## 1. Master table (38 rows, registry order)

Columns: `Name` = registry key; `Class (file:line)` = class definition; `Grid raw/valid` = product of `sweep_params()` before and after `_is_valid_combo` (`lab/sweep.py:38-60`), counted by `s4_grid_count.py` using the same `build_grid` filter semantics; `Snap` = line in `research/RESEARCH_SNAPSHOT.md` (the registry table, lines 145-182).

| # | Name | Class (file:line) | target_kind | Hand-written indicators / rolling stats | External data beyond OHLCV | Paper path (`src/ggTrader/paper/`, `scripts/paper_trade.sh`) | Snapshot verdict (quoted, Snap line) | Grid raw/valid (sweep_params line) | Primary | Alt / why |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `ema_cross` | `EmaCrossSignal` signals.py:81 (grid :97) | signals | `ema_signals` indicators.py:141 (ewm span, adjust=False; crossover by shift) | none | Class not imported. Its voter function `ema_signals` is on the live path via `EnsembleSignal` | "Component; independent edge unverified" (145) | 20 / 16 (fast<slow filter removes 4 pairs) | A | — |
| 2 | `wfo_tournament` | `WfoTournamentSignal` signals.py:151 (grid :167) | signals | `_vectorized_ema_tournament` signals.py:~40-75 (4 EMA combos, `vbt.Portfolio.from_signals` + `sharpe_ratio`, signals.py:37/57/69); `ema_signals`; per-period pair cache in `to_targets` | none | no | "Historical soft NO-GO \| Pre-rewrite tournament; no fresh modern measurement" (146) | 4 / 4 | D | B: already uses vbt `from_signals`; per-rebalance selection and piecewise signal assembly are glue |
| 3 | `bb_reversion` | `BollingerReversionSignal` signals.py:318 (grid :334) | signals | `bb_signals` indicators.py:67 (pandas rolling mean and std, lower band, prev/now crossing) | none | no (voter only, via `EnsembleSignal`) | "Component; independent edge unverified" (147) | 16 / 16 | A | BBANDS/MA/MSTD plus crossed accessors |
| 4 | `rsi_reversion` | `RsiReversionSignal` signals.py:381 (grid :404) | signals | `rsi_signals` indicators.py:118 (Wilder-style `ewm(alpha=1/p, adjust=False)`, cross at 30 and 50) | none | no (voter only) | "Component; independent edge unverified" (148) | 27 / 27 | A | A with caveat: vbt RSI default is not Wilder (see §0); ewm=True equivalence **UNVERIFIED** |
| 5 | `macd_divergence` | `MACDDivergenceSignal` signals.py:532 (grid :553) | signals | `macd_signals` indicators.py:182 (ewm MACD and signal; rolling min of close and of histogram over `divergence_window`; warmup mask `iloc[:slow+signal+window]=False`; exit = histogram crosses below 0) | none | no (voter only) | "Component; independent edge unverified" (149) | 8 / 8 | B | vbt `MACD` (with `macd_ewm=True, signal_ewm=True`) plus rolling-min divergence and warmup mask glue |
| 6 | `volume_bb_reversion` | `VolumeBBReversionSignal` signals.py:599 (grid :620) | signals | `volume_bb_signals` indicators.py:237 (BB entry AND `volume > vol_mult * SMA(volume)`; exit = BB-mid cross) | volume (OHLCV) | no (voter only) | "Component; independent edge unverified" (150) | 12 / 12 | A | BBANDS, MA on volume, boolean AND, crossed accessors. B if the AND is counted as glue |
| 7 | `mtf_reversion` | `MultiTimeframeReversionSignal` signals.py:669 (grid :692) | signals | `mtf_signals` indicators.py:294; `_weekly_rsi` indicators.py:281 (`resample("W").last()`, RSI, `reindex(method="ffill")`) | none | no (not in the default voter set) | "Historical NO-GO \| Added timeframe confirmation degraded the ensemble" (151) | 32 / 32 | D | B: weekly resample and reindex glue on vbt RSI/BBANDS |
| 8 | `overnight_gap` | `OvernightGapReversionSignal` signals.py:454 (grid :478) | signals | `overnight_gap_signals` indicators.py:84 (gap = (open - prev close)/prev close; rolling mean/std z-score; crossings) | open price (OHLCV) | no | "Historical NO-GO \| Reported Sharpe −0.01; pre-PIT evidence" (152) | 27 / 27 | D | B: derived gap series then MA/MSTD and crossing; no vbt gap indicator |
| 9 | `ensemble` | `EnsembleSignal` ensemble.py:55 (grid :120) | signals | Voter indicators (`bb_signals`, `rsi_signals`, `ema_signals`, `macd_signals`, `volume_bb_signals`; `mtf_signals` only if voted). Vote sum and threshold (ensemble.py:195-198), RSI exit OR exit-vote threshold (:201-204), time stop `_apply_time_stop` (:211) | none beyond OHLCV. Live universe is PIT S&P membership via `universe_members_asof` (`data/core/index_constituents.py:195`), called from `paper/signal_runner.py` | **YES.** `paper/signal_runner.py:11,86` (`EnsembleSignal(cfg)`, `select`, `to_targets`); `paper/overlay.py:13,55` (same, then `lab/simulate.simulate_signals`, `overlay.py:67`). `scripts/paper_trade.sh` runs `ggt.py paper` | "Daily-bar construction NO-GO; live unverified" (153). Deployed live (paper), per `paper/` wiring | 48 / 48 | B | A-able voters plus vote-count glue, RSI-exit OR, time stop, PIT eligibility mask |
| 10 | `ensemble_ic` | `EnsembleICSignal` ensemble_ic.py:35 (grid :98) | signals | Voter raw scores `rsi_raw`, `bb_raw`, `ema_raw`, `macd_raw`, `vbb_raw` (indicators.py:342-380); `ic_weight_schedule` ic_weights.py:42 (`forward_returns` :14, `daily_cross_sectional_ic` :19: per-date Spearman via rank; trailing window; quarterly weights) | none | no | "NO-GO under historical PIT rebaseline \| Sharpe 0.41 versus core 0.59" (154) | 15 / 15 | D | C: cross-sectional rank-IC weighting has no vbt equivalent |
| 11 | `ensemble_kelly` | `EnsembleKellySignal` ensemble_kelly.py:24 (grid :87) | signals | Base = `EnsembleSignal._generate_signals`; sizing `kelly_sizes` kelly.py:129 (`extract_trades` :54, `expanding_kelly_fraction` :84) | none | no | "NO-GO under historical PIT rebaseline \| Sharpe 0.50 versus core 0.59; prior garbage-bar result rejected" (155) | 3 / 3 | D | C: custom trade extraction and expanding Kelly sizing |
| 12 | `congress_trades` | `CongressTradeMirrorStrategy` congress_trades.py:42 (grid :65) | weights | None (no indicator). PIT gate in `select`: filing date plus 1-day lag (`REPORT_LAG_DAYS`, congress_trades.py:34), hold window `hold_days` | House STOCK Act PTR transactions: `lab/house_ptr_data.py:182` `load_house_ptr_transactions` (DB-cached) | no | "Historical NO-GO, tape caveat \| Matched-window Sharpe 0.36 versus SPY 0.58" (156) | 3 / 3 | D | B: PIT event mask and hold window; no vbt indicator needed |
| 13 | `conviction_bb` | `ConvictionBBSignal` conviction.py:14 (grid :39) | signals | `bb_signals` indicators.py:67; `_compute_conviction_sizes` conviction.py:54 recomputes rolling SMA/STD inline (duplicates the BB math) | none | no | "Shipped, no independent current GO \| Conviction sizing; no dedicated modern validation" (157) | 36 / 36 | A | BBANDS plus a per-entry size array (vbt `size`) |
| 14 | `ensemble_conviction` | `EnsembleConvictionSignal` ensemble.py:280 (grid :339) | signals | Voter indicators plus strength functions `bb_strength` :152, `rsi_strength` :161, `ema_strength` :174, `macd_strength` :215, `volume_bb_strength` :266, `mtf_strength` :327 (indicators.py); conviction average of agreeing voters; size = min + conviction*(max-min) (ensemble.py:~435-441) | none | no | "Soft NO-GO \| Historical conviction/MTF lineage; not independently remeasured" (158) | 48 / 48 | D | B: vbt-able voters plus normalisation and masked averaging glue |
| 15 | `xs_momentum` | `CrossSectionalMomentum` momentum.py:13 (grid :23) | weights | Inline 12-1 return in `select` (momentum.py:~36-42: `closes.iloc[-(lookback+1)]`, `iloc[-(skip+1)]`); top-N and equal weight | none | no | "Closed NO-GO on 100-year screen after publication \| French idea-level screen: neither size passes Holm" (159) | 12 / 12 | D | B: vbt ROC-type return plus cross-sectional rank glue |
| 16 | `dual_momentum` | `DualMomentum` momentum.py:76 (inherits grid :23) | weights | Same as 15, plus `momentum >= 0` filter (momentum.py:79-80) | none | no | "Closed NO-GO on 100-year screen after publication \| Same portfolio as xs_momentum on the corrected replay" (160) | 12 / 12 (inherited) | D | B |
| 17 | `idio_vol` | `IdioVolStrategy` idio_vol.py:39 (grid :59) | weights | `idiosyncratic_variance` idio_vol.py:14 (rolling cov and var vs EW universe market factor; beta; residual variance) | none | no | "Historical NO-GO \| Standalone 0.57 versus SPY 0.58; highly correlated blend" (161) | 6 / 6 | D | C: rolling beta and residual variance; no vbt equivalent (**UNVERIFIED** that vbt has no cov/beta accessor) |
| 18 | `index_deletion_fade` | `IndexDeletionFadeStrategy` index_deletion.py:54 (grid :76) | weights | None. Event list from `deletion_events` index_deletion.py:36 (diff of consecutive membership snapshots) | S&P 500 membership history: `data/core/index_constituents.py:47` `load_sp500_history` (CSV) | no | "Historical NO-GO, tape caveat \| Original Sharpe 0.30, drawdown −68.7%" (162) | 3 / 3 | D | B: event mask and hold window |
| 19 | `insider_cluster_buy` | `InsiderClusterBuyStrategy` insider_cluster.py:80 (grid :104) | weights | `cluster_events` insider_cluster.py:25 (python loop over trades; 14-day window; min 3 distinct insiders; known date = max(filing date, trade date)) | SEC Form 4 transactions: `lab/form4_data.py` (`load_form4_transactions`, DB-cached) | no | "Historical NO-GO; data feasibility UNRESOLVED (2026-10-23) \| Original matched-window Sharpe 0.39" (163) | 3 / 3 | D | C: event-detection loop with filing-date discipline |
| 20 | `max_effect` | `MaxEffectStrategy` max_effect.py:22 (grid :39) | weights | `trailing_max_return` max_effect.py:14 (`pct_change` then rolling max); lowest-quintile bucket | none | no | "Historical NO-GO \| Original Sharpe 0.39–0.45, high blend correlation" (164) | 6 / 6 | D | B: rolling max plus cross-sectional quintile (vbt has no MAX indicator: **UNVERIFIED**) |
| 21 | `pairs_stat_arb` | `PairsStatArb` pairs_stat_arb.py:149 (grid :183) | weights | `enumerate_sector_pairs` :24; `pairwise_correlations` :42; `spread_zscore` :63 (log spread, rolling mean and std); `_cached_qualifying_pairs` :113; `pair_signal` :77 is defined but has no call site in the module; stateful hold logic in `select` | GICS sector map: `lab/strategies/registry.py` `load_sector_map` (reads `data/universe/sp500_sectors.json`) | no | "NO-GO on corrected-tape rerun \| Sharpe −0.42, WFE −0.16" (165) | 54 / 54 | D | C: pair selection and stateful hold logic |
| 22 | `pead` | `PeadStrategy` pead.py:37 (grid :70) | weights | `_most_recent_surprise` (per-symbol last surprise inside `max_age_days`); quintile | earnings surprises: `lab/earnings_surprise_data.py:127` `load_earnings_surprises`; PIT lag via `available_as_of` | no | "Historical NO-GO; data feasibility UNRESOLVED (2026-10-23) \| Matched-window Sharpe tied SPY" (166) | 6 / 6 | D | B on the logic (PIT mask, rank). The data side is C |
| 23 | `retail_attention` | `RetailAttentionStrategy` retail_attention.py:36 (grid :61) | weights | `_spikes` (latest / trailing-mean − 1) inline; top quintile | Google Trends: `lab/google_trends_data.py:123` `load_search_interest` | no | "Paused; no verdict \| Google Trends backfill blocked" (167) | 6 / 6 | B | Not D: no verdict. Blocked data |
| 24 | `short_interest` | `ShortInterestStrategy` short_interest.py:34 (grid :64) | weights | `_days_to_cover_trend` inline (pct change over `lookback_cycles`); bottom quintile | FINRA short interest: `lab/short_interest_data.py:227` `load_short_interest` | no | "Historical NO-GO, tape caveat \| Original Sharpe 0.27 versus SPY 0.61" (168) | 6 / 6 | D | B |
| 25 | `short_volume_ratio` | `ShortVolumeRatioStrategy` short_volume_ratio.py:35 (grid :61) | weights | `_avg_ratios` inline (tail-mean of daily ratio); long-short quintiles | FINRA daily short volume: `lab/short_volume_data.py:135` `load_short_volume` | no | "Historical NO-GO, tape caveat \| Original Sharpe 0.21 versus SPY 0.72; frequent halt" (169) | 6 / 6 | D | B |
| 26 | `fx_hedge_overlay` | `FxHedgeOverlayStrategy` fx_hedge_overlay.py:114 (grid :146) | weights | `trend_signal` :94 (log ratio momentum); `real_fx_index` :82 (CPI-adjusted real FX); `rolling_zscore` :76; `unhedged_weight` :108 (tanh squash) | FRED: TB3MS, CPIAUCSL, per-pair policy rate, CPI and spot series (`lab/fred_data.py:131` `load_fred_series`); ETF prices EWJ/DXJ, EZU/HEZU | no | "Historical NO-GO \| Dynamic 0.41 versus static hedged 0.73 Sharpe" (170) | 6 / 6 | D | C: macro real-FX index with CPI; no vbt equivalent |
| 27 | `headline_sentiment` | `HeadlineSentimentStrategy` headline_sentiment.py:36 (grid :58) | weights | groupby mean of sentiment score; top quintile | Alpaca news plus LLM scores: `lab/headline_sentiment_data.py:348` `load_sentiment_scores`; scoring via LiteLLM proxy (`_default_llm_call` :182) | no | "Provisional NO-GO; underpowered \| Three usable folds" (171) | 6 / 6 | D | C: LLM-scored text pipeline (ranking alone is B) |
| 28 | `fomc_drift` | `FomcDriftStrategy` fomc_drift.py:50 (grid :70) | signals | `_event_signals` fomc_drift.py:117 (calendar offsets; entry `entry_offset_days` before, exit `hold_days` after) | FOMC announcement dates: cache `data/universe/fomc_meeting_dates.csv` (fomc_drift.py:~36), else scrape via `lab/fomc_calendar.py:112` | no | "Corrected-harness NO-GO \| Day-clustered gross t 2.258, net t 0.981 over 113 dates" (172) | 6 / 6 | D | B: calendar event mask |
| 29 | `month_end_treasury` | `MonthEndTreasuryStrategy` month_end_treasury.py:31 (grid :43) | signals | `month_end_signals` month_end_treasury.py:77 (month-ends from the bar index, groupby period) | none beyond ETF prices (IEF, TLT, EDV) | no | "NO-GO as SPY-funded sleeve \| Real standalone premium did not beat its funding leg" (173) | 12 / 12 | D | B: calendar mask |
| 30 | `commodity_trend` | `CommodityTrendStrategy` commodity_trend.py:86 (grid :110) | weights | `cross_sectional_momentum_scores` :53 (12-1 return); `regime_vol_zscore` :70 (rolling std, mean across symbols, z-score); skip rebalance if z above threshold | none beyond 14 commodity ETF prices (hardcoded list, commodity_trend.py:~24-40) | no | "Historical NO-GO \| Sharpe 0.13; trails static commodity benchmarks" (174) | 9 / 9 | D | B: vbt MSTD/ROC-type indicators plus rank and regime gate |
| 31 | `cross_asset_sleeve` | `CrossAssetSleeveStrategy` cross_asset_sleeve.py:47 (grid :66) | weights | `leg_is_on` :35 (sign of trailing return); static arm has no indicator | prices only: SPY, TLT, GLD, PDBC, BIL | no | "Static shadow GO; trend not preferred \| Diversification/gold exposure, not validated timing skill" (175) | 3 / 3 | B | Static arm is trivially A. Trend arm needs trailing-return and month-end glue |
| 32 | `treasury_curve` | `TreasuryCurveStrategy` treasury_curve.py:82 (grid :104) | weights | `curve_slope_zscore` :57 (rolling mean and std z over 1260 bars); `curve_regime_weights` :70 (one-hot) | FRED DGS10 and DGS2 via `lab/fred_data.py:131`; default loader treasury_curve.py:40 | no | "Historical NO-GO \| Dynamic 0.19 versus static SHY 0.90 Sharpe" (176) | 9 / 9 | D | B: MA/MSTD plus one-hot regime |
| 33 | `leveraged_rotation_sp500` | `LeveragedRotationSp500` leveraged_rotation.py:172 (base :101; grid :126) | weights | `compute_breadth` :21 (fraction of EnsembleSignal entries across the breadth universe); `_cached_breadth` :90 (runs `EnsembleSignal`); `rotate_positions` :32 (hysteresis loop; min hold) | none beyond OHLCV (SPY-universe constituents; UPRO/SPXU/SSO/SDS) | no | "NO-GO \| Breadth-driven leveraged rotation collapsed" (177) | 54 / 54 | D | C: hysteresis state machine on ensemble breadth |
| 34 | `leveraged_rotation_nasdaq100` | `LeveragedRotationNasdaq100` leveraged_rotation.py:179 | weights | As 33 (Nasdaq-100 universe; TQQQ/SQQQ/QLD/QID) | as 33 | no | "NO-GO \| Same failed mechanism, separate index universe" (178) | 54 / 54 | D | C |
| 35 | `leveraged_rotation_russell2000` | `LeveragedRotationRussell2000` leveraged_rotation.py:186 | weights | As 33 (Russell 2000; TNA/TZA/UWM/TWM) | as 33 | no | "NO-GO \| Same failed mechanism, separate index universe" (179) | 54 / 54 | D | C |
| 36 | `leveraged_trend_sp500` | `LeveragedTrendSp500` leveraged_trend.py:168 (base :77; grid :101) | signals | Underlying SMA (`_trend_state` :46; inline rolling mean in `sweep_signals`); `rotate_positions` min-hold helper (imported from leveraged_rotation.py:32); `_entries_exits` :70 | underlying index OHLCV via `lab/data.py:237` `load_ohlcv` (SPY) | no | "NO-GO \| Trend/volatility timing lost to own buy-and-hold" (180) | 72 / 72 | D | B: vbt MA plus min-hold glue |
| 37 | `leveraged_trend_nasdaq100` | `LeveragedTrendNasdaq100` leveraged_trend.py:173 | signals | As 36 (QQQ; TQQQ/QLD) | as 36 | no | "NO-GO \| Same failed mechanism, separate index universe" (181) | 72 / 72 | D | B |
| 38 | `leveraged_trend_russell2000` | `LeveragedTrendRussell2000` leveraged_trend.py:178 | signals | As 36 (IWM; TNA/UWM) | as 36 | no | "NO-GO \| Same failed mechanism, separate index universe" (182) | 72 / 72 | D | B |

Notes on the table:
- Snapshot verdicts are quoted from the registry table (`research/RESEARCH_SNAPSHOT.md:145-182`). The table cites reports under `research/`, which I did not open; only the verdict text is quoted.
- Paper-path column is limited to `src/ggTrader/paper/` and `scripts/paper_trade.sh`. Only `EnsembleSignal` is imported by the paper package (`paper/overlay.py:13`, `paper/signal_runner.py:11`). `paper/feature_gate.py` (LightGBM gate) is loaded by `paper/signal_runner.py:111-114` and is OFF unless `ML_GATE_ENABLED` is truthy (`feature_gate.py:22-26`).
- `lab/strategies/leveraged_rotation.py:17` imports `EnsembleSignal` for breadth, so the leveraged rotation rows depend on the ensemble code too (but not on the paper path).
- `ensemble_ic`, `ensemble_kelly`, `leveraged_rotation_*` import from `ensemble.py` (ensemble_ic.py:9; ensemble_kelly.py:14; leveraged_rotation.py:17).

## 2. Grid-size details (sweep_params and `_is_valid_combo`)

- `build_grid` (`lab/sweep.py:64-74`) takes the Cartesian product of `sweep_params()` and filters with `_is_valid_combo` (`lab/sweep.py:38-61`). The fast/slow check zips sorted keys containing "fast" with sorted keys containing "slow".
- Only `ema_cross` loses combos to that filter (20 raw, 16 valid, from the 4 pairs with fast ≥ slow). Every other strategy has raw = valid.
- Total raw grid points across the 38 registry entries: sum of the raw column in §1. (The sum is computed in `s4_grid_count.py`; not re-added by hand.)
- `dual_momentum` inherits the `xs_momentum` grid (12 points).
- `ensemble_ic` sweeps only the IC axes (consensus_threshold 5 x ic_lookback_months 3 = 15); voter params are pinned at baseline (ensemble_ic.py:98-101).
- `ensemble_kelly` sweeps only `kelly_multiplier` (3). Ensemble params are pinned.

## 3. The deployed `ensemble`: voter section

Source: `src/ggTrader/lab/strategies/ensemble.py`.

Voter registry (`ALL_VOTERS`, ensemble.py:31) and default sets:
- `ALL_VOTERS = ("bb", "rsi", "ema", "macd", "vbb", "mtf")` (:31)
- `FIVE_VOTERS = ("bb", "rsi", "ema", "macd", "vbb")` (:37)
- `DEFAULT_VOTERS = FIVE_VOTERS` (:39). This is the default for `EnsembleSignal` (`voters=` default at :90) and for live.
- `THREE_VOTERS = ("bb", "rsi", "ema")` (:42). Not the default.
- `mtf` is valid but not in the default set (per the file's own comment, :32-36).

Default constructor values (`EnsembleSignal.__init__`, ensemble.py:67-91):

| Param | Default | Note |
|---|---|---|
| `min_agree` | 2 | entry votes needed |
| `min_agree_exit` | `None` → `min_agree` (= 2) | line `self.min_agree_exit = ... else min_agree` |
| `bb_period` / `bb_std` | 20 / 2.0 | shared by `bb` and `vbb` |
| `rsi_period` / `rsi_oversold` / `rsi_exit` | 14 / 30 / 50 | |
| `ema_fast` / `ema_slow` | 20 / 50 | |
| `macd_fast` / `macd_slow` / `macd_signal` | 12 / 26 / 9 | |
| `divergence_window` | 20 | `macd` only |
| `vol_period` / `vol_mult` | 20 / 2.0 | `vbb` only |
| `weekly_rsi_period` / `_oversold` / `_exit` | 14 / 30 / 50 | `mtf` only (not active by default) |
| `td_stop` | `None` | no time stop by default (`_apply_time_stop`, :211) |
| `exits_enabled` | `True` | |
| `voters` | `DEFAULT_VOTERS` | 5 voters |

Live construction: `paper/signal_runner.py:84-86` uses `LabConfig(min_history_bars=60)` and `EnsembleSignal(cfg)` with no overrides, so the live values equal the defaults above. `paper/overlay.py:55` does the same (its docstring says "identical"). `LabConfig.max_sector_count` defaults to `None` (`lab/strategy.py:30`), so no sector cap applies on the live path.

Voter definitions (each is a boolean entry frame and a boolean exit frame, time x symbol; from `indicators.py`):

| Voter key | Function (indicators.py) | Entry | Exit (votes) |
|---|---|---|---|
| `bb` | `bb_signals` :67 | close crosses below SMA − 2·std (prev close ≥ lower; now close < lower) | close crosses above SMA (prev close < SMA; now ≥ SMA) |
| `rsi` | `rsi_signals` :118 | RSI(14, Wilder-style ewm alpha=1/14) crosses below 30 | RSI crosses above 50. Also **independent** exit (see below) |
| `ema` | `ema_signals` :141 | EMA20 crosses above EMA50 (ewm span, adjust=False) | EMA20 crosses below EMA50 |
| `macd` | `macd_signals` :182 | close ≤ rolling min(close, 20) AND histogram > rolling min(histogram, 20). Entries masked False for the first 26+9+20 = 55 bars (`warmup = slow + signal + divergence_window`) | histogram crosses below 0 |
| `vbb` | `volume_bb_signals` :237 | `bb` entry AND volume > 2.0 · SMA20(volume) | same formula as `bb` exit (prev close < SMA; now ≥ SMA) |
| `mtf` | `mtf_signals` :294 | weekly RSI(14) < 30 AND daily close < BB(20,2) lower | weekly RSI crosses ≥ 50 OR daily close crosses above SMA. Only if `mtf` is in `voters` |

How votes combine (`_generate_signals`, ensemble.py:153-209):
1. `entry_votes = Σ_voters entry_frame.astype(int)` (:195). `entries = entry_votes >= min_agree` (:198). Then `entries &= entry_mask` (PIT eligibility, `eligibility_mask` in `indicators.py:30`), applied before the time stop (:199-200).
2. `exit_votes = Σ_voters exit_frame.astype(int)` (:196) over all active voters, including `rsi`.
3. If `exits_enabled`: `exits = ext["rsi"] (if rsi active, else False) | (exit_votes >= min_agree_exit)` (:201-204). The RSI exit alone closes a position. Other exits need `min_agree_exit` votes.
4. `_apply_time_stop` (:211): if `td_stop` is set, exit `td_stop` bars after each entry. Not set by default.
5. `to_targets` (:223) builds the masks and calls `_generate_signals`.

Facts that follow from the code (no recommendation implied):
- `vbb` entry is `bb` entry AND a volume spike (indicators.py:237-263), with the same `bb_period`/`bb_std`. So every `vbb` entry is also a `bb` entry, and one bar with a volume spike counts as **two** entry votes, which meets the default `min_agree = 2` on its own.
- `vbb` exit uses the same formula as `bb` exit, so one close-above-SMA event contributes two exit votes.
- `ensemble` entries and exits are computed bar-by-bar on closes; there is no ATR or trailing stop in the signal path. `ATR`/`ts_stop` exist only in the simulator overlay (`lab/simulate.py:123-150`, `:203-246`) and are not in `STOCK_BASE_CONFIG` (`lab/data.py:18-33`, no stop keys). The ensemble.py comment that mentions "ATR exits" (:33-36) therefore describes the simulator overlay, not the signal function. Whether the live path applies one is outside this inventory.
- Sweep axes (ensemble.py:120-137): min_agree [2,3,4], min_agree_exit [1,2], bb_std [2.0,2.5], rsi_oversold [25,30], ema_fast [10,20]; all others pinned. 3·2·2·2·2 = 48 valid combos.

## 4. Indicator functions used by the deployed live path

Live path = `paper/signal_runner.py` and `paper/overlay.py`, which import `EnsembleSignal` with default voters. Functions reached by the default 5-voter set:

| Function (indicators.py) | Line | Used by live path via |
|---|---|---|
| `extract_close` | :12 | `ensemble.py:225` (to_targets) |
| `extract_volume` | :231 | `ensemble.py:226` |
| `eligible_symbols` | :24 | `ensemble.py:145` (`select`) |
| `eligibility_mask` | :30 | `ensemble.py:230` (to_targets) |
| `bb_signals` | :67 | `bb` and `vbb` voters |
| `rsi_signals` | :118 | `rsi` voter (also the independent RSI exit) |
| `ema_signals` | :141 | `ema` voter |
| `macd_signals` | :182 | `macd` voter |
| `volume_bb_signals` | :237 | `vbb` voter |

Imported on the live module but not called on the default voter set: `mtf_signals` (:294), and the strength helpers `bb_strength` (:152), `rsi_strength` (:161), `ema_strength` (:174), `macd_strength` (:215), `volume_bb_strength` (:266), `mtf_strength` (:327). These are used only by `EnsembleConvictionSignal` (ensemble.py:~367-440). `_weekly_rsi` (:281) is internal to `mtf`.

Not on the live path: `overnight_gap_signals` (:84), `extract_open` (:18), the `*_raw` functions (:342-380), which are used only by `ensemble_ic`.

Also on the live overlay path (`paper/overlay.py:67`): `lab/simulate.simulate_signals`, which calls vectorbt `Portfolio.from_signals` (`lab/simulate.py:277-292`). That is simulation, not a hand-written indicator.

## 5. Distinct hand-written indicator functions and their users

Set of functions in `lab/strategies/indicators.py` (24 defs), with the registry strategies that call each. "Direct" means the strategy module calls it. "Via ensemble" means the strategy calls `EnsembleSignal`, which calls it.

| Function (indicators.py) | Computes | Strategies using it |
|---|---|---|
| `extract_close` :12 | (time x symbol) close frame | ensemble, ensemble_conviction, ensemble_ic, ensemble_kelly, conviction_bb, ema_cross, wfo_tournament, bb_reversion, rsi_reversion, macd_divergence, volume_bb_reversion (via their modules), mtf_reversion, overnight_gap, fomc_drift, month_end_treasury, treasury_curve, cross_asset_sleeve, fx_hedge_overlay, commodity_trend, idio_vol, max_effect, pairs_stat_arb |
| `extract_open` :18 | open frame | overnight_gap |
| `eligible_symbols` :24 | filter by history | ensemble, ensemble_conviction, ensemble_ic, ensemble_kelly, conviction_bb, ema_cross, wfo_tournament, bb_reversion, rsi_reversion, macd_divergence, volume_bb_reversion, mtf_reversion, overnight_gap, idio_vol, max_effect, commodity_trend, pead, short_interest, short_volume_ratio, retail_attention, headline_sentiment, insider_cluster_buy, congress_trades |
| `eligibility_mask` :30 | PIT mask from plans | ensemble, ensemble_conviction, ensemble_ic, ensemble_kelly, ema_cross, bb_reversion, rsi_reversion, macd_divergence, volume_bb_reversion, mtf_reversion, overnight_gap |
| `bb_signals` :67 | BB entry/exit | ensemble (bb, vbb via `bb` calls), ensemble_ic, conviction_bb, bb_reversion; `volume_bb_signals` inlines the same BB entry |
| `overnight_gap_signals` :84 | gap z-score entry/exit | overnight_gap |
| `rsi_signals` :118 | RSI entry/exit | ensemble, ensemble_ic, rsi_reversion |
| `ema_signals` :141 | EMA cross | ensemble, ensemble_ic, ema_cross, wfo_tournament |
| `bb_strength` :152 | normalized BB depth | ensemble_conviction (also used by `volume_bb_strength`) |
| `rsi_strength` :161 | normalized RSI depth | ensemble_conviction |
| `ema_strength` :174 | normalized EMA gap | ensemble_conviction |
| `macd_signals` :182 | MACD divergence entry, histogram-cross exit | ensemble, ensemble_ic, macd_divergence |
| `macd_strength` :215 | normalized histogram magnitude | ensemble_conviction |
| `extract_volume` :231 | (time x symbol) volume frame | ensemble, ensemble_conviction, ensemble_ic, ensemble_kelly, volume_bb_reversion |
| `volume_bb_signals` :237 | BB entry AND volume spike | ensemble, ensemble_ic, ensemble_conviction, volume_bb_reversion |
| `volume_bb_strength` :266 | BB depth × volume ratio | ensemble_conviction |
| `_weekly_rsi` :281 | weekly RSI, ffill to daily | mtf_signals, mtf_strength (internal) |
| `mtf_signals` :294 | weekly RSI + daily BB | ensemble (only if `mtf` voted), ensemble_conviction (same), mtf_reversion |
| `mtf_strength` :327 | average weekly RSI and BB depth | ensemble_conviction |
| `rsi_raw` :342 | negated RSI (score) | ensemble_ic |
| `bb_raw` :354 | negated %b (score) | ensemble_ic |
| `ema_raw` :364 | signed EMA gap (score) | ensemble_ic |
| `macd_raw` :371 | MACD histogram (score) | ensemble_ic |
| `vbb_raw` :380 | %b × volume ratio (score) | ensemble_ic |

Hand-written rolling or statistical helpers **outside** `indicators.py`:

| Helper (file:line) | Computes | Strategy |
|---|---|---|
| `ic_weights.forward_returns` :14, `daily_cross_sectional_ic` :19, `ic_weight_schedule` :42 | forward return; Spearman rank IC per date; trailing-window weights | ensemble_ic |
| `kelly.extract_trades` :54, `expanding_kelly_fraction` :84, `kelly_sizes` :129 | trade extraction; expanding Kelly fraction; size array | ensemble_kelly |
| `conviction._compute_conviction_sizes` conviction.py:54 | rolling SMA/STD inline; BB-depth size | conviction_bb |
| `ensemble._generate_signals_with_sizes` (ensemble.py:367-440) and its strength averaging | conviction-weighted size | ensemble_conviction |
| `momentum.CrossSectionalMomentum.select` (inline, momentum.py:~26-42) | 12-1 return | xs_momentum, dual_momentum |
| `max_effect.trailing_max_return` :14 | rolling max of daily returns | max_effect |
| `idio_vol.idiosyncratic_variance` :14 | rolling beta and residual variance vs EW market | idio_vol |
| `commodity_trend.cross_sectional_momentum_scores` :53, `regime_vol_zscore` :70 | 12-1 return; rolling vol z-score | commodity_trend |
| `cross_asset_sleeve.leg_is_on` :35 | trailing-return sign | cross_asset_sleeve |
| `treasury_curve.curve_slope_zscore` :57, `curve_regime_weights` :70 | rolling z of yield slope; one-hot regime | treasury_curve |
| `fx_hedge_overlay.rolling_zscore` :76, `real_fx_index` :82, `trend_signal` :94, `unhedged_weight` :108 | rolling z; CPI-adjusted FX; log ratio momentum; tanh squash | fx_hedge_overlay |
| `leveraged_rotation.compute_breadth` :21, `rotate_positions` :32 | breadth fraction; hysteresis state machine | leveraged_rotation_* (3); `rotate_positions` also used by leveraged_trend_* (3) |
| `leveraged_trend._trend_state` :46 (uses `rotate_positions`), inline rolling mean in `sweep_signals` | SMA trend state with min hold | leveraged_trend_* (3) |
| `pairs_stat_arb.enumerate_sector_pairs` :24, `pairwise_correlations` :42, `spread_zscore` :63, `pair_signal` :77 (unused in the module) | sector pairs; correlation; log-spread z | pairs_stat_arb |
| `insider_cluster.cluster_events` :25 | cluster detection loop | insider_cluster_buy |
| `index_deletion.deletion_events` :36 | snapshot diff | index_deletion_fade |
| `fomc_drift._event_signals` :117; `month_end_treasury.month_end_signals` :77 | calendar event masks | fomc_drift; month_end_treasury |
| `pead._most_recent_surprise` :82; `short_interest._days_to_cover_trend` :76; `short_volume_ratio._avg_ratios`; `retail_attention._spikes`; `headline_sentiment` groupby mean | per-symbol event statistics | pead; short_interest; short_volume_ratio; retail_attention; headline_sentiment |
| `congress_trades` PIT filter inside `select` | filing-date gate and hold window | congress_trades |
| `signals._vectorized_ema_tournament` signals.py:35-72 | 4-combo in-sample tournament via `vbt.Portfolio.from_signals` | wfo_tournament |

Counts of distinct hand-written functions: 24 in `indicators.py`; about 34 further helpers in strategy modules and `ic_weights.py` / `kelly.py` (counted from the table above; approximate, since `short_interest`, `short_volume_ratio`, `retail_attention`, `pead` and `headline_sentiment` are inline per-strategy code rather than named functions).

## 6. Counts

Primary class (D where the snapshot verdict is any NO-GO, otherwise the technical class), over 38 entries:

| Class | Count | Members |
|---|---|---|
| A | 5 | ema_cross, bb_reversion, rsi_reversion, volume_bb_reversion, conviction_bb |
| B | 4 | macd_divergence, ensemble, retail_attention, cross_asset_sleeve |
| C | 0 | (every C-class strategy is also NO-GO, so it is D as primary) |
| D | 29 | wfo_tournament, mtf_reversion, overnight_gap, ensemble_ic, ensemble_kelly, congress_trades, ensemble_conviction, xs_momentum, dual_momentum, idio_vol, index_deletion_fade, insider_cluster_buy, max_effect, pairs_stat_arb, pead, short_interest, short_volume_ratio, fx_hedge_overlay, headline_sentiment, fomc_drift, month_end_treasury, commodity_trend, treasury_curve, leveraged_rotation_sp500, leveraged_rotation_nasdaq100, leveraged_rotation_russell2000, leveraged_trend_sp500, leveraged_trend_nasdaq100, leveraged_trend_russell2000 |
| **Total** | **38** | |

Technical class ignoring verdict (all 38):

| Class | Count |
|---|---|
| A | 5 (ema_cross, bb_reversion, rsi_reversion, volume_bb_reversion, conviction_bb) |
| B | 23 |
| C | 10 (ensemble_ic, ensemble_kelly, idio_vol, insider_cluster_buy, pairs_stat_arb, fx_hedge_overlay, headline_sentiment, leveraged_rotation_sp500, leveraged_rotation_nasdaq100, leveraged_rotation_russell2000) |
| **Total** | **38** |

Other counts:
- Registry entries with a live-path dependency: 1 (`ensemble`, which is `DEPLOYED` in code and whose snapshot verdict is "Daily-bar construction NO-GO; live unverified").
- Entries that read a non-OHLCV source file or API: 11 (congress_trades, insider_cluster_buy, index_deletion_fade, pead, retail_attention, short_interest, short_volume_ratio, fx_hedge_overlay, headline_sentiment, fomc_drift, treasury_curve). Two more read a reference file: pairs_stat_arb (sector map) and the live ensemble (PIT membership in `paper/signal_runner.py`). `month_end_treasury` is calendar-only and is not counted.
- Strategies with no indicator at all (event or calendar only): congress_trades, index_deletion_fade, fomc_drift, month_end_treasury (4).
- Registry names that are voter components of the deployed ensemble, not standalone verdicts: 5 (ema_cross, bb_reversion, rsi_reversion, macd_divergence, volume_bb_reversion).

## 7. Caveats and unverified items

- vectorbt 1.2.0 API names and exact numeric equivalence for crossing accessors, `MAX`, cov/beta, and the EWM RSI/MACD variants are **UNVERIFIED** in this pass. The equivalence facts cited in §0 come from the earlier note `scratchpad/vbt-research/B-vectorbt-1x-and-rust.md` (rows 117-122), not from a fresh run.
- The report verdicts were quoted from `RESEARCH_SNAPSHOT.md`. The linked research reports were not opened.
- Grid counts used the repo `.venv` python with `PYTHONPATH=src`. They reflect `sweep_params()` plus `_is_valid_combo`, not the WFO overlay axes (`ts_stop`, `atr_mult`, `tp_stop`, `vol_target` in `lab/sweep.py:13-14`), which are applied outside the class grid.
- Line numbers were taken from `grep -n` and the file reads above. A few helper line numbers in §5 are approximate and are marked `~`.
- Nothing here recommends retiring the paper trader, the stock-picking sleeve, or any registry entry.
