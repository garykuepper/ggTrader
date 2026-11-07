# C: ggTrader hand-rolled inventory (lab + paper)

Scope: `src/ggTrader/lab/` (incl. `strategies/`) and `src/ggTrader/paper/`, plus the vectorbt call sites in `scripts/`. Facts only. No replacement recommendations.

Method: every module listed below was opened or grepped in this session. Line numbers are of the working tree on branch `research/visual-reports` as read (uncommitted changes exist: `research/audits/root-cause-2026-11-04/2026-11-05-executive-summary.md` modified, `data/external/event_timing_audit/` untracked; neither touches the modules inventoried). Items marked UNVERIFIED were not opened line by line.

---

## 0. Counts

| Quantity | Value | How counted |
|---|---|---|
| Lab lines | 18,591 | `wc -l` over `lab/*.py` + `lab/strategies/*.py` (includes 3,346-line `event_timing_audit.py`, 907-line `event_timing_sources.py`, 492-line `event_timing_reconstruction.py`, 145-line `event_acceptance_diagnostic.py`, which are evidence-collection tooling, not strategy code) |
| Paper lines | 3,236 | `wc -l` over `paper/*.py` |
| Total | 21,827 | sum of the above |
| Hand-rolled primitive lines | approx. 3,999 (lab 3,482 + paper 517) | AST line spans (`end_lineno - lineno + 1`) of the ~150 functions and classes listed in sections 2 and 3 as primitives. Approximate because some spans are accounting or guardrail code rather than textbook primitives: `TaxLotBook` 378 lines, `a23_rebalance.Book` ~130, `RiskGuard` ~120, `replica.run_replica` 195. The measuring script is `scratchpad/vbt-research/count_primitives.py`. Ratio: about 18% of all src lines in the scope. |

---

## 1. vectorbt call inventory (complete for `src/` and `scripts/`)

Only four `Portfolio` constructors appear in `src/`. Everything else (metrics, WFO, gates, Kelly, paper trader) never calls vectorbt.

| Location | Call | Notes |
|---|---|---|
| `lab/simulate.py:9` | `import vectorbt as vbt` | module-level import |
| `lab/simulate.py:61-72` | `vbt.Portfolio.from_orders(close, size, size_type="targetpercent", init_cash, fees, slippage, freq, cash_sharing=True, group_by, call_seq="auto").copy()` | weight strategies, one grouped call. Reached from `harness.py:107`, `wfo.py:397`, `sweep.py:322` |
| `lab/simulate.py:74` | `pf.value()` | equity extraction |
| `lab/simulate.py:97` | `pf.trades.count()` | per-group closed-trade count, feeds `diags["n_trades"]` used by the WFO expectancy grid (`wfo.py:580`) |
| `lab/simulate.py:277-290` | `vbt.Portfolio.from_signals(close, entries, exits, size, size_type="percent", init_cash, fees, slippage, freq, cash_sharing=True, group_by, **stop_kwargs).copy()` | `stop_kwargs` can carry `sl_stop`, `sl_trail=True`, `tp_stop` (set at `simulate.py:202-246`). Reached from `harness.py:114`, `sweep.py:179`, `paper/overlay.py:67` |
| `lab/simulate.py:292` | `pf.value()` | equity extraction |
| `lab/strategies/signals.py:37` | `import vectorbt as vbt` (inside `_vectorized_ema_tournament`) | |
| `lab/strategies/signals.py:57-68` | `vbt.Portfolio.from_signals(close, entries, exits, size=1.0/n_syms, size_type="percent", init_cash=10000.0, fees=0.0, freq="1d", group_by, cash_sharing=True)` | EMA-combo tournament inside `WfoTournamentSignal.select` (`signals.py:172-207`). `init_cash=10000.0` is hard-coded, not read from config |
| `lab/strategies/signals.py:69` | `pf.sharpe_ratio()` | the only vbt metric accessor used in `src/`. Result picks the best EMA pair |
| `lab/strategies/signals.py:75-76` | bare `except Exception` returns `-inf` for every combo | any vbt failure inside the tournament silently yields a flat tie |
| `scripts/position_sizing_wfo.py:118, 149` | `vbt.Portfolio.from_orders(...)` | research driver |
| `scripts/position_sizing_regimes.py:45, 157` | `vbt.Portfolio.from_orders(...)`; `:175` `deployment_stats(pf)` | research driver; docstring lines 3-9 describe a from_signals variant |
| `scripts/leveraged_trend_research.py:12, 43` | `vbt.Portfolio.from_holding(...)` | research driver |

Docstring-only mentions (not calls): `harness.py:51-52`, `strategy.py:36-37`, `ensemble.py:211-217` (states "vbt 0.28.5 has no native time stop"; UNVERIFIED against the installed package, not checked).

Not vectorbt, but relevant: `paper/` contains no vectorbt import. The live overlay (`paper/overlay.py:12, 67-69`) reaches vbt indirectly through `lab/simulate.simulate_signals`.

---

## 2. Lab modules

Libraries are pandas and numpy unless noted. "vbt" means a direct vectorbt call in that file.

### 2.1 Core pipeline

#### `lab/simulate.py` (304 lines)
Purpose: vbt wrapper (grouped from_orders / from_signals), vol-target scalar, ATR trailing-stop fraction.
Key API: `simulate_weights` 12-84; `_add_trade_counts` 87-101; `compute_vol_scalar` 104-120; `compute_atr_stop` 123-150; `simulate_signals` 153-304.
vbt: lines 9, 61, 74, 97, 277, 292 (section 1).

| Item | Lines | Behaviour (what the code does) |
|---|---|---|
| `compute_vol_scalar` | 104-120 | equal-weight mean of `pct_change` across symbols; realized = rolling std (pandas ddof=1) x sqrt(252); scalar = clip(target/realized shifted 1 bar, 0.1, vol_cap), NaN filled with 1.0; `vol_cap` default 2.0 |
| `compute_atr_stop` | 123-150 | per symbol TR = max(H-L, abs(H-prevC), abs(L-prevC)); ATR = **simple** rolling mean of TR (`:149`), not Wilder; stop fraction = atr_mult x ATR / close, floored at 0.001 |
| `simulate_signals` stop block | 202-246 | `ts_stop` -> `sl_stop` + `sl_trail=True`; `atr_mult` -> per-bar `sl_stop` frame, ffill, fillna(inf); `tp_stop` passed through |
| `simulate_signals` sizing | 248-275 | `size = base_size x vol_scalar` per bar (`:249-263`); base_size default 0.02 (`:249`); conviction `sizes` override via `DataFrame.update` (`:265-275`) |
| `simulate_signals` returns | 296 | `value.pct_change(fill_method=None).fillna(0.0)` (returns are derived from the value curve) |
| `simulate_weights` returns | 82 | same derivation; empty-universe short-circuit to flat cash `:42-47`, flat fill for empty strategies `:78-80` |

PROJECT-SPECIFIC:
- `simulate.py:35-47` comment: an event strategy with zero columns is short-circuited to flat cash because vbt "degenerates" (quoted from code comment; not re-verified).
- `group_by=strategy` with `cash_sharing=True` (`:69-70`, `:287-288`).
- Default `base_size` 0.02 (`:249`) differs from `STOCK_BASE_CONFIG["SIGNAL_POSITION_SIZE"]` 0.03 (`data.py:29`).

#### `lab/metrics.py` (56 lines)
Purpose: curve statistics and SPY benchmark.
Key API: `curve_stats` 13-38; `benchmark` 41-56. vbt: none.

| Item | Lines | Behaviour |
|---|---|---|
| total return | 17 | last/first - 1 |
| CAGR | 16, 33 | years = calendar days / 365.25 (`:16`); ((1+total)^(1/years)-1) |
| Sharpe | 19-23 | mean(r) / std(r) x sqrt(252), std ddof=1; NaN if std == 0; **no risk-free subtraction** |
| Sortino | 24-29 | mean(r) / std(r where r<0) x sqrt(252); **downside std uses only the negative returns**, not a semi-deviation over all returns |
| annual vol | 18 | std x sqrt(252) |
| max drawdown | 30 | min(curve / cummax - 1) |
| monthly hit rate vs SPY | 46-49 | resample("ME").last(), pct_change, fraction of months strategy > SPY |
| benchmark curve | 42-43 | SPY rescaled to START_CASH |

Not present (grep): Calmar, win rate as a metric (only the monthly hit rate; Kelly has its own internal win rate at `kelly.py:102`).

#### `lab/harness.py` (148 lines)
Purpose: plan -> simulate -> persist -> score for non-WFO runs.
Key API: `leak_check` 19-34; `walkforward` 37-148. vbt: none directly (calls `simulate_weights` `:107`, `simulate_signals` `:114`).

| Item | Lines | Behaviour |
|---|---|---|
| `leak_check` | 19-34 | calls `strategy.select(asof, ...)` three ways (truncated, deep copy, unmasked frame), compares JSON dumps |
| plan loop | 76-99 | per-strategy, per-rebalance `select`, resumable via `persist.plan_done` |
| warm-up trim | 120-125 | scoring starts at the first bar after the first rebalance |
| benchmark | 131-133 | `start_cash * spy / spy[0]` |

PROJECT-SPECIFIC: `leak_check` (lookahead guard); resumable plan persistence; warm-up trim.

#### `lab/wfo.py` (1,076 lines) — the largest hand-rolled block
Purpose: walk-forward optimisation, gates, circuit breaker, anchor fallback, live-param selection, table rendering.
Key API: `compute_wfe` 44; `WfoPolicy` 59; `WfoState` 67; `check_circuit_breaker` 79; `check_shadow_reentry` 128; `AnchorSet` 167; `Fold` 177; `WfoResult` 184; `generate_folds` 194; `_min_max_normalize` 213; `composite_score` 222; `_sweep_fold` 251; `_run_one_weight_combo` 311; `_sweep_fold_weights` 354; `_sweep_fold_dispatch` 415; `compute_anchor_set` 465; `_extract_grid_arrays` 524; `_compute_dsr_inputs` 590; `run_wfo` 605; `_pick_live_winner` 894; `select_live_params` 922; `format_wfo_table` 979.
Libraries: numpy, pandas, joblib (`:11`, `Parallel(n_jobs=-1)` at `:388`). vbt: none directly; reaches vbt through `simulate_weights` (`:397`) and `simulate_signals` via `sweep.py`.

| Item | Lines | Behaviour |
|---|---|---|
| WFE | 44-56 | OOS Sharpe / IS Sharpe; returns None when IS Sharpe < 0.4 (`is_floor`, `:47`) |
| chronic-decay breaker | 99-113 | mean of trailing 4 non-None WFE < 0.25 -> halt (window 4 and floor 0.25 are defaults at `:81-82`) |
| acute breaker | 115-123 | last two OOS Sharpes both < 0 -> halt |
| shadow re-entry | 155-164 | clean window = NDH pass AND DSR pass AND WFE >= 0.5; 2 of last 3 clean -> resume |
| fold generation | 194-210 | fixed-width rolling: train 12 months, test 3 months, step 3 months, requires test_end <= eval_end |
| min-max normalise | 213-219 | (v - min)/(max - min); all zeros if max == min |
| composite score | 222-248 | 0.5 x norm(Sharpe) + 0.3 x norm(Sortino) - 0.2 x norm(abs(MaxDD)), normalised within the candidate set; NaN replaced by the set minimum |
| weight-path fold sweep | 354-412 | joblib parallel per combo, `simulate_weights` at `:397`, `curve_stats` at `:409` |
| anchor set | 465-521 | min-drawdown combo among those with CAGR > `risk_free_rate=4.0` (`:472`, a percent figure); falls back to all combos if none qualify (`:509-510`); `max(...)` on drawdown (`:513`) |
| expectancy grid | 573-580 | expectancy = (total_return_pct/100) / max(1, n_trades) |
| DSR inputs | 590-602 | pandas `skew()` and `kurtosis()` (excess) of pct_change returns; non-finite set to 0 |
| main loop | 648-833 | per fold: anchor fit on data <= train_end (`:656-665`); winner = argmax composite (`:695-697`); NDH (`:706-727`); DSR with `n_trials=len(grid)` (`:732-743`); fallback to anchor when gates fail or halted (`:751-753`); test sweep (`:757-769`); normalised OOS splice (`:777-779`) |
| OOS aggregate | 836-849 | `curve_stats` on the spliced OOS curve and on SPY over the same index |
| live params | 922-976 | train on last 12 months ending at eval_end; `_pick_live_winner` prefers combos that won >= `MIN_LIVE_STABILITY` folds |
| `_pick_live_winner` | 894-919 | durable pool first, else all combos, argmax composite |
| table | 979-1076 | string rendering only |

PROJECT-SPECIFIC (all of it): `REGIME_PARAMS` (`:31`) excluded from NDH neighbourhood; `LAB_CONFIG_COMBO_KEYS` (`:38`); `MIN_LIVE_STABILITY = 1` (`:27`); PIT entry mask built once (`:631-637`); the anchor is fit on the expanding window per fold (`:651-665`, deliberate leak fix); circuit breaker, shadow re-entry, anchor fallback; `WfoPolicy` options (`:59-64`); the composite weights (0.5/0.3/0.2); expectancy definition; SPY comparison built inside the WFO.

#### `lab/gates.py` (215 lines)
Purpose: robustness gates (NDH, DSR).
Libraries: numpy, scipy.stats.norm (`:9`). vbt: none.

| Item | Lines | Behaviour |
|---|---|---|
| neighbour indices | 22-54 | von Neumann +/-1 step along each allowed axis (`:46-53`) |
| NDH | 57-119 | density = count(neighbour Sharpe > 0 AND expectancy > 0) / n (`:92-93`); variance_ratio = np.std(neighbour Sharpes) / peak Sharpe (`:105`); passes if density >= 0.85 AND (density == 1.0 OR variance_ratio <= 0.20) (`:111-112`) |
| SR standard error | 134-144 | sqrt((1 - g3 x SR + (kurt_excess+2)/4 x SR^2) / (T-1)); Bailey and Lopez de Prado form (`:142`) |
| PSR | 147-164 | Phi((SR - SR*)/SE) |
| expected max SR | 167-187 | SE(0) x ((1-gamma) Phi^-1(1-1/N) + gamma Phi^-1(1-1/(N e))), gamma = Euler-Mascheroni |
| DSR | 190-215 | annual SR / sqrt(252) to per-period (`:207`); PSR vs E[max SR]; passes if >= 0.80 (`:210`) |

PROJECT-SPECIFIC: the plateau exemption (`:107-111`, variance cap waived when density == 1.0, commented as a deliberate 2026-06-26 change); thresholds 0.85 / 0.20 / 0.80 as defaults.

#### `lab/sweep.py` (350 lines)
Purpose: grid construction, validity filters, stop-config batching, sweep orchestration.
Key API: `build_grid` 64; `_is_valid_combo` 38; `combo_name` 77; `group_by_stop_config` 88; `pit_entry_mask` 101; `sweep_signal_group` 130; `format_results_table` 184; `run_sweep` 223. vbt: none directly (via `simulate_*`, `:179`, `:322`).

| Item | Lines | Behaviour |
|---|---|---|
| grid | 64-74 | `itertools.product` over sorted keys, then filter |
| validity | 38-61 | rejects ts_stop + atr_mult together (`:44-45`); exit_z >= entry_z (`:46-47`); exits_enabled False with no td/tp stop (`:50-54`); **fast < slow by key-name match** ("fast"/"slow" substrings, `:56-60`) |
| PIT entry mask | 101-127 | `universe_fn(d, data<=d)` per bar, boolean frame; one call per bar (comment `:118-120`, about 5 min per 6-year run) |
| masked entries | 160-169 | entries AND mask; exits left unmasked |
| sorting | 199 | results sorted by Sharpe descending (NaN as -inf) |

PROJECT-SPECIFIC: PIT mask; the stop-config grouping exists so each group is one vbt call (`:278-289` in `run_sweep`).

#### `lab/data.py` (314 lines)
Purpose: OHLCV loading, universe, rebalance calendar, eligibility.
Key API: `STOCK_BASE_CONFIG` 18; `CRYPTO_BASE_CONFIG` 39; `fetch_stock_ohlcv` 57; `collapse_daily_duplicates` 187; `load_ohlcv` 237; `equity_universe_between` 255; `rebalance_dates` 266; `eligible_at` 297. vbt: none.

| Item | Lines | Behaviour |
|---|---|---|
| fetch cascade | 98-153 | DB cache (CachedYFinanceLoader) -> yfinance -> Tiingo for still-missing symbols; gap fetches persisted back to DB |
| coverage filter | 173-179 | keep symbols whose close non-NaN fraction >= min_coverage (default 0.0, so off) |
| date trim | 169-171 | inclusive start/end filter |
| `collapse_daily_duplicates` | 218-234 | keeps the modal time-of-day convention by cross-sectional coverage (`:232-234`) |
| `rebalance_dates` | 285-294 | last trading day of each calendar month except the final month (`:289-291`), plus the window's first bar if it is not already a month-end (`:292-293`) |
| `eligible_at` | 303-314 | members with >= `min_history_bars` non-NaN closes in `past`; optional `max_stocks` by sorted name |

PROJECT-SPECIFIC: `collapse_daily_duplicates` (documents a one-day timestamp shift and SPY duplicate rows in the DB, citing an audit); negative-cache opt-in; `min_history_bars=400` default (`strategy.py:28`); universe membership from `index_constituents` (`universe_members_asof`, `universe_all_between`).

#### `lab/strategies/indicators.py` (391 lines)
Purpose: shared vectorised indicators and PIT eligibility mask.
Libraries: numpy, pandas. vbt: none. Stochastic, Keltner, OBV, Donchian, VWAP, Calmar: not present (grep of `lab/` and `paper/`). Wilder ATR: not present (ATR exists only in `simulate.py:123-150`, as a simple mean).

| Function | Lines | Formula / behaviour |
|---|---|---|
| `extract_close`, `extract_open`, `extract_volume` | 12-21, 231-234 | column pivot of multi-level OHLCV |
| `eligible_symbols` | 24-27 | present and >= min_history_bars closes |
| `eligibility_mask` | 30-64 | PIT: symbol eligible from first plan date it appears, forward-filled to next rebalance date, never before first plan |
| `bb_signals` | 67-81 | SMA(period), rolling std ddof=1, lower = SMA - k x std; entry on cross below lower; exit on cross above SMA |
| `overnight_gap_signals` | 84-115 | gap = (open - prevClose)/prevClose; rolling z of gap; entry on z crossing below gap_z_entry; exit on crossing above gap_z_exit |
| `rsi_signals` | 118-138 | gains/losses, `ewm(alpha=1/p, adjust=False, min_periods=p)` (Wilder-form smoothing), RS = avg_gain/avg_loss (0 -> NaN), RSI = 100 - 100/(1+RS); cross-below oversold entry, cross-above exit level |
| `ema_signals` | 141-149 | `ewm(span, adjust=False)` fast and slow; cross-up entry, cross-down exit |
| `bb_strength` | 152-158 | (SMA - k x std - close) / (k x std), clipped [0,1] |
| `rsi_strength` | 161-171 | (oversold - RSI)/oversold, clipped [0,1] |
| `ema_strength` | 174-179 | (fast - slow)/slow, clipped [0,1] |
| `macd_signals` | 182-212 | MACD = EMA(fast) - EMA(slow); signal = EMA(signal_period) of MACD; hist = MACD - signal; entry = close <= rolling min(close, w) AND hist > rolling min(hist, w) (bullish divergence); warm-up mask `:205-206` (`slow+signal+w` bars); exit = hist crosses below 0 |
| `macd_strength` | 215-228 | abs(hist) / rolling max(abs(hist), 50), clipped; first `slow` bars NaN |
| `volume_bb_signals` | 237-263 | bb entry AND volume > vol_mult x SMA(volume, vol_period); exit as bb exit |
| `volume_bb_strength` | 266-278 | (bb_strength + min(vol/vol_avg, 3)/3)/2, clipped |
| `_weekly_rsi` | 281-291 | `resample("W").last()`, RSI as above, reindex ffill to daily |
| `mtf_signals` | 294-324 | entry = weekly RSI < oversold AND daily close < daily BB lower; exit = weekly RSI crosses up through exit OR daily close crosses above SMA |
| `mtf_strength` | 327-339 | average of weekly RSI depth and BB depth |
| `rsi_raw` | 342-351 | -RSI (higher = more oversold) |
| `bb_raw` | 354-361 | -%b = -(close - lower)/(upper - lower) |
| `ema_raw` | 364-368 | (fast - slow)/slow |
| `macd_raw` | 371-377 | MACD histogram |
| `vbb_raw` | 380-391 | -%b x min(vol/vol_avg, 3) |

PROJECT-SPECIFIC: `eligibility_mask` (PIT entry gating; `:36-45` explains the lookahead problem it fixes).

#### `lab/strategies/signals.py` (754 lines)
Purpose: signal strategies (entries/exits) wrapping the indicators; EMA tournament.
Key API: `_vectorized_ema_tournament` 35; `EmaCrossSignal` 81; `WfoTournamentSignal` 151; `BollingerReversionSignal` 318; `RsiReversionSignal` 381; `OvernightGapReversionSignal` 454; `MACDDivergenceSignal` 532; `VolumeBBReversionSignal` 599; `MultiTimeframeReversionSignal` 669.
vbt: `:37, :57, :69` (section 1).

| Item | Lines | Behaviour |
|---|---|---|
| EMA combo tournament | 35-78 | 4 fixed (fast, slow) pairs `:27-32`; one from_signals call; ranks by `sharpe_ratio()` |
| `EmaCrossSignal.to_targets` | 116-124 | `ema_signals` AND `eligibility_mask` |
| `EmaCrossSignal.sweep_signals` | 126-149 | span-cached ewm; entries = cross up; exits = cross down (inline, not via `ema_signals`) |
| `WfoTournamentSignal.select` | 172-207 | forward-fills closes, takes first `is_fraction` of rows, picks best EMA pair by the tournament, applies sector cap if configured |
| `WfoTournamentSignal.to_targets` | 209-293 | piecewise: each rebalance period uses its own (fast, slow); memoised `pair_cache`; exit on symbols dropped from plan |
| `WfoTournamentSignal.sweep_signals` | 294-315 | per `is_fraction` combo: tournament on IS slice, then `ema_signals` on full close |
| other reversion classes | 318-743 | each is a thin wrapper: `select` = eligible symbols with history; `to_targets` = indicator + `eligibility_mask`; `sweep_signals` = indicator per unique param tuple |

PROJECT-SPECIFIC: the tournament selects on an in-sample fraction of each rebalance window (`:188-189`); the bare `except` at `:75` (see section 1).

#### `lab/strategies/ensemble.py` (489 lines) — the deployed live strategy
Purpose: N-of-M majority vote over sub-signals.
Key API: `ALL_VOTERS` 31; `FIVE_VOTERS` 37; `THREE_VOTERS` 42; `_validate_voters` 45; `EnsembleSignal` 55; `_opt_int` 275; `EnsembleConvictionSignal` 280.
vbt: none (vbt is reached through the caller's simulate).

| Item | Lines | Behaviour |
|---|---|---|
| vote counting | 195-198 | entry_votes = sum of voter entry booleans; entries = votes >= min_agree |
| entry mask | 199-200 | entries AND PIT mask (before time-stop) |
| exits | 201-207 | exit = RSI exit (independent) OR exit_votes >= min_agree_exit; `exits_enabled=False` drops indicator exits |
| time stop | 211-221 | `entries.shift(td_stop)` ORed into exits |
| validated voter set | 31-42 | default FIVE_VOTERS (bb, rsi, ema, macd, vbb); MTF excluded (commented ablation, `:32-37`) |
| `EnsembleConvictionSignal` sizes | 434-441 | conviction = sum(strength where voter fired) / entry_votes (`:435-436`); size = min_size + conviction x (max_size - min_size) (`:438`); defaults 0.01 and 0.04 (`:311-312`); NaN where no entry |

PROJECT-SPECIFIC: voter set chosen by a 2026-06-24 ablation (`:32-37`); RSI exit is independent of the vote (`:203`); time stop implemented by hand (`:211-221`).

#### `lab/strategies/ensemble_ic.py` (205 lines)
Purpose: IC-weighted voting (NO-GO per memory notes; code is non-default).
Key API: `EnsembleICSignal` 35; `_entries_exits_raw` 115; `_apply_time_stop` 142; `_generate_signals` 148.

| Item | Lines | Behaviour |
|---|---|---|
| raw voter scores | 115-140 | bb_raw, rsi_raw, ema_raw, macd_raw, vbb_raw (from indicators) |
| weighted consensus | 148-177 | score = sum_j w_j[t] x entry_j[t]; entries = score >= consensus_threshold - eps (`:166`) |
| time stop, exits | 142-146, 170-177 | same pattern as ensemble |

#### `lab/strategies/ic_weights.py` (87 lines)
Purpose: causal Spearman IC weights for voters.

| Function | Lines | Behaviour |
|---|---|---|
| `forward_returns` | 14-16 | close[t+h]/close[t] - 1 |
| `daily_cross_sectional_ic` | 19-39 | Spearman per date = Pearson of cross-sectional ranks; NaN if fewer than `min_names` valid symbols |
| `ic_weight_schedule` | 42-87 | at each rebalance date t_k, window = trailing `lookback_months`; **last `horizon` bars dropped** (`:66-72`, leak guard); mean IC per voter; weight_j = max(0, IC_j) / sum_k max(0, IC_k); warm-up equal weights; forward-filled |

PROJECT-SPECIFIC: the horizon-drop leak guard (`:66-72`); equal-weight fallback.

#### `lab/kelly.py` (178 lines) and `lab/strategies/ensemble_kelly.py` (183 lines)
Purpose: pooled causal Kelly sizing.
Key API: `_symbol_trades` 15; `extract_trades` 54; `expanding_kelly_fraction` 84; `kelly_fraction_asof` 115; `kelly_sizes` 129; `EnsembleKellySignal` 24 (`ensemble_kelly.py`).

| Item | Lines | Behaviour |
|---|---|---|
| trade pairing | 15-51 | per symbol: entry ignored while open, closes on next exit; unrealised trailing entry dropped; mirrors vbt from_signals semantics (docstring) |
| trade returns | 54-81 | simple return exit/entry - 1 per round trip |
| expanding f* | 84-112 | cumulative W = wins/n; R = avg_win / avg_loss; f* = W - (1-W)/R (`:109`); NaN until n >= min_trades and both wins and losses exist (`:111-112`) |
| sizing | 129-178 | size = min(kelly_multiplier x f*, max_size) where f* > 0, else base_size; f* looked up at each entry time via searchsorted (strictly before) |

PROJECT-SPECIFIC: strictly-before lookup avoids lookahead; the pairing rule is a project choice; `base_size` fallback.

#### `lab/strategies/conviction.py` (94 lines)
`ConvictionBBSignal._compute_conviction_sizes` 54-63: depth = clip((lower - close)/(k x std), 0, 1); size = min + depth x (max - min); NaN where no entry.

#### `lab/strategies/momentum.py` (95 lines)
Purpose: 12-1 cross-sectional momentum (weights).

| Item | Lines | Behaviour |
|---|---|---|
| score | 30-42 | close[-skip-1]/close[-lookback-1] - 1; skip 21 and lookback 252 from `LabConfig` (`strategy.py:26-27`) |
| sector cap | 47-55 | `apply_sector_constraints` (static JSON map) |
| top-N equal weight | 57-60 | weight = 1/len(ranked) |
| `to_targets` | 62-73 | weights set on the bar after rebalance; unselected symbols set to 0 |
| `DualMomentum` | 76-85 | keeps only momentum >= 0; no renormalisation |

#### `lab/strategies/idio_vol.py` (100 lines)
- `idiosyncratic_variance` 14-36: beta_t = rolling cov(r_i, r_m) / rolling var(r_m); residual = r_i - beta x r_m; output rolling var of residual.
- `IdioVolStrategy.select` 65-88: market proxy = equal-weight mean return; ranks residual variance ascending; bottom bucket of size len // quintile; equal weight.
- `to_targets` 89-100: same next-bar pattern as momentum.

#### `lab/strategies/max_effect.py` (79 lines)
- `trailing_max_return` 14-19: rolling max of daily returns over `window` (min_periods = window).
- `MaxEffectStrategy.select` 45-66: ranks by trailing MAX ascending; bottom bucket of size len // quintile; equal weight.

#### `lab/strategies/leveraged_rotation.py` (190 lines)
- `compute_breadth` 21-29: fraction of breadth-universe entries that are True per row, fixed denominator.
- `rotate_positions` 32-87: state in {long, inverse, cash}; raw state from thresholds; a state change takes effect only after `min_hold_months` consecutive raw readings (streak counter, `:58-86`).
- `to_targets` 144-171: breadth sampled at rebalance dates; states -> long/inverse weights on next bar.
- Memoised `_cached_breadth` 90-99 (`:90`).

#### `lab/strategies/leveraged_trend.py` (180 lines)
- `_trend_state` 46-67: SMA(trend_window) on underlying; `raw = close > SMA`; `rotate_positions` with thresholds 0.5 (hysteresis = min_hold_days only).
- `_entries_exits` 70-74: entry on False->True, exit on True->False.

#### `lab/strategies/pairs_stat_arb.py` (269 lines)
Purpose: market-neutral pairs.

| Function | Lines | Behaviour |
|---|---|---|
| `pairwise_correlations` | 42-60 | Pearson of daily log returns over trailing window; requires >= 0.8 x lookback overlap |
| `spread_zscore` | 63-74 | spread = log(A) - log(B); z = (spread - rolling mean)/rolling std (ddof=1) |
| `pair_signal` | 77-110 | +1 when z <= -entry_z, -1 when z >= entry_z, 0 when abs(z) <= exit_z; ffill between thresholds |
| `PairsStatArb.select` | 191-241 | same-sector pairs filtered by correlation; greedy non-overlapping selection by abs(z); hold logic keeps a pair while abs(z) > exit_z or min_hold not elapsed; equal weight 1/max_pairs |

PROJECT-SPECIFIC: process-local pair cache `_pair_candidate_cache` (`:113-`); 1:1 log spread (no hedge ratio); `exit_z >= entry_z` rejected in `sweep.py:46`.

#### `lab/strategies/commodity_trend.py` (149 lines)
- `cross_sectional_momentum_scores` 53-67: close[-skip-1]/close[-lookback-1] - 1 (or last close when skip = 0).
- `regime_vol_zscore` 70-83: average per-symbol rolling std across symbols; z-score of latest vs trailing window; returns 0.0 if history < max(window//3, 5).
- `select` 116-137: if regime vol z > `vol_z_threshold` return no plan (flat); else top-N by momentum, equal weight.

#### `lab/strategies/cross_asset_sleeve.py` (94 lines)
- `leg_is_on` 35-44: trailing lookback total return > 0 (strict); insufficient history counts as off.
- `select` 70-80: core weight = 1 - sleeve_weight; each leg weight = sleeve_weight / n_legs if on, else moved to CASH.
- `to_targets` 82-94: month-end plan takes effect on the next bar.

#### `lab/strategies/treasury_curve.py` (150 lines)
- `curve_slope_zscore` 57-67: z-score of latest slope vs trailing window; 0.0 if history < max(window//5, 20).
- `curve_regime_weights` 70-79: z > steep -> TLT 1.0; z < flat -> SHY 1.0; else IEF 1.0 (one-hot).

#### `lab/strategies/fx_hedge_overlay.py` (222 lines)
- `rolling_zscore` 76-79: (s - rolling mean)/rolling std.
- `real_fx_index` 82-91: spot x (US CPI index / foreign CPI index), indices rebased to first obs.
- `trend_signal` 94-105: log(end/start) over the 12-1 window of the ratio.
- `unhedged_weight` 108-111: 0.5 x (1 + tanh(score/2.25)).
- `_pair_score` 152-185: carry_component = tanh(carry_diff / carry_scale); trend_component = tanh(trend / trend_scale); value_component = tanh(-value_z); sum.

#### `lab/strategies/month_end_treasury.py` (108 lines)
- `month_end_signals` 77-108: entry `days_before_end` bars before last bar of each calendar month, exit on the last bar; drops the final month if the index stops before the last weekday (`:99-100`, weekday check not an exchange calendar, per the code's own note).

#### `lab/strategies/fomc_drift.py` (149 lines)
- `_event_signals` 117-149: entry `entry_offset_days` bars before each event date, exit `hold_days` bars after; date matching on normalised calendar date (`:133-137`).

#### `lab/strategies/index_deletion.py` (129 lines)
- `deletion_events` 36-51: diff of consecutive membership snapshots; a ticker in row i-1 but not i is an event dated at row i.

#### `lab/strategies/insider_cluster.py` (168 lines)
- `cluster_events` 25-78: open-market buys (code P, not 10b5-1); window 14 days; min 3 distinct insiders; event date = latest filing date in the cluster; trade-date fallback = trade + 2 days (`:58`); window reset after firing (`:77`).

#### `lab/strategies/short_interest.py` (134 lines)
- `_days_to_cover_trend` 76-89: (latest - prior)/prior over `lookback_cycles` settlements; requires lookback+1 readings.

#### `lab/strategies/short_volume_ratio.py` (142 lines)
- `_avg_ratios` 73-80: mean short_volume_ratio over the last `lookback_days` rows per symbol.

#### `lab/strategies/retail_attention.py` (130 lines)
- `_spikes` 73-86: latest / mean(previous `lookback_months` readings) - 1.

#### `lab/strategies/pead.py` (139 lines)
- `_most_recent_surprise` 82-94: latest surprise_pct if its age <= max_age_days.

#### `lab/strategies/headline_sentiment.py` (102 lines) and `congress_trades.py` (137 lines)
Selection and mean-score code (`headline_sentiment.py:76` groupby mean) only. Orchestration bodies not opened line by line (UNVERIFIED).

#### `lab/strategies/__init__.py` (143 lines), `registry.py` (69 lines)
Registry: 38 entries in `STRATEGY_REGISTRY` (`__init__.py`, counted). `apply_sector_constraints` (`registry.py:58-68`): keeps first max_sec symbols per sector using static `data/universe/sp500_sectors.json`; unknown sector bucket "Unknown".

#### `lab/allocation.py` (114 lines) — sleeve overlay math (used by blend and the live overlay)

| Function | Lines | Behaviour |
|---|---|---|
| `trailing_realized_vol` | 15-20 | rolling std x sqrt(252), default window 60 |
| `inverse_vol_weights` | 23-35 | w_i = (1/vol_i)/sum(1/vol_j); non-finite or non-positive vols dropped; equal fallback |
| `target_vol_scale` | 38-48 | clip(target_vol / blend_vol, 0, max_leverage); 0.0 when vol invalid |
| `combine_sleeves` | 51-114 | monthly rebalance (`resample("ME")`); weights and scale from returns strictly before each rebalance date (`:80`); equal weight and scale 1.0 during warm-up (`:82`, `:106-111`); decisions applied forward until next rebalance (`:101-104`) |

PROJECT-SPECIFIC: strictly-before-rebalance estimation (lookahead rule in code); warm-up convention.

#### `lab/blend.py` (194 lines)
- `blend_curves` 25-46: align sleeve curves on common index, `pct_change`, dropna, `combine_sleeves`, cumprod to START_CASH.
- `run_blend` 64-194: runs each sleeve through `run_wfo` (`:112-126`), then blend, SPY table, persistence. `max_leverage` default 2.0 (`:74`) with a warning docstring (`:79-87`) that the live value is 1.0.

#### `lab/regime.py` (52 lines)
- `classify_regime` 13-39: trend = SPY close vs EMA(200) (`:26-27`); realized = rolling std(20) x sqrt(252); bucket by rolling 33/66 percentiles over 252 bars (`:31-32`); `shift(1)` on all outputs (`:37`).
- `compute_regime_scalar` 42-52: map label -> scalar, default 1.0.

#### `lab/significance.py` (90 lines)
- `newey_west_se` 30-40: Bartlett kernel, variance = sum with weights (1 - lag/(L+1)), divided by n (`:37`, `:39`); SE = sqrt(var / n).
- `excess_stats` 43-78: aligns curves, daily excess return, NW SE, t = mean/SE (`:67`), annualised excess (`:68`), tracking error with ddof=1 (`:69`), IR = excess / TE (`:75`), CAGR diff via `curve_stats` (`:76-77`).
- `years_for_t` 81-83: (target_t / IR)^2.
- `passes_pretax_bar` 86-90: t >= 3.0 AND CAGR diff >= 1.0 pp (constants `:13`, `:15`).

PROJECT-SPECIFIC: pre-registered thresholds (`T_BAR`, `CAGR_BAR_PP`).

#### `lab/event_study.py` (125 lines)
- `in_position_mask` 34-41: cumsum of entries - exits, shifted 1 bar.
- `split_event_returns` 44-50 and `split_event_returns_by_date` 53-70: in-position vs out-of-position returns.
- `event_study_test` 84-110: Welch's t-test via `scipy.stats.ttest_ind(equal_var=False)` (`:103`).

#### `lab/momentum_screen.py` (235 lines) — research screen, not on the trading path
- `net_excess` 45-48: winners - market - annual_cost/12.
- `nw_t` 51-55: mean / Newey-West SE (calls `significance.newey_west_se`).
- `one_sided_p` 58-60: `scipy.stats.norm.sf(t)`.
- `holm` 63-72: step-down Holm rejection.
- `cagr_diff_pp` 75-82: monthly compounded CAGRs; annualised = expm1(sum(log(gross)) x 12/n).
- `capm` 85-96: OLS (numpy lstsq) on excess returns; alpha x 12 x 100; beta.
- `relative_drawdown` 109-131: wealth ratio, cummax drawdown, peak/trough/recovery months.
- `worst_rolling_relative` 134-147: rolling 12-month relative log-wealth minimum.
- `screen` 173-: assembly (not opened line by line, UNVERIFIED for exact tests).

#### `lab/factor_screen.py` (182 lines), `net_issuance_screen.py` (170 lines)
- `value_momentum_combo` 34-36: 0.5 value + 0.5 momentum.
- Screens reuse momentum_screen primitives (import at `factor_screen.py:18`, `net_issuance_screen.py:16`).

#### `lab/replica.py` (290 lines) — event-loop replica of the live trader
Key API: `ReplicaConfig` 35; `ReplicaResult` 50; `build_signal_panel` 69; `run_replica` 96-290.
Imports from `paper/` (`cash_sweep` 14, `catastrophe_stop` 22, `risk` 27): reuses live pure guard code.

| Item | Lines | Behaviour |
|---|---|---|
| simulation loop | 178-290 | day loop with lagged signals (`:178`, `signal_lag_bars`), cash accrual by daily rate (`:179`), delisting sells (`:182-183`), NAV record with cash-conservation check, halts via `RiskGuard` (`:187`), catastrophe stops (`:200-202`), signal sells (`:208`), cost in bps |
| costs | 96-134 | per-side bps on stock and SPY sells/buys |

PROJECT-SPECIFIC: the whole loop (it is a hand-written broker); parity intent documented in module docstring.

#### `lab/a23_rebalance.py` (319 lines) — hand-written broker for the SPY/SSO 70/30 sleeve
Key API: `Lot` 18; `Book` 25-150; `_trade_to` 152-167; `band_trigger` 170-172; `check_rebalance` 175-.

| Item | Lines | Behaviour |
|---|---|---|
| `Book` methods | 43-144 | value (43), qty (48), buy (51), sell (59), distributions (100), year_end (116), pay_tax (130), liquidate (144) |
| `band_trigger` | 170-172 | weight outside target +/- 0.08 (`SSO_BAND`, `:12`), inclusive |
| `_trade_to` | 152-167 | sells first, then buys; min trade $1 (`:156`, `:161`) |

#### `lab/century_leverage.py` (271 lines) — 1926-2026 static leverage study
- `build_legs` 116-145: leverage leg construction from French factors and Shiller yield.
- `simulate` 159-242: own portfolio loop (A23 Book, lines 159-242); CAGR `(terminal/INITIAL)^(365.2425/days) - 1` (`:219`); max drawdown `min(nav/peak - 1)` with peaks from `np.maximum.accumulate` (`:226-227`); recovery-session count (`:229-238`).
- `parse_french_daily` 47-79; `load_french_daily` 81-93; `load_shiller_yield` 94-115 (data I/O).

PROJECT-SPECIFIC: tax-aware (ST/LT rates `0.24/0.15`, `:22`) via the Book; wash-sale logic inside Book.

#### `lab/tax_lots.py` (583 lines) — standalone taxable-lot ledger
Key API: `TaxParams` 26; `Realization` 40; `LedgerEvent` 82; `verify_events` 116; `TaxLotBook` 206-583.

| Item | Lines | Behaviour |
|---|---|---|
| `sell` | 328-394 | HIFO (highest-cost lots first) per docstring `:329` |
| wash matching | 298-327 | `_match` records wash adjustments against replacement lots |
| tax assessment | 466-523 | `_taxes` (ST/LT netting with carryforwards), `_refresh_assessments` |
| year end | 524-538 | assesses annual tax |
| hash chain | 97-131 | SHA-256 chain of ledger events (`verify_events`) |

PROJECT-SPECIFIC: entire file (standalone taxable account rules, wash-sale handling, hash-chained audit trail).

#### `lab/persist.py` (334 lines), `lab/cli.py` (256 lines), `lab/negative_cache.py` (73 lines), `lab/strategy.py` (60 lines)
Storage (SQLAlchemy `text`, `create_engine`), CLI parsing, no-data cache, strategy protocol. No quant primitives. `strategy.py:21-30` holds `LabConfig` defaults (top_n 50, lookback 252, skip 21, min_history 400).

#### `lab/train_gate.py` (275 lines) — offline ML gate training
- `_build_dataset` 66-135: feature frame from entries.
- `train` 136-275: `TimeSeriesSplit` (`sklearn.model_selection`, `:33`), `lightgbm.LGBMClassifier` (`:202`, `:236`), `predict_proba` (`:205`), joblib dump. Libraries: lightgbm, sklearn, joblib (`:30-33`).

#### `lab/wfo_attribution.py` (68 lines), `lab/wfo_policy_synthetic.py` (115 lines)
Pure classification and synthetic-path simulation for WFO policy studies. `simulate_run` 22-68 and `run_grid` 89-: numpy simulation of preregistered policies (not opened line by line, UNVERIFIED for exact formulas).

#### `lab/fomc_calendar.py` (137 lines)
Scrape and parse FOMC dates (requests; BeautifulSoup not used here). No quant primitives.

#### Data loaders: `french_data.py` (194), `fred_data.py` (145), `form4_data.py` (235), `house_ptr_data.py` (200), `short_interest_data.py` (244), `short_volume_data.py` (150), `google_trends_data.py` (138), `earnings_surprise_data.py` (143), `headline_sentiment_data.py` (387)
Data I/O only (HTTP fetch, parse, SQL cache). Grep for `rolling`, `pct_change`, `ewm` in these files returned zero hits. French parse uses sha256 checksum (`french_data.py`, `century_leverage.py`).

#### Event-timing audit: `event_timing_audit.py` (3,346), `event_timing_sources.py` (907), `event_timing_reconstruction.py` (492), `event_acceptance_diagnostic.py` (145)
Read-only evidence collection (psycopg2, bs4, zoneinfo). Grep for `rolling`, `pct_change`, `ewm`, `from_signals`, `sharpe` returned zero hits in all four files. No return math.

---

## 3. Paper modules (`src/ggTrader/paper/`)

Live path imports lab code: `overlay.py:11-14` (`lab.data`, `lab.simulate`, `lab.strategies.ensemble`, `lab.strategy`), `signal_runner.py:10-12`, `allocation` imported inside `overlay.py:82-86`. No vectorbt import in `paper/`.

#### `paper/risk.py` (135 lines)
`RiskConfig` 13-21 (defaults: max_positions 30, position_pct 0.033, max_concentration 0.05, daily_loss 0.03, max_drawdown 0.15; `MAX_POSITIONS` env override `:24-27`); `RiskGuard` 30-135.

| Item | Lines | Behaviour |
|---|---|---|
| drawdown halt | 45-55 | (peak - value)/peak >= 0.15 -> halt |
| daily loss | 57-67 | (day_start - value)/day_start >= 0.03 -> halt |
| sleeve slot caps | 89-117 | floor(w x max_positions), minimum 1 for w > 0, trim largest cap until total <= max_positions |
| sleeve notional | 119-135 | round(value x w x scale x position_pct, 2) |
| concentration | 77-87 | (current + prospective)/value >= 0.05 -> reject |

PROJECT-SPECIFIC: all of it (limits and env override); `position_pct` 0.033 matches `STOCK_BASE_CONFIG` comment (`data.py:28`).

#### `paper/overlay.py` (114 lines)
- `compute_sleeve_curve` 19-71: rebuilds one sleeve's trailing curve by replaying EnsembleSignal over daily plans (`:54-66`) then `simulate_signals` (`:67-69`, vbt from_signals).
- `compute_weights_and_scale` 74-105: per-sleeve `pct_change`, `trailing_realized_vol` (60), `inverse_vol_weights`, blended returns, `target_vol_scale` with `max_leverage=1.0` (`:78`).
- `should_rebalance` 108-114: True on first run of a new calendar month.

PROJECT-SPECIFIC: 90-day window, 20-day trim margin, 90-day warm-up (`:38-40`), explained as a `pct_change` row-count fix in the code comment (`:26-37`); target_vol 0.068 (`:76`).

#### `paper/cash_sweep.py` (179 lines)
Env-driven defaults: sweep off by default (`:44-46`), SPY, reserve 5% (`:54`), min clip $500 (`:60`), buy trigger 8% (`:67`) (hysteresis dead-band above the reserve, per the docstring). `compute_sweep_buy` 86-121; `compute_sweep_sell_for_funding` 122-149; `estimate_prospective_buy_notional` 150-.

PROJECT-SPECIFIC: all of it.

#### `paper/catastrophe_stop.py` (80 lines)
`unrealized_pct` 47-63: unrealized_pl / cost_basis (not broker unrealized_plpc). `find_catastrophe_stops` 66-80: symbols with pct <= threshold (default -0.25, `:33`). Default OFF via env (`:36-38`).

PROJECT-SPECIFIC: all of it.

#### `paper/feature_gate.py` (200 lines)
- `extract_features` 29-119: 10 features. vol_5d/vol_20d = rolling std of pct_change x sqrt(252) (`:50-53`); RSI(14) via `ewm(alpha=1/14, adjust=False, min_periods=14)` (`:60-64`); BB %B with SMA(20) and 2 x std (`:67-72`); 5-day and 20-day returns (`:75-76`); volume ratio vs 20-day mean (`:79-80`); ATR ratio = mean TR over 5 / mean TR over 20 (`:83-92`) with true range when high/low given, else abs diff of close.
- `FeatureGate` 122-200: LightGBM predict_proba (`:153`), joblib load; env flag `_gate_enabled_by_env` (`:25`).

PROJECT-SPECIFIC: 10-feature set; default-off gate (memory notes say disabled; not re-verified here).

#### `paper/signal_runner.py` (196 lines)
`refresh_benchmark_tape` 30-59; `generate_signals` 60-135: one EnsembleSignal select/to_targets on last bar, default `lookback_days=365` (`:60`, rationale in docstring `:63`); reads last-bar entries and exits (`:97-98`). `generate_blended_signals` 136-174; `generate_core_signals` 175-196.

#### `paper/split_check.py` (194 lines)
`compute_split_corrections` 39-68, `_qty_around_ex_date` 69-92, `find_split_applied_symbols` 93-153, `corrected_market_value` 155-159 (value x factor), `corrected_unrealized_pl` 160-164 (value x factor - cost), `apply_corrections_to_positions` 165-.

PROJECT-SPECIFIC: all of it (broker split corrections, ex-date logic).

#### `paper/dividend_check.py` (145 lines)
`normalize_since` 38-54; `qty_held_at_ex_date` 55-76; `compute_dividend_accruals` 77-145 (accrues dividends the broker did not credit; idempotency keyed on prior runs).

PROJECT-SPECIFIC: all of it.

#### `paper/trader.py` (1,118 lines) — orchestrator
`PaperTrader` 89; `run` 493-1100 (single method, all sizing and halts inline):
- missed-exit catch-up `:642-646` (uses `_missed_exits` 104-131);
- drawdown halt `:665-675`; daily loss `:690`; catastrophe stops `:720`;
- sleeve caps `:769`, sleeve notional `:774-778`, slot loop `:880-928`;
- cash sweep `:800`, `:944`.
Not opened line by line beyond these call sites (UNVERIFIED for exact ordering).

#### `paper/alpaca_broker.py` (255 lines), `paper/persist.py` (465 lines), `paper/notifier.py` (155 lines)
Broker adapter (alpaca-py), SQL persistence, Telegram. No quant primitives.

---

## 4. Consolidated table: hand-rolled primitive -> where it lives

Each row is a primitive implemented in Python/pandas/numpy, not by a vectorbt call. "vbt use" records only what the file itself calls.

| Primitive | Where (file:lines) | vbt use in that place |
|---|---|---|
| SMA / rolling mean | `indicators.py:69, 154, 246, 305, 356, 384`; `conviction.py:56`; `treasury_curve.py:64`; `commodity_trend.py:80`; `leveraged_trend.py:59`; `fx_hedge_overlay.py:77` | none |
| Bollinger bands / %B | `indicators.py:67-81, 152-158, 237-278, 354-361, 380-391`; `feature_gate.py:67-72`; `conviction.py:54-63` | none |
| RSI (Wilder-form ewm) | `indicators.py:118-138, 161-171, 281-291, 342-351`; `feature_gate.py:60-64` | none |
| EMA / EMA cross / EMA gap | `indicators.py:141-149, 174-179, 364-368`; `signals.py:116-149`; `leveraged_trend.py:59` | `signals.py:57` tournament only |
| MACD (line, signal, hist, divergence) | `indicators.py:182-212, 215-228, 371-377` | none |
| Stochastic, Keltner, OBV, Donchian, VWAP, Calmar, Wilder ATR | not present in `lab/` or `paper/` (grep) | n/a |
| ATR (simple mean of TR) | `simulate.py:123-150`; `feature_gate.py:83-92` | ATR stop fraction consumed by `sl_stop` at `simulate.py:229-241` |
| Overnight-gap z-score | `indicators.py:84-115` | none |
| Volume spike gate | `indicators.py:237-263`; `feature_gate.py:79-80` | none |
| Weekly resample RSI | `indicators.py:281-291` | none |
| Rolling std, realized vol (x sqrt 252) | `simulate.py:116-119`; `allocation.py:15-20`; `regime.py:29-32`; `feature_gate.py:50-53`; `overlay.py:60` (via allocation) | none |
| Vol-target scalar | `simulate.py:104-120`; `allocation.py:38-48` | sized via `size` param at `simulate.py:255-263` |
| Inverse-vol weights | `allocation.py:23-35`; `overlay.py:74-105` | none |
| Target-vol blend and monthly overlay | `allocation.py:51-114`; `blend.py:25-46` | none |
| Regime classifier (EMA200, vol tercile) | `regime.py:13-39` | none |
| Sharpe | `metrics.py:19-23` (curve_stats); `signals.py:69` (vbt accessor, tournament only) | `signals.py:69` only |
| Sortino (downside std of negatives) | `metrics.py:24-29` | none |
| Max drawdown | `metrics.py:30`; `wfo.py` uses via curve_stats; `replica` via RiskGuard `risk.py:45-55` | none |
| CAGR | `metrics.py:16, 33`; `century_leverage.py:219` (365.2425 days) | none |
| Total return, annual vol | `metrics.py:17-18` | none |
| Monthly hit rate vs SPY | `metrics.py:46-49` | none |
| Benchmark curve | `metrics.py:41-56`; `harness.py:131-133`; `wfo.py:840-849` | none |
| Trade pairing and round-trip returns | `kelly.py:15-81`; `simulate.py:97` (vbt `trades.count`) | `simulate.py:97` only |
| Kelly fraction (pooled, causal) | `kelly.py:84-126, 129-178`; `ensemble_kelly.py:122-138` | none |
| Newey-West SE | `significance.py:30-40`; `momentum_screen.py:51-55` (via significance) | none |
| Excess-return t, IR | `significance.py:43-78` | none |
| Welch t-test | `event_study.py:84-110` (scipy) | none |
| CAPM OLS alpha/beta | `momentum_screen.py:85-96` (numpy lstsq) | none |
| Holm step-down | `momentum_screen.py:63-72` | none |
| Relative wealth and drawdown | `momentum_screen.py:99-131` | none |
| Spearman rank IC | `ic_weights.py:19-39` | none |
| IC-weight schedule (leak-guarded) | `ic_weights.py:42-87` | none |
| Expectancy grid | `wfo.py:573-580` | `simulate.py:97` supplies `n_trades` |
| Walk-forward fold generation | `wfo.py:194-210` | none |
| Composite score (min-max) | `wfo.py:213-248` | none |
| Anchor set (min-DD with CAGR floor) | `wfo.py:465-521` | none |
| Circuit breaker (chronic WFE, acute negative Sharpe) | `wfo.py:79-125` | none |
| Shadow re-entry | `wfo.py:128-164` | none |
| WFE | `wfo.py:44-56` | none |
| NDH (density, variance cap, plateau exemption) | `gates.py:22-119` | none |
| PSR, expected max SR, DSR | `gates.py:134-215` | none |
| Parameter grid and validity | `sweep.py:38-74`; `wfo.py` winner paths | none |
| Stop-config batching | `sweep.py:88-98` | groups become one vbt call at `sweep.py:179` |
| PIT eligibility mask | `indicators.py:30-64`; `sweep.py:101-127`; `data.py:297-314` | none |
| Rebalance calendar | `data.py:266-294`; `paper/overlay.py:108-114` | none |
| Leak check | `harness.py:19-34` | none |
| Warm-up trim | `harness.py:120-125` | none |
| ATR trailing stop | `simulate.py:123-150, 209-241` | `sl_stop` and `sl_trail` passed to vbt, lines 205-207, 240-241 |
| Time stop (max hold) | `ensemble.py:211-221`; `ensemble_ic.py:142-146` | comment at `ensemble.py:213-217` says vbt has no native time stop (UNVERIFIED) |
| Take-profit | `simulate.py:244-246` (param only) | `tp_stop` passed to vbt, `:246` and `:289` |
| Conviction sizing | `ensemble.py:434-441`; `conviction.py:54-63` | `sizes` into vbt `size` via `simulate.py:265-275` |
| Multi-sleeve OOS overlay | `allocation.py:51-114` | none |
| Hysteresis rotation | `leveraged_rotation.py:32-87`; `leveraged_trend.py:46-67` | none |
| Pair spread z and state | `pairs_stat_arb.py:63-110` | none |
| Momentum 12-1 | `momentum.py:30-42`; `commodity_trend.py:53-67` | none |
| Idio residual variance | `idio_vol.py:14-36` | none |
| Trailing max return | `max_effect.py:14-19` | none |
| Insider cluster and deletion event logic | `insider_cluster.py:25-78`; `index_deletion.py:36-51` | none |
| Event-window signals | `fomc_drift.py:117-149`; `month_end_treasury.py:77-108` | none |
| FX real rate and trend | `fx_hedge_overlay.py:76-111` | none |
| Tax lots (HIFO, wash, ST/LT tax) | `tax_lots.py:206-583` | none |
| A23 broker (buy/sell/band) | `a23_rebalance.py:25-172` | none |
| Century leverage simulator | `century_leverage.py:159-242` | none |
| Replica day loop | `replica.py:96-290` | none |
| Live risk guard (halts, caps) | `paper/risk.py:13-135`; `paper/trader.py:665-720, 769-800` | none |
| Catastrophe stop | `paper/catastrophe_stop.py:33-80` | none |
| Cash sweep sizing | `paper/cash_sweep.py:54-149` | none |
| Feature gate indicators | `paper/feature_gate.py:29-119` | none |
| Split and dividend corrections | `paper/split_check.py:39-164`; `paper/dividend_check.py:38-145` | none |
| Universe construction (sector cap) | `strategies/registry.py:58-68`; `data.py:255-263` | none |
| Data cache and fallback chain | `data.py:57-184`; `negative_cache.py` | none |

---

## 5. Observations stated as facts from the code (no recommendation)

1. vectorbt is used for portfolio accounting and for one tournament Sharpe. Metrics, WFO, gates, Kelly, IC, and the paper trader do not use it (section 1).
2. Returns are derived from equity curves (`pct_change` on value), not from vbt's returns accessor (`simulate.py:82`, `:296`).
3. The ATR used for stops is a simple mean of TR (`simulate.py:149`), not Wilder smoothing. The RSI uses ewm with alpha = 1/period (`indicators.py:125-126`), a Wilder-form smoothing.
4. Sortino's denominator is the std of negative returns only (`metrics.py:24-27`).
5. Sharpe has no risk-free term (`metrics.py:19-23`). The anchor-set CAGR floor is a percent (`wfo.py:472`).
6. `signals.py:75` swallows every exception from the vbt tournament and returns -inf for every combo.
7. `init_cash=10000.0` in the tournament (`signals.py:63`) is not read from `base_config`.
8. `base_size` fallback 0.02 (`simulate.py:249`) differs from the lab default 0.03 (`data.py:29`).
9. `blend.run_blend` defaults `max_leverage` to 2.0 (`blend.py:74`) while `paper/overlay.py:78` uses 1.0; the docstring at `blend.py:79-87` records a past mismatch.
10. Stochastic, Keltner, OBV, Donchian, VWAP, Calmar and Wilder ATR are absent from `lab/` and `paper/`. Memory notes mention StochRSI and Keltner strategies added April 2026; they are not in `STRATEGY_REGISTRY` (`strategies/__init__.py`). Whether they were removed is UNVERIFIED (git history not inspected).

## 6. Not opened line by line (UNVERIFIED)

- Selection and to_targets bodies of: `headline_sentiment.py`, `congress_trades.py`, `index_deletion.py` (select), `insider_cluster.py` (select, to_targets), `short_interest.py` (select, to_targets), `short_volume_ratio.py` (select), `retail_attention.py` (select), `pead.py` (to_targets), `treasury_curve.py` (select, to_targets), `fomc_drift.py` (select, to_targets), `fx_hedge_overlay.py` (select, to_targets), `month_end_treasury.py` (select, to_targets), `ensemble_ic.py` (select), `ensemble_kelly.py` (select, to_targets, sweep_signals), `conviction.py` (select, to_targets, sweep_signals), `momentum_screen.py` (screen body), `wfo_policy_synthetic.py` (formulas), `wfo_attribution.py`, `train_gate.py` (feature construction), `paper/trader.py` (full run body beyond the call sites), `paper/signal_runner.py` (generate_core_signals, generate_blended_signals).
- Event-timing audit (all four files): confirmed to have no return math by grep, not read.
- The Book class in `a23_rebalance.py` (methods listed by signature; `buy`, `sell`, `distributions`, `year_end`, `pay_tax`, `liquidate` bodies not read).
