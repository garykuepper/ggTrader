# S5: Validation and re-baseline protocol for the vectorbt 1.2.0 + Rust migration

Date 2026-11-06. Author: Opus 5.5 (S5 workstream). Repo read-only at `6774b4d3` (`main`).
Evidence: `.../scratchpad/vbt-spec/S5-evidence/` (EV below). Prototype: `s5_dump_tape.py`,
`s5_signals.py`, `s5_compare.py`, `s5_engine_metrics.py`. Outputs: `signal_diff_report.{txt,json}`,
`engine_metrics.json`, `tape_meta.json`, `arms/*/`. Prior work: `vbt-research/D-...md` §3, `F-...md`, `B-...md`.

## 0. Bottom line

1. **Configuring vbt's own kernels to the old conventions is a near-zero-change migration.** On 150 real SP500
   symbols x 1,968 bars (2019-01-02 to 2026-10-29), changing the stack (pandas 2.3.3 to 3.1.1) changed 0 signals.
   The configured vbt kernels changed 215 of 24,829 ensemble exit cells and 1 of 4,936 entry cells. All but one
   differing cell sit in each symbol's first 200 bars. The one exception is an exact float tie on a junk tape
   (PARA, see 6.7). The **live 365-day window gave 0 differences in 120 of 120 sessions**. vbt's own `crossed_above`
   and `crossed_below` gave 0 differences against ggTrader's prev/now comparisons (EV `signal_diff_report.txt`).
2. **vbt's default indicators define a different strategy, not a re-measurement.** 59% of ensemble entry cells
   and 48% of exit cells differ. 91% of RSI entries and 95% of MACD exits differ. The live signal set differs in
   95 of 120 sessions (entries) and 119 of 120 (exits). Adopting the defaults is a new strategy that needs the
   full forward-paper standard, and that is an owner decision. I recommend configuring the kernels (section 1).
3. **Rust is bit-identical to numba on the real tape and really dispatches.** `engine="auto"` and `"rust"` called
   9-11 `*_rs` kernels per portfolio. Max abs value diff = 0.0 on 3 configs (cash-shared, trailing 8%, trail+TP).
   Indicator kernels match: RSI, MACD and EMA 0.0, BBANDS 1.3e-11 (EV `engine_metrics.json`).
4. **Metric traps confirmed on 1.2.0.** The default `year_freq='365 days'` raises Sharpe 0.336 to 0.404 (+20%)
   and "CAGR" 2.29% to 3.35% (+46%). A flat curve still gives Sharpe **+inf**, and a curve with no downside gives
   Sortino +inf. Sortino differs by definition even at 252 (0.487 vs 0.431). `deflated_sharpe_ratio` on one
   column returns NaN unless `var_sharpe` is passed.

## 1. Change classification

**(i) Must be bit-for-bit** (any diff = STOP): tape loading and the parquet snapshot hash; the PIT membership mask
(`pit_entry_mask`/`eligible_at`; F verified the vectorised form 0 mismatches on 123 dates); `generate_folds`
(`wfo.py:194-210`, 17 folds 2021-01-31..2026-04-30); grid and `_is_valid_combo`; order and portfolio mechanics for
identical signals (`from_signals`/`from_orders`). This is proved across 0.28.5, 1.0 and 1.2 and numba/Rust by B
(5 cases + 102 fuzz configs) and here on real tape (0.0). Also gates, anchor and breaker logic for identical
inputs; and Sharpe, ann. vol and MaxDD at `year_freq='252 days'` (0.33582513683196 vs ...687; equal to 1e-15 here).

**(ii) Equal within tolerance, judged by signal flips, not by value diff**: SMA/rolling mean (4.6e-12),
BBANDS ddof=1 (7.6e-10 vs pandas; Rust vs numba 1.3e-11), MSTD ddof=1 (3.8e-10), ATR `ewm=False` (1.2e-13),
rolling vol (3.8e-15), MACD with both ewm flags (exact after ~200 bars). Tolerance: |diff| <= 1e-9 x scale on
post-burn-in cells, AND 0 signal flips after a 200-bar burn-in except flips that a tie audit attributes to an
exact tie (|x - threshold| < 1e-9). The prototype found exactly one such flip (PARA 2021-12-31: close equals the
pandas SMA exactly and vbt's SMA is 9.1e-13 higher; `>=` flips). Value tolerances alone are not enough.

**(iii) Intentionally different definitions. Each needs a documented choice.** Recommendation per item;
"configure" means use vbt's own kernel with a vbt-exposed knob, with no hand-written maths.

| Item | Old (ggTrader) | vbt 1.2.0 default | Measured effect | Recommendation |
|---|---|---|---|---|
| Annualisation | x sqrt(252) (`metrics.py:10`) | `year_freq='365 days'` (`vbt.settings.returns`) | Sharpe +20.4%, ann. vol +20%, "CAGR" +46% rel. | **Configure** `vbt.settings.returns['year_freq']='252 days'` at import of one compat module, and pass it explicitly on every call. Test asserts it. Daily equity bars: 365 is wrong. |
| Return std ddof | pandas ddof=1 | returns `ddof=1` | equal | Adopt default (same). |
| BB / MSTD ddof | ddof=1 | `ddof=0` | BB entries 7,154 to 7,708 (+7.7%); 1,185 post-burn-in flips | **Configure** `ddof=1` (exposed, `basic.py` kwargs_to_args). ddof=0 is Bollinger's own population SD, so a separate Phase-4 experiment *may* adopt it: **OWNER**. |
| RSI form | Wilder alpha=1/n, min_periods=n | Cutler SMA (`ewm=False`, window 14) | 91% of RSI entry cells differ | **Configure** `RSI.run(window=2n-1, ewm=True)`, wrapped as `wilder_rsi(n)`, with a contract test that n=14 maps to 27. Warm-up 27 vs 14 bars (burn-in only). Zero-loss gives 100 vs NaN: adopt vbt's 100 (it is the textbook value). |
| MACD form | EMA 12/26/9, pandas seed bar 0 | SMA MACD (`macd_ewm=False`) | 95% of exits differ | **Configure** `macd_ewm=signal_ewm=True` (Appel's MACD is EMA). The remaining seed difference decays by bar ~200. |
| EMA warm-up | pandas emits from bar 0 | NaN for first w-1 bars | 166 entry / 164 exit cells, all in first 200 bars | **Adopt vbt** (seed-dominated early EMA values are noise). Live window: 0 diffs. |
| ATR | SMA of TR (`simulate.py:149`) | `ewm=True`, span-based (not Wilder) | 1.23 abs | **Configure** `ewm=False`. vbt's first TR = H-L (one bar earlier): accept, burn-in only. |
| OBV | unused | 1.2.0 fixed tie handling | n/a | Adopt vbt if ever used. |
| Crossovers | `prev>=b & now<b` | `crossed_below` (stateful `was_above`, strict, NaN resets; `generic/nb.py:1972`) | **0 cells** on 150x1,968 | **Adopt vbt.** Keep a tie-case unit test, because the semantics differ on exact ties. |
| Sortino | mean / std(neg returns) (`metrics.py:23-28`) | mean / sqrt(mean(min(r,0)^2)) | 0.431 vs 0.487 (+13%) | vbt's version is the textbook downside deviation. **Adopt only via a Phase-4 one-at-a-time WFO.** It enters `composite_score` at weight 0.3 and can flip fold winners: **OWNER**. |
| CAGR | calendar days/365.25 | `annualized()`, bars/year_freq | 2.295 vs 2.301 at 252 | **Adopt vbt at 252** for strategy *and* SPY, documented as "bar-annualised". Small effect, but it is a headline metric: **OWNER**. Never at 365. |
| Deflated Sharpe | n_trials=len(grid), var=1/(T-1), non-excess kurtosis | trials = n columns, var across columns; 1.2.0 kurtosis fixed (`fisher=False`) | single column gives NaN | **Configure** `deflated_sharpe_ratio(nb_trials=len(grid), var_sharpe=1/(T-1))`. It reproduces ggTrader exactly in D's test (0.6236297914). Re-check the 0.80 gate on the reference: **OWNER** if any gate flips. |
| Zero-variance Sharpe | NaN | **+inf** (`returns/nb.py:419-420`, even for mean <= 0) | 16-20 of 48 combos were inf in D's synthetic folds | **No knob, so write code:** one `finite_or_nan()` guard before every `idxmax`/rank, plus a test. Same for Sortino +inf. |

## 2. Test pyramid

**(a) Indicator contract tests** (`tests/lab/vbt_compat/`): one fixture CSV per indicator (tiny hand-checkable
series: monotone, flat, tie-at-threshold, leading NaN, interior NaN gap, zero-loss) with golden values from
a textbook reference: Wilder 1978 RSI worked example; Bollinger SD convention; Appel MACD. Also assert the
*configuration*, e.g. `wilder_rsi(14)` calls vbt with `window=27, ewm=True`, `year_freq == '252 days'`,
`ddof == 1`. Goldens are frozen JSON, regenerated only by an explicit script plus owner-visible diff.

**(b) Signal-diff report (prototype run).** Design: arms that differ by **one** change each:
A = legacy code / pandas 2 (reference); B = legacy code / pandas 3 (stack); C = vbt kernels configured (kernel
values); D = C + vbt crossovers (semantics); E = vbt defaults. Per voter and side it reports cells in each arm,
XOR cells, XOR/(signal-union), XOR after the 200-bar burn-in, symbols affected and the first diff. It also reports
one-at-a-time ensemble attribution (swap one voter into the legacy set and re-vote), and a live-window panel
(365 calendar days to each of the last 120 sessions, last-bar buys/sells, mirroring `signal_runner.py:60-100`).
Deployed params = `EnsembleSignal` defaults, `FIVE_VOTERS`, min_agree 2/2 (`ensemble.py:31-42, 66-87`).

Prototype results (150 symbols; tape sha256 `2df2de85...`, EV `tape_meta.json`):

| Pair | ENS entries XOR (post-200) | ENS exits XOR (post-200) | Live sessions differing (ent / exit) |
|---|---|---|---|
| A to B (pandas 3) | 0 (0) | 0 (0) | 0/120 / 0/120 |
| A to C (vbt configured) | 1 (0) | 215 (1, PARA tie) | 0/120 / 0/120 |
| C to D (vbt crossovers) | 0 (0) | 0 (0) | 0/120 / 0/120 |
| A to E (vbt defaults) | 5,161 (4,732) | 16,469 (14,682) | 95/120 / 119/120 |

Attribution under defaults (ensemble entry XOR when that voter alone is swapped in): RSI 3,099, MACD 2,153,
BB 509, VBB 48, EMA 0. Under the configured kernels, the exit differences come from RSI warm-up (81), EMA
warm-up (133) and MACD warm-up (133), all within the burn-in. **Full-run spec:** the same harness over the full
PIT SP500 set (~500-650 symbols) for 2019-03-03..eval-end, plus `ensemble`'s 48-combo grid (`ensemble.py:120-140`)
rather than only the defaults. **Pass for configured migration:** 0 post-burn-in flips that the tie audit does
not explain, and 0 live-panel diffs. Limits: 150 sampled symbols, no PIT mask applied (raw signals),
`td_stop` not exercised.

**(c) WFO comparison old vs new on a pinned window.** Reference = `sp500_core_17fold`, `--eval-start
2021-01-31 --eval-end 2026-04-30`, leverage 1.0, STOCK_BASE_CONFIG (fees 0, slippage 5 bp, size 0.03; `data.py:18-34`).
Legacy figures: Sharpe 0.592, CAGR 3.08%, MaxDD -5.28%, SPY 0.779, gates 17/17
(`research/audits/rebaseline/_pit_rebaseline_20260923.json`). First re-run it on the legacy engine (P0) to
confirm it still reproduces. Compare per fold: winner combo, anchor used, NDH/DSR pass, IS/OOS Sharpe, WFE,
`n_trades` *in the scoring window*. Compare spliced: OOS Sharpe/CAGR/MaxDD, SPY same-window figures, trade
count, monthly hit rate.
* **Consistent** (configured migration, classes i and ii): identical winners 17/17, identical gate verdicts,
  |dSharpe| < 0.01, trade count within 0.5%, verdict vs SPY unchanged.
* **Materially different**: any winner flip, gate flip, verdict change vs SPY, or |dSharpe| >= 0.05. Note that
  the OOS Sharpe SE over ~4.25 test-years is about 0.5, so 0.05 is a *pipeline* tripwire, not a significance test.
* **Attribution:** every class (iii) change gets its own run against the configured baseline, one at a time
  (ddof, then Sortino, then CAGR, then DSR, ...), never bundled. The memory lesson "config tweaks one at a time" applies.
  Each run's JSON has a `changed:` field naming exactly one definition.

**(d) Leak / lookahead.** (1) Keep `harness.leak_check` (`harness.py:19-35`) for every weight strategy. (2) Add a
**signal truncation test**: signals at bar t computed on `close.loc[:t]` must equal the full-tape signals at t,
for 50 random t x all voters, in vbt form. This catches any centred or forward-looking kernel. (3) **Ban
`vectorbt.labels`** (FMEAN, FIXLB, TRENDLB, ... are forward-looking by design) in `src/ggTrader/lab/strategies/`
and `paper/` with a grep test. (4) If `range_split` is used: assert train_end <= test_start, the fold count is 17
for the reference, and the position-to-date map round-trips (`split` drops the index, D §1.1). (5) Prefix
causality: a full-history simulation sliced must equal a per-fold prefix (F: 0.0 on real code), signal path only.
The weight path starts flat per window and is NOT equivalent (F §D3). (6) Fill convention: pin
`from_signals` fill = close of the signal bar (ggTrader's documented convention, `indicators.py:82-93`) with a test,
and pin that the live trader's 12:45 PT intraday "close" is an approximation, not a change. (7) PIT: no buy fill
where the mask is False (prototype: 0 of 6,000+ buys), and membership as-of is 0-diff against `eligible_at`.

**(e) Engine tests.** Rust really dispatched: wrap the `vectorbt_rust.*_rs` callables with counters (B's
`install_rust_counters`; prototype saw 11 kernel calls per cash-shared portfolio) and assert > 0 under `auto`.
Assert `vbt.settings['engine']`. Assert `vectorbt` and `vectorbt-rust` share major.minor, because a mismatch
silently falls back to numba with one warning (B §4.2). Parity: numba vs Rust `value()`, order and trade counts
equal (exact) on a real-shaped fixture (>= 100 symbols x 1,500 bars, NaN holes). Indicators within 1e-10.
Callback paths (`adjust_sl_func_nb`, `from_order_func`) are never Rust: assert they are not on the hot path, or accept numba.

**(f) Invariants (property tests, hypothesis-style on random masks/configs):** cash >= -1e-9 per group with
`cash_sharing` (prototype min 1,774); recorded fees = sum(size x price x fee) (1043.381 vs 1043.381);
no buy fill outside the PIT mask or on a non-entry bar (0/0); position count <= cap where a cap is modelled;
total equity = cash + sum(position x close); no NaN in `value()` after the first bar; fold scoring windows
disjoint for test sets.

## 3. Reference artefacts and legacy treatment

**New references** (re-run on the new engine, each writing `research/audits/engine-migration/_<name>_<date>.json`):
1. `sp500_core_17fold`: 2021-01-31..2026-04-30, leverage 1.0, STOCK_BASE_CONFIG, sp500 PIT.
2. A 2-fold smoke variant (2021-01-31..2022-07-31) for CI-speed regression.
3. SPY buy-and-hold benchmark for both windows (pure pandas, but pandas 3 can still move it).
4. Live golden: last 120 sessions of 365-day-window buys/sells (the panel above, full universe).
5. Any study still feeding an open decision (production replica 14.19 vs 14.83, cross-asset 80/20 shadow) gets
   re-run only if its code path imports vbt or the changed indicators. Otherwise it is marked "pandas-3 unverified".

Each JSON records `{eval_start, eval_end, universe, max_leverage, fees, slippage, size, grid_hash, tape_sha256,
git_sha, vectorbt, vectorbt_rust, pandas, numpy, numba, engine, year_freq, definitions:{rsi, bb_ddof,
macd, sortino, cagr, dsr}, fold_rows[...], spliced{...}, spy{...}}`. The tape comes from a **frozen parquet
snapshot** with a recorded hash, not the live DB. The prototype showed `load_ohlcv` makes network fetches and
attempts `INSERT` / `CREATE TABLE` even on a research read (blocked here by `PGOPTIONS=-c
default_transaction_read_only=on`, EV `dump.log`). Reference runs must use that read-only guard.

**Legacy treatment (do not port closed NO-GOs):**
* `git tag legacy-engine-vbt0285 <last pre-migration main sha>` plus `requirements-legacy.lock` (`pip freeze` of
  `.venv`: vectorbt 0.28.5, pandas 2.3.3, numpy 2.2.6, numba 0.61.2, plotly 6.6.0) plus a recorded legacy image
  digest (the live image `ghcr.io/garykuepper/ggtrader@sha256:0f2403f1...`, readiness review). Add a
  `scripts/legacy_env.sh` that builds the venv from the lock at the tag, so any closed study is reproducible as
  `git worktree add ../legacy legacy-engine-vbt0285`.
* `research/ENGINE_PROVENANCE.json`: one row per results JSON (60 `_*.json` under `research/`) with `engine: legacy`
  plus the tag. New JSONs carry the engine block above.
* Docs: do **not** edit old numbers. Add one banner line to each report that cites a vbt-derived figure: "Numbers
  computed on the legacy engine (vbt 0.28.5 / pandas 2.3.3, tag legacy-engine-vbt0285); not re-run." Add an
  `Engine` column to `RESEARCH_SNAPSHOT.md` (regenerated by the `research-snapshot` skill, not by hand).
  152 md files under docs/research mention Sharpe or CAGR. Generate the banner list mechanically from
  ENGINE_PROVENANCE links.
* NO-GO verdicts stay NO-GO. A NO-GO is re-opened only by a registered re-test, never by a re-baseline that happens to look better.

## 4. Live trader protection

**Frozen until the readiness decision** (`docs/next_steps.md:9-13, 30-31`; readiness review lines 3, 46-54: about
2026-12-29 is the *earliest possible* review, "not an established clearance date"; a calendar date alone does not
lift the freeze). Do not change before then: `src/ggTrader/paper/*`, `lab/strategies/ensemble.py`,
`indicators.py`, `lab/simulate.py`, `lab/strategy.py`, `pyproject.toml` deps, Dockerfile/compose, `.env`. Do not
merge the migration to `main`. A push to `main` publishes `:latest` (`docker-build.yml`), and the cron's
`docker compose run --rm` would pick it up after any pull (readiness review lines 42-47). All work stays on a branch.

**Shadow protocol** (it needs no production change):
* A separate host job (own venv/image, not `ggtrader_live`) runs at about 13:05 PT on trading days. It fetches the
  365-day window once, freezes it to parquet with a hash, and computes buys/sells with **both** the legacy engine
  (legacy tag/lock) and the new engine on **that same frame**. It then reads (read-only) that day's
  `paper_trades`/`paper_snapshots`.
* Report per day (JSON plus one Telegram-free markdown table): old-shadow vs new-shadow buys/sells/last_exit
  (the engine diff, which must be 0); old-shadow vs live actual (input/timing noise baseline: intraday bar
  drift, AVB exclusion, risk caps); new-shadow vs live actual; and every diff with symbol, voter votes, indicator
  values and distance to threshold.
* **N >= 20 trading days and >= 30 shadow buy+sell signals**, whichever is later.
* **Success:** old-shadow vs new-shadow = 0 diffs on every symbol with >= 200 bars. Diffs on shorter histories
  must each be explained by a documented class (iii) warm-up rule. Old-shadow vs live agreement must be within
  the noise baseline, with each mismatch explained.
* **Gate: OWNER sign-off** on the shadow report, *and* the separate freeze-readiness decision, before
  `ggtrader-deploy` touches live. A rollback image digest must be recorded and pull-tested first.

## 5. Phase-gated acceptance checklist

| Phase | Evidence required | Artefacts | Decides | Stop if |
|---|---|---|---|---|
| P0 Freeze legacy | legacy 17-fold and 2-fold re-run reproduce cited JSON; live golden; tape snapshot | tag, lock, `_legacy_reference_*.json`, tape parquet+sha | Agent | 17-fold != 0.592/17-of-17 (investigate drift first) |
| P1 Stack port (pandas 3, numpy 2.5, vbt 1.2.0, **numba**, no definition change; drop dead `pandas-ta`) | signal-diff A to B = 0; 2-fold and 17-fold bit-for-bit; 44 `pct_change` sites pinned (F §d.4) | `_p1_*.json`, signal report | Agent | any diff |
| P2 Rust on | engine tests (e); WFO bit-for-bit vs P1 | `_p2_*.json`, kernel-call counts | Agent | any diff, or 0 Rust calls |
| P3 vbt kernels **configured** + vbt crossovers + inf guard | (a)-(d) pass; signal diff A to D within the (ii) rule; 17-fold "consistent" | `_p3_*.json`, tie audit | Agent proposes, **OWNER** accepts (it touches live-imported code) | unexplained post-burn-in flip; winner/gate flip |
| P4 Definitional changes, **one per run** (Sortino, CAGR basis, DSR kurtosis, optional BB ddof=0, any vbt default) | WFO vs P3 per change; attribution table | `_p4_<change>_*.json` | **OWNER per change** | (stop = owner rejects; never bundle) |
| P5 Re-baseline | new references §3; provenance manifest; banners; snapshot regenerated | `research/audits/engine-migration/` report | **OWNER** accepts new baseline | verdict vs SPY changes without an attributed cause |
| P6 Shadow | §4 report, N >= 20 days | daily JSON + summary | **OWNER** | any engine diff on a >= 200-bar symbol |
| P7 Live switch | freeze lifted by readiness decision; rollback tested | deploy record | **OWNER** | any pre-flight failure (`ggtrader-deploy`) |

**Owner decision points where results change:** P3 (live-imported code; burn-in-only diffs),
every P4 item, P5 baseline acceptance, P6, P7. P0-P2 must produce no number changes, or they do not ship.

## 6. Likeliest silent-failure modes and their tripwires

1. **365-day annualisation flatters everything** (+20% Sharpe, +46% rel. CAGR, measured). Tripwire: a test that
   `vbt.settings.returns['year_freq'] == '252 days'` after importing the compat module; a golden Sharpe on a fixture.
   Every results JSON records `year_freq`, and the reader rejects a JSON without it.
2. **+inf Sharpe/Sortino winners** (flat or no-trade combos). Tripwire: `assert np.isfinite(score[winner])`, a test
   with one flat column in a 48-column grid that must not win, and a count of non-finite scores per fold in the JSON.
3. **Silent vbt defaults** (ddof=0, Cutler RSI, SMA MACD, ATR ewm). Tripwire: every vbt indicator call goes
   through one compat module; a grep test forbids `vbt.RSI.run`/`BBANDS.run`/`MACD.run`/`ATR.run` elsewhere; contract
   tests assert the knobs.
4. **Warm-up NaN shifts first trades** (EMA w-1, RSI 2n-1, MACD 33 bars). Ensemble exits shifted in 215 burn-in cells.
   In live, a recent index addition with < 200 bars is exactly the case. Tripwire: a signal-diff "burn-in"
   column in every report; a live-golden test on a symbol with 60-120 bars.
5. **Float-tie flips at thresholds** (PARA case, 1e-12). Tripwire: a tie audit listing every flip with |x-thr|;
   unexplained flips stop the phase.
6. **Engine fallback** (version mismatch, int64 close, callbacks route silently to numba). Tripwire: Rust call counter
   > 0 in a smoke test; a version-pair assert.
7. **Junk or non-frozen tape** (PARA closes about 3,400 from late 2021, 303 zero-change days; the loader fetched from
   the network during the "read"). Tripwire: run references only from a hashed parquet snapshot; a tape sanity test
   (price jumps > 10x, runs of identical closes) before any reference run. Root cause of PARA: UNVERIFIED.
8. **pandas 3 semantics** (`pct_change` no padding, `agg(np.std)` ddof=0, `datetime64[s]` epochs, read-only arrays).
   Tripwire: P1 bit-for-bit; a `-W error::FutureWarning` lane on pandas 2 first; a unit-aware timestamp write test
   (the naive-UTC `ohlcv.timestamp` history).
9. **DSR semantics** (per-column trials; NaN on a single column; kurtosis fix). Tripwire: a test that the compat DSR
   equals D's reference 0.6236297914 on its fixture, with explicit `nb_trials` and `var_sharpe`.
10. **`n_trades` over the whole simulation** (F §D3) leaks into expectancy if full-history slicing is adopted.
    Tripwire: count trades by entry index inside the scoring window; a test against a hand-counted fold.

Limits of this protocol's evidence: the prototype covers 150 symbols, default params and raw (unmasked) signals,
with no WFO run. Old vs new WFO equality is a spec, not yet measured. Rust parity was checked on 3 configs, not the
full fuzz set (B did that on synthetic data).
