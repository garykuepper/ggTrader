# Strategy Implementation Audit: Are the NO-GOs Real, or Our Own Bugs?

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-25
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

Seventeen strategies are registered in `STRATEGY_REGISTRY`; fourteen carry
dated NO-GO reports written between 2026-07-16 and 2026-07-24. A rejection
rate that high invites an obvious question, and this audit exists to answer
it honestly: **are these strategies genuinely failing, or are we rejecting
sound ideas because of our own implementation defects?**

Three explanations had to be separated per candidate:

1. **Real signal failure** — implemented correctly, edge isn't there. Verdict stands.
2. **Implementation defect** — lookahead, wrong benchmark, missing data, or a
   strategy that silently never traded. Verdict invalid; retest warranted.
3. **Underpowered test** — correct code, but too few folds or too small a
   sample to conclude anything. Verdict should be re-labeled provisional.

**Headline conclusion: the verdict *directions* are sound, but every Sharpe
number reported for a window covering 2023 or later is understated by ~29%
because of a data-integrity bug in how the SPY benchmark is loaded (§2.0).**

The single most important sanity check is a positive control — *does this
harness ever say yes?* It does: the deployed 5-voter `EnsembleSignal` core
passes at OOS Sharpe **1.12** with gates **16/17**, and the live 3-sleeve blend
at **1.14** (`docs/research/RESEARCH_SNAPSHOT.md` §1). The harness
discriminates; it is not a reject-everything machine, which is the failure mode
that would have invalidated every verdict at once.

Supporting that: point-in-time universe construction is survivorship-safe (the
SP500 universe contains SIVB, FRC, TWTR, ATVI and other delisted names; only
10/596 symbols (1.7%) lack OHLCV, and those are mostly *renames* such as
ANTM→ELV and ABC→COR where the economic exposure continues under a new
ticker). Cost assumptions (`FEES=0.0`, `SLIPPAGE=0.0005`) are if anything
**optimistic**, especially for midcaps — meaning rejections are conservative,
not artifacts of over-charging. Every strategy applies `data.loc[:asof]`, and
weight strategies enter strictly next-bar (`data.index > asof`).

**But the audit did find real defects**, two of which are genuine bugs and one
of which materially downgrades a verdict issued yesterday. None of them
overturn a NO-GO — the two bugs both bias *in the strategy's favour*, so
correcting them would make the rejected strategies look worse, not better.
Details in §2 and §4.

## 2. Findings

### 2.0 ⚠️ SEVERE: the SPY benchmark load duplicates every trading day from 2023

**This is the most consequential finding in the audit.**

`SPY` is the **only** yfinance symbol in `ohlcv` carrying bars at times-of-day
other than the 16:00/17:00 close convention every other equity uses. It has
**826 extra rows** at `04:00`/`05:00`, beginning **2022-12-27** and continuing
through 2026 — roughly one duplicate row per trading day. (SPY also has 782
`kraken_spot` rows at 21:00, a separate ingest anomaly.) Confirmed by direct
query:

```
yfinance 1d buckets:  17:00 -> 3,656,297 rows / 1,410 symbols
                      16:00 -> 2,003,776 rows / 1,406 symbols
                      04:00 ->       521 rows /     1 symbol   <- SPY only
                      05:00 ->       305 rows /     1 symbol   <- SPY only
```

Both research entry points join SPY into the same frame as the universe —
`lab/cli.py:152` (`load_ohlcv(universe + ["SPY"], ...)`, driving every `--wfo`
and `--sweep`) and `lab/blend.py:86` (every `--blend`). SPY's columns are
dropped afterwards (`cli.py:158`), **but the damaged index is not.** The result
is a union index with two rows per trading day from 2023 on, where every
non-SPY symbol is NaN on the duplicate row:

| Rows per calendar year | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| clean (no SPY) | 252 | 251 | 250 | 252 | 250 |
| with SPY (harness path) | 252 | 255 | **462** | **504** | **500** |

**Measured impact on reported metrics.** Identical asset, identical period
(2023-01→2026-06), simple buy-and-hold AAPL through `curve_stats`:

| | rows | NaN | Sharpe | CAGR | MaxDD |
|---|---|---|---|---|---|
| contaminated (harness path) | 1,671 | 816 | **0.848** | 30.70% | -33.36% |
| clean | 855 | 0 | **1.187** | 30.70% | -33.36% |

Sharpe is understated by **29%** — precisely the 1/√2 that row-doubling
predicts (returns spread over 2× rows deflates the per-row mean/σ ratio, then
the fixed `freq="1d"` annualization cannot correct it). CAGR and MaxDD are
**unaffected**, since they depend on endpoint values and the drawdown path
rather than per-row statistics.

**Independent corroboration.** The A8 report cites the harness's SPY benchmark
at Sharpe **1.14** over 2025-08→2026-07. Measured independently from yfinance,
SPY's true Sharpe over that window is **1.61**. Ratio: 1.14/1.61 = **0.708**,
versus 1/√2 = 0.707.

**What this does and does not invalidate.**

- *Does not change stock selection.* Probing `select()` on contaminated vs
  clean frames returned **identical** picks for `max_effect` (99/99),
  `idio_vol` (99/99), `pead` (97/97) and `short_interest` (98/98) at 2023 and
  2024 dates. Per-symbol signal computation tolerates the NaN rows.
- *Does not flip verdict direction*, because the SPY benchmark is deflated by
  the same factor — both sides of every comparison shrink together, so
  "strategy < SPY" conclusions survive.
- **Does corrupt every absolute Sharpe figure** in every report covering
  2023+, and therefore any absolute threshold judgement.
- **Does plausibly cause spurious gate failures.** `dsr_check` is fed
  `observed_sr=winner["sharpe"]` (`wfo.py:683`). A systematically deflated
  Sharpe produces a deflated DSR, so folds fail the gate more often than they
  should and fall back to anchor params — which depresses aggregate OOS. This
  is a real mechanism by which the bug makes strategies look worse, and it is
  **not** symmetric with the benchmark.
- **Does completely break `pairs_stat_arb`** (§2.6), whose
  `len(sub) >= lookback*0.8` coverage guard rejects every pair once half the
  rows are NaN.

**Likely root cause:** a second ingest path (live/paper trading fetches SPY as
its benchmark) writing SPY with a different timezone normalization. Since the
`ohlcv` key includes the timestamp, a differently-normalized timestamp inserts
a new row instead of upserting the existing one.

### 2.1 Confirmed bugs (not fixed — see §5)

**A. `oos_score` is structurally always 0.00 — every WFO table ever printed is
misleading.** `composite_score()` min-max normalizes within the list it is
given, and `_min_max_normalize` returns `[0.0]` when `min == max`
(`wfo.py:199-205`). The out-of-sample score is computed as
`oos_score = composite_score(test_metrics)[0]` (`wfo.py:718`) where
`test_metrics` comes from `winner_grid = [deploy_params]` — **always exactly
one element**. So the `OOS` column in every WFO summary table is a hardcoded
`0.00` regardless of actual performance. Verified directly:

```
composite_score([{sharpe:1.44, sortino:2.0, maxdd:-5.0}])   -> [0.0]
composite_score([{sharpe:-9.9, sortino:-9.9, maxdd:-99.0}]) -> [0.0]   # identical
```

**Impact: display-only, no verdict affected.** `oos_score` is written at
`wfo.py:718/753` and read only by the renderer at `wfo.py:949` — it feeds no
decision. Verdicts rest on `oos_sharpe`, the stitched OOS equity curve, and
WFE, all computed correctly. But it means every report's per-fold "OOS" column
is uninformative, and anyone reading those tables has been reading a constant.

**B. `insider_cluster` has a real ~2-day lookahead.** The strategy keys off
`transaction_date` (`insider_cluster.py:49,116`) — the date the insider
*traded* — never `filing_date`, the date the Form 4 became public. Measured on
the 783,982 stored Form 4 rows: median filing lag **2 days**, p95 **5 days**.
The strategy therefore acts on disclosure it could not have seen for a median
of two trading days.

The contrast proves this is an oversight rather than a house convention:
`congress_trades` does it correctly, computing
`known_by = filing_date + report_lag_days` with a comment explaining exactly
why (`congress_trades.py:31-32,93-97`).

**Impact: inflates `insider_cluster`'s results → its NO-GO stands and is
strengthened.** Correcting it can only make performance worse.

### 2.2 The NDH density gate is uninformative on small parameter grids

The NDH (Neighborhood Density Hurdle) gate requires ≥85% of a peak's ±1-step
grid neighbors to be profitable. On A8's 6-combo grid (`lookback_days` × 3,
`quintile` × 2) every peak has only **2–3 neighbors**, so attainable density
values are {0, 0.33, 0.5, 0.67, 1.0} — and a 0.85 threshold is only cleared by
**1.0**. The gate degenerates into "every neighbour must be profitable," with
no ability to distinguish a robust plateau from a lucky spike.

This is exactly what failed A8's fold 2 (density 0.67 = 2 of 3 neighbours).
Larger grids don't have this problem (a 27-combo grid gives an interior peak 6
neighbours, granularity 0.167). **Not a bug — a calibration/grid-size
interaction worth documenting**, and a reason to prefer ≥3 swept parameters.

### 2.3 Benchmark mismatch: `commodity_trend` reaches a wrong conclusion

The harness hardcodes SPY as the sole benchmark (`wfo.py:780-783`; correctly
reindexed to the stitched OOS window, so the *construction* is sound). For
equity strategies that's right. For non-equity candidates it is not, and most
of those reports compensated manually — `fx_hedge_overlay` and `treasury_curve`
both ran static-baseline comparisons on their own instruments, and
`fomc_drift` explicitly states "SPY isn't the right framing."

**`commodity_trend` did not.** It compares a commodity strategy to SPY and
concludes the vol-regime filter failed because drawdown was "worse than the
equity benchmark." Measured against its actual asset class (2014-01→2026-06):

| | Sharpe | CAGR | MaxDD |
|---|---|---|---|
| commodity_trend (reported) | 0.13 | 1.0% | **-37.0%** |
| DBC (broad commodity) | 0.24 | 2.6% | -59.9% |
| PDBC (optimized roll) | 0.31 | 4.0% | -49.5% |
| GSG (front-month) | 0.11 | 0.1% | -76.9% |
| SPY (what the report used) | 0.85 | 14.0% | -33.7% |

**The NO-GO survives** — 0.13 still trails DBC's 0.24 and PDBC's 0.31. But the
report's *reasoning* is wrong in a way worth correcting: against its own asset
class the vol filter **did** work, cutting drawdown to -37.0% versus -59.9% to
-76.9% for commodity buy-and-hold. The report states the opposite.

### 2.4 Did every strategy actually trade?

A strategy that silently returns empty plans would "fail" WFO having tested
nothing. Each strategy's `select()` was called on real data across its own
universe and eval window, mirroring the harness (`eligible_at` for
point-in-time membership):

| Strategy | Probes | Non-empty | Avg positions | Verdict |
|---|---|---|---|---|
| `pead` | 11 | 11 | 91.7 | traded |
| `max_effect` | 11 | 11 | 98.8 | traded |
| `short_interest` | 11 | 11 | 98.2 | traded |
| `short_volume_ratio` | 11 | 11 | 98.4 | traded |
| `idio_vol` | 11 | 11 | 98.8 | traded |
| `congress_trades` | 11 | 9 | 165.1 | traded |
| `insider_cluster_buy` | 11 | 9 | 10.3 | traded |
| `commodity_trend` | 30 | 29 | 4.8 | traded |
| `treasury_curve` | 30 | 30 | 1.0 | traded (1-of-3 rotation, by design) |
| `fx_hedge_overlay` | 25 | 25 | 3.9 | traded |
| `fomc_drift` | 30 | 30 | 3.0 | traded |
| `index_deletion_fade` | 11 | 7 | 1.7 | thin but by design (sparse events) |
| `headline_sentiment` | 5 | 5 | **7.4** | traded, but see §2.5 |
| `pairs_stat_arb` | 11 | 3 | 2.7 | **see §2.6** |
| `retail_attention` | 11 | **0** | **0.0** | **never traded** |

**`retail_attention` produced zero positions on all 11 probes**, because
`google_trends_interest` contains **0 rows** — the backfill never completed
(Google 429 rate-limit lockout). **This is correctly documented**: the project
labels it *"Built, tested, PAUSED (not GO/NO-GO)"* in
`RESEARCH_SNAPSHOT.md:73`. No false verdict was issued. The audit confirms the
documentation rather than contradicting it.

*Residual risk worth noting:* the strategy remains live in `STRATEGY_REGISTRY`
and will silently return empty plans if anyone runs it. A guard that raises on
an empty backing table would convert a silent no-op into an obvious error.

### 2.5 A8 (`headline_sentiment`) is materially weaker than its report implies

Three independent problems, none of them a code defect:

1. **Fold count.** A8 ran on **3 folds**. Every other verdict in the book used
   **20–54** (index-deletion/max-effect/pairs 42, PEAD 40, congress/insider 40,
   commodity 50, FOMC 54, FX 38, treasury 50). One of A8's three folds
   gate-failed, leaving effectively two usable observations.
2. **Sample is unrepresentative.** The pilot backfilled the alphabetically
   first 50 midcap400 names. Sector mix vs a random 60-name baseline from the
   rest of the universe: Industrials **30% vs 17%** (~2× overweight),
   Consumer Cyclical 10% vs 15% under. This is the check the A8 report itself
   flagged as "the one cheap thing worth verifying" — now done, and it does
   show a skew.
3. **Portfolio far too concentrated to measure.** Only 50 of ~400 midcap400
   symbols had sentiment coverage, so the top-quintile bucket resolved to
   **7–8 positions** (probe: avg 7.4). Compare `pead` at 91.7 or `max_effect`
   at 98.8. A 7-stock portfolio is dominated by idiosyncratic variance, which
   mechanically explains both the wild fold-to-fold swing (0.27 / 1.44 / -0.38)
   and the NDH variance-gate failure.

**What does *not* rescue A8:** benchmark choice. It trades midcaps and was
judged against SPY, but over its own OOS window midcaps did essentially as
well (MDY Sharpe 1.39 / CAGR 22.9%; IJH 1.41 / 23.3%; SPY 1.61 / 21.6%). A8's
0.25 / 4.8% underperforms its own universe just as badly. **The direction of
the result is unfavourable and robust — the confidence attached to it is not.**

### 2.6 `pairs_stat_arb` — verdict **invalid** from 2023 onward

The probe showed positions on only **3 of 11** dates: active through 2022-11,
then completely flat from 2023-04. Tracing the pair-qualification stage
produced two conflicting measurements at the same dates, and reconciling them
is what uncovered §2.0.

- Computing `enumerate_sector_pairs` + `pairwise_correlations` on a **clean**
  frame: qualifying pairs decline *smoothly* — 1,870 (2022-06) → 2,753
  (2022-11) → 1,104 (2023-04) → 552 (2023-09) → 371 (2024-06). Genuine
  post-2022 correlation breakdown, gradual as one would expect.
- On the **contaminated** frame the harness actually builds: **0** from
  2023-04 onward.

The mechanism is now explained. `pairwise_correlations` requires
`len(sub) >= lookback * 0.8` jointly-valid rows after `dropna()`
(`pairs_stat_arb.py:55-56`). Once §2.0 injects a NaN row for every real row,
no pair can reach 80% coverage of a 126-row window, so **every** pair is
silently skipped and `corrs.get(p, -1.0)` returns the -1.0 default. Hence the
cliff to exactly zero, precisely at the 2022-12-27 boundary where SPY's
duplicate rows begin.

Ruled out along the way: the memo cache is an unbounded plain dict with a
correct key and no eviction (`pairs_stat_arb.py:106`); there is no equity data
gap (symbol counts *rise* through 2023); and the `00:00` bucket appearing in
`ohlcv` from 2023-01 is **crypto** (`kraken_spot`/`binanceus_spot`), which
never enters equity frames.

**Verdict impact: this one genuinely is invalid for the post-2022 portion.**
Unlike the strategies in §2.0 whose *selection* was unaffected,
`pairs_stat_arb` selected **nothing at all** from 2023 on — roughly the last
third of its 42-fold window was tested as a flat book. The reported OOS Sharpe
**-0.42** and MaxDD **-28.2%** therefore reflect real trading only through
2022. The NO-GO may well be correct, but it has not actually been tested on
2023-2026 and should be **re-run after the §2.0 fix** before being treated as
settled.

## 3. On-Deck Candidate Feasibility

One verdict per remaining register entry, against what this project actually has.

| Candidate | Feasibility | Evidence |
|---|---|---|
| **A2** Commodity carry | ⚠️ **Thin** | Signal is real: log(front/deferred) drift is −7.7%/yr for oil (USO/USL) and −19.8%/yr for natgas (UNG/UNL), with genuine variation (σ 0.44 / 0.76). But only ~3 distinct pairs exist (oil, natgas, GSG/DBC basket) — too thin for the paper's *cross-sectional* carry ranking, and A3 already rejected this asset class via ETF proxy. |
| **A4** Futures-basis reversal | ❌ **Infeasible** | Needs adjacent-contract (front/second-month) futures series. DB venues are `yfinance`, `kraken_spot`, `binanceus_spot` only — **zero** `=F` continuous-contract symbols. Register's own assessment confirmed. |
| **A6** ETF-implied FX expectations | ❌ **Infeasible** | Needs point-in-time AUM/flow history. yfinance exposes only a **current snapshot** (`totalAssets`, `navPrice`) with no history — the same look-ahead-biased-snapshot problem that already sank earlier candidates. |
| **A9** Direct CIP basis | ❌ **Infeasible** | Requires forwards/swaps, funding and balance-sheet capacity. Retail ETF proxy is an unvalidated different hypothesis, per the register itself. |
| **B1** Stablecoin stress | ⚠️ **Blocked (not by data)** | Data is excellent and already local: `BTC-USD` and `USDT-USD` 1-minute bars (1.57M / 2.05M rows). But bars stop **2025-12-31** (stale), and crypto trading is **dormant** (`kraken_ledger` last entry 2026-06-06). B1 is a *risk overlay* — there is no live crypto exposure to overlay. |
| **B2** FOMC country-ETF | ❌ **Infeasible** | Mechanism needs the ETF's first 15–30 min reaction. Every non-crypto symbol in the DB is **daily-only**; free intraday equity history is far too short for a multi-year backtest. Structural gap, not a backfill. |
| **B3** Bond-ETF stale NAV | ❌ **Blocked** | Needs historical NAV / premium-discount series. Confirmed absent: yfinance price history has no NAV column, and `info` exposes only a current `navPrice`/`netAssets` snapshot. Register's recommended fair-value-residual redesign is also a heavier build than anything attempted so far. |
| **B4** Crypto vol-premium (DVOL) | ✅ **Data feasible** | Deribit's public DVOL endpoint returns free daily OHLC history with no auth (verified live). *But* it needs real options mechanics, and the same crypto-dormancy caveat as B1 applies. |

**Nothing on deck is both feasible and deployable today.** The two with usable
data (B1, B4) are gated on crypto trading being dormant; the rest are gated on
data this project does not have and cannot cheaply get.

## 4. Verdict Summary

| Category | Strategies |
|---|---|
| **Direction sound; absolute Sharpe understated ~29%** (§2.0) — selection verified unaffected, benchmark deflated equally | `pead`, `max_effect`, `short_interest`, `short_volume_ratio`, `idio_vol`, `congress_trades`, `index_deletion_fade`, `leveraged_rotation_*`, `leveraged_trend_*` |
| **Direction sound, unaffected by §2.0** — non-equity universes, own benchmarks | `fx_hedge_overlay`, `treasury_curve`, `fomc_drift` |
| **Direction sound, strengthened by a bug** — lookahead inflated it | `insider_cluster_buy` (§2.1B) |
| **Direction sound, reasoning flawed** — right call, wrong benchmark logic | `commodity_trend` (§2.3) |
| **Provisional** — underpowered, should not be treated as settled | `headline_sentiment` (§2.5) |
| **INVALID for 2023+** — selected nothing; must be re-run | `pairs_stat_arb` (§2.6) |
| **Correctly unresolved** — no data, properly labeled PAUSED | `retail_attention` (§2.4) |

**One verdict is invalidated** (`pairs_stat_arb`, untested on ~a third of its
window), one downgraded to provisional (`headline_sentiment`), one has flawed
supporting reasoning (`commodity_trend`), and **every equity Sharpe figure
covering 2023+ is ~29% too low** — which changes no ranking, since the
benchmark is deflated identically, but does invalidate any absolute-threshold
reading and plausibly caused spurious DSR gate failures.

**Answering the question that prompted this audit:** the strategies are, with
one exception, genuinely failing rather than being failed by our code. But the
measurement apparatus has a real defect that has been silently deflating every
headline number for eighteen months of research, and it is worth fixing before
the next candidate is tested.

## 5. Operational Roadmap: Recommended First Action

**Fix the SPY duplicate-timestamp contamination (§2.0) before running another
backtest.** Nothing else in this list matters as much. Three parts, in order:

1. **Clean the data.** Delete or re-normalize the 826 anomalous SPY rows
   (`venue='yfinance'`, `interval='1d'`, time-of-day ∉ {16:00, 17:00}), and
   decide whether SPY's 782 `kraken_spot` rows belong in the table at all.
2. **Stop the re-contamination at the source.** Find the second ingest path
   (most likely the live/paper trader fetching SPY as its benchmark) and make
   its timestamp normalization match the research loader's, so the write
   upserts rather than inserting a parallel row.
3. **Add a guard.** `load_ohlcv` should assert one row per symbol per bar —
   a cheap `index.duplicated()` check would have caught this on day one and
   would catch the next occurrence of the same class of bug.

**Then re-run `pairs_stat_arb`'s WFO** (§2.6). It is the one verdict this audit
invalidates: it selected nothing at all from 2023 onward, so roughly a third of
its window was never really tested.

**Then fix the two lesser bugs** — neither requires re-running a verdict:

4. **`oos_score`** — either compute the OOS composite against a meaningful
   reference set, or drop the column and print `oos_sharpe` instead (which is
   what readers assume it already shows). Lowest-risk fix: change the renderer
   at `wfo.py:949` to use `r['oos_sharpe']`.
5. **`insider_cluster`** — switch the event date from `transaction_date` to
   `filing_date` (already stored, `form4_data.py:62`), mirroring
   `congress_trades.py:93-97`. Its NO-GO will not change direction.

**Do not** spend LLM budget on an A8 full-universe retest yet. The cheap
version of that decision is already made below.

**Discipline rule unchanged:** no live-config change from any of this without a
separate, explicitly approved WFO + gate pass over the live baseline.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** with 14 of 17 strategies rejected, is the real problem
that the *bar* is wrong rather than the strategies? A blend-relative hurdle
(beat OOS Sharpe 1.14 with low correlation) is demanding enough that a
genuinely useful but modest diversifier gets rejected for not clearing a bar it
was never meant to clear alone.

**Resolution — partly valid, and the snapshot already half-concedes it.**
`idio_vol` is the cleanest case: Sharpe 0.57 vs SPY 0.58, but MaxDD -17.2% vs
SPY's -22.1% and only 0.447 correlation to the deployed core — it was kept as a
"documented, non-default diversification-sleeve candidate," which is the right
call. The counter-evidence is stronger though: nine consecutive diversification
candidates were tested *as blends against the live configuration* and every one
**hurt** it (1.14 → 1.06/1.12/1.04). That is not a too-high-bar artifact; that
is a real finding about correlation-capped diversification. The bar is
defensible.

**The more actionable concern is statistical power, not the bar.** The project
has genuinely good multiple-testing machinery — DSR (Bailey & López de Prado)
deflates each fold's Sharpe by `n_trials`, and `event_study.py` runs a proper
Welch t-test. What is missing is any confidence interval on the *final*
strategy-vs-SPY comparison; verdicts compare point estimates. At 40+ folds that
is harmless. At A8's 3 folds it is not.

### Parked: A8 full-universe retest

Gate to reopen: backfill the remaining ~350 midcap400 symbols so the
top-quintile bucket holds ~80 names instead of 7, and extend the window to
yield ≥20 folds. Cost is tens of thousands of LLM calls and many hours. **Not
recommended now** — the authors' own crowding/decay warning means a 2026
backfill tests an even weaker version of the effect, and the direction of the
current result (0.25 vs MDY 1.39) is unfavourable even if its confidence is
low. Revisit only if a cheaper sentiment source removes the LLM cost.

### Parked: `retail_attention`

Gate to reopen: a completed Google Trends backfill (currently 0 rows, 429
lockout). Until then it is correctly PAUSED, not rejected — and should arguably
raise rather than silently no-op.
