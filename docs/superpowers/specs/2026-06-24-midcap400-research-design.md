# MidCap 400 Research (Bias-Quantified) — Design Spec

**Date:** 2026-06-24
**Status:** Design (pending implementation plan)
**Author:** brainstormed with Claude

## Problem

The 5-voter ensemble + 3% exposure beats SPY outright on the S&P 500
(OOS CAGR 16.2% / Sharpe 1.09 / DD -11%). The S&P MidCap 400 is plausibly
*more* fertile for mean-reversion — less institutional arbitrage than large caps,
more liquid than small caps (roadmap research direction G). **Goal: determine
whether the same strategy is deployable on the MidCap 400.**

The blocker is data. A clean, long-history **point-in-time** S&P MidCap 400
constituents dataset does not exist publicly (fja05680 is SP500-only;
yfiua/index-constituents has no MidCap 400 and only starts ~2023; other sources
are current snapshots). True-PIT rigor would require building membership history
ourselves or buying it. **Chosen approach: run on a current snapshot, but
*measure* the survivorship bias rather than ignore it** — a deployable-grade read
with an explicit, calibrated bound.

### MidCap-specific bias nuance

S&P 400 churns in *both* directions: the biggest winners **graduate up** to the
S&P 500, and losers **drop down** to the S&P 600 or get acquired. A current
snapshot therefore omits past *winners* as well as losers, so the survivorship
bias may be *smaller or even downward* — unlike the usual upward inflation. This
makes a bias-bounded snapshot more defensible here than for a typical universe,
but the bias is still real and must be quantified.

## Objective (success criteria)

On the 17-fold WFO (`run_wfo`, rolling 12mo/3mo, `eval_start=2021-01-31`),
report the 5-voter + `SIGNAL_POSITION_SIZE=0.03` result on `midcap400`, against:

- **MDY** (SPDR S&P MidCap 400 ETF) — the primary benchmark
- **SPY** — cross-reference (does midcap reversion beat large-cap buy-and-hold?)
- the **static SP500 5-voter+0.03 result** — strategy portability check

The deliverable is a **verdict with a measured bias bound**: does midcap reversion
beat MDY after applying the calibrated survivorship haircut, and how does it
compare to the large-cap edge?

## Components

### 1. MidCap 400 snapshot + universe wiring

- Source the current S&P 400 ticker list (Wikipedia "List of S&P 400 companies"
  or an equivalent maintained source), yfinance-normalized.
- Commit as `data/universe/midcap400_tickers_snapshot_<YYYY-MM-DD>.txt`
  (newline-delimited, same format as the nasdaq100/russell2000 snapshots).
- Register `"midcap400"` in `_SNAPSHOT_REGISTRY`
  (`src/ggTrader/data/core/index_constituents.py`) and in `UNIVERSE_CHOICES`
  (`src/ggTrader/lab/cli.py`).
- No new lookup logic: `universe_members_asof` already routes any non-sp500
  universe to `snapshot_members`, so `--universe midcap400` flows through
  `equity_universe_between` / `eligible_at` unchanged.

### 2. OHLCV backfill

- Backfill the ~400 MidCap 400 symbols **plus MDY** into TimescaleDB via
  `scripts/equity_backfill.py` (extend it to accept a `--universe` arg if it is
  currently SP500-hardcoded; otherwise feed the midcap list).
- `coverage_stats()` reports per-period missing symbols. Smaller-cap names have
  more yfinance gaps than large caps — record the coverage so the result carries
  its data-quality caveat.

### 3. Survivorship-bias calibration

The methodological core. A one-off harness quantifies the snapshot-vs-PIT gap on
a universe where we have *both*:

- Run 5-voter + 0.03 WFO on **SP500 PIT** (the existing path; ≈ known
  16.2% / 1.09 result).
- Run the identical WFO on the **current SP500 snapshot**
  (`data/universe/sp500_tickers_snapshot_2026-06-09.txt`, loaded as a fixed
  member list — add a thin snapshot route for sp500 in the harness, NOT in the
  production `universe_members_asof`).
- `Δ = snapshot_metrics − pit_metrics` (CAGR and Sharpe) = the large-cap
  survivorship inflation under this exact strategy.
- This Δ is the **haircut** applied to the midcap-snapshot result, reported with
  the two-way-churn caveat (true midcap bias likely ≤ Δ, possibly negative).

### 4. MidCap research run + verdict

- Run 5-voter + 0.03 WFO on `midcap400`, benchmark = **MDY** (pass MDY's close
  as the benchmark series to `run_wfo`, where SPY is passed today — a call-site
  change, no refactor).
- Also report the SPY cross-reference.
- Apply the calibrated haircut; state the verdict against the objective.
- Delivered as an analysis script writing a findings table; record the verdict
  into `docs/roadmap.md` research direction G.

## Out of scope (YAGNI)

- Building true PIT MidCap 400 membership (effort) or buying it (cost) — only if
  the bias-bounded snapshot looks promising enough to warrant deployable rigor.
- Regime-gating, conviction/Kelly sizing — separate research directions.
- S&P 600 small-cap universe — a possible follow-on once the midcap method is
  proven.

## Testing

- **Unit:** snapshot loader resolves `midcap400` (registry + file present, tickers
  normalize); `UNIVERSE_CHOICES` includes it.
- **Integration:** `equity_universe_between("...", universe="midcap400")` returns
  the expected member count; a small-universe WFO smoke run completes.
- **Validation:** the calibration harness produces a finite Δ on SP500; the
  midcap run produces an OOS table vs MDY. These are research outputs, not unit
  tests — correctness evidence is clean end-to-end runs with sensible numbers.

## Risks

- **Data coverage** — smaller-cap yfinance gaps could thin the tradeable universe;
  `coverage_stats()` must be reported, not hidden. If coverage is poor (<~85%),
  the result is unreliable and that is itself a finding.
- **Bias haircut is an estimate** — the large-cap Δ is a proxy for the midcap
  bias, not an exact measure; the two-way-churn argument bounds its direction but
  not its magnitude. Report it as a bound, never as a precise correction.
- **Benchmark availability** — MDY history must backfill cleanly; if not, fall
  back to IJH (iShares Core S&P Mid-Cap, same index).
