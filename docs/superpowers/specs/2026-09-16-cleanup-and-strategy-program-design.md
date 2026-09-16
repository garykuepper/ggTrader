# Cleanup and Strategy-Improvement Program — Design

**Date:** 2026-09-16
**Status:** approved design, awaiting implementation plan
**Supersedes:** the ordering in `docs/next_steps.md` ACTIVE STEP (2026-09-10);
reuses `docs/superpowers/plans/2026-09-11-paper-trading-remediation.md`
Tasks 1–2 verbatim for the split-state work.

## 1. Why

Twelve weeks of live paper trading (2026-06-24 → 2026-09-15) returned
-0.20% as-reported / +0.71% split-fair against SPY +3.29%. The account is
62% passive SPY sweep, 33% active strategy, 5% cash. Since the sweep went
in on 8/24 the account has fallen faster than SPY, so the active third is
the part losing money.

Root causes, all previously identified and all still unfixed as of this
date:

- The live trader still runs the 3-sleeve blend (`generate_blended_signals`),
  which three independent WFO measurements show underperforms the
  standalone SP500 core (0.69 vs 0.99 Sharpe on the corrected tape).
- The blend overlay sizes each slot at 3.3% × sleeve weight (~0.30) ×
  scale (0.695) ≈ 0.7% of the account (~$730), versus 3.3% (~$3,400) when
  the core ran standalone in June. The validated strategy governs a third
  of the book at a quarter of its intended size.
- `_SPLIT_LOOKBACK_DAYS = 14` expired while MNST is still held; MNST
  displays -50.7% unrealized (true ≈ -5%). Arming the catastrophe stop
  today would force-sell it on a fictitious loss.
- The cash sweep round-trips SPY inside a single session (9/15: sold $673,
  bought $786).
- The benchmark tape is dead: no SPY bars after 2026-08-21, TLT/GLD/DBC
  stop at 2026-07-20. No "vs SPY" lab run is citable until fixed.
- `scripts/daily_pnl_report.sh` calls a deleted CLI subcommand; a crash
  before the notifier's first call pages nobody.
- `PaperTrader.run()` is 565 lines (`trader.py:386-951`); the strategy
  registry carries 36 entries of which ~22 are closed NO-GOs with their
  own data pipelines, backfill scripts, and tests.

## 2. Target

**Beat SPY on total return at lower drawdown.** On the pinned window
2021-01-31 → 2026-04-30 at leverage 1.0, a GO requires:

| Metric | Must beat |
|---|---|
| CAGR | SPY 12.8% |
| MaxDD | SPY -22.1% (i.e. shallower) |
| Sharpe | core 0.99 (at or above) |

Reference numbers: `docs/research/_rebaseline_corrected_tape_20260822.json`
(core 0.99 / 8.0% / -7.65%, 12/17 gates; SPY 0.78 / 12.79% / -22.09%).

## 3. Structure: two parallel tracks, one shared gate

- **Ops track** touches the live trader, one deploy at a time.
- **Research track** touches only the lab and never the live path.
- **Prune** rides on the research track after the candidate list is final.
- **Shared gate:** nothing from research deploys until the ops track has
  landed the core revert and the paper/ refactor. A research GO becomes a
  new ops-track deploy at that point.

Live trading keeps running throughout. The standing rule holds: one live
change per deploy cycle so attribution stays clean.

## 4. Ops track

Each step is its own commit, deployed via push → CI image
(`.github/workflows/docker-build.yml`) → `docker compose pull && up -d` →
verified on the next 12:45 PT run before the next step starts.

### 4.1 Core revert (first — the only step that changes returns)

- Add `generate_core_signals()` to `src/ggTrader/paper/signal_runner.py`:
  returns the SP500 sleeve alone with `weights={"sp500": 1.0}`,
  `scale=1.0`, same dict shape as `generate_blended_signals()`.
- Switch the single call site at `trader.py:412`.
- Existing midcap/nasdaq positions exit on their own signals; no
  force-sell.
- **Acceptance:** next live run logs `Scale: 1.00x`, new buys size near
  $3,400 (3.3% of PV), no midcap/nasdaq symbols in buys.

### 4.2 MNST split-state persistence

Reuse `docs/superpowers/plans/2026-09-11-paper-trading-remediation.md`
Tasks 1–2 unchanged:

- `paper_split_state` table (symbol, ex_date, factor); persist a correction
  until the broker quantity changes or the position closes. Removes the
  14-day lookback.
- One-off script restates snapshots 2026-08-26 → present.
- **Acceptance:** MNST shows ≈ -5% unrealized in the daily summary; NAV
  restated.

### 4.3 Sweep hysteresis

- Dead band in `src/ggTrader/paper/cash_sweep.py`: sweep buy only when
  cash > 8% of PV, sweep sell only when cash < 2% of PV. Env-overridable
  like the existing reserve.
- **Acceptance:** no same-session SPY sell-then-buy in the log.

### 4.4 Catastrophe stop armed

- `CATASTROPHE_STOP_ENABLED=true` in `.env`, container recreated. Only
  after 4.2 is verified live.
- **Acceptance:** log shows armed, zero triggers (worst true position is
  well inside -25%).

### 4.5 Alerting

- Replace `scripts/daily_pnl_report.sh` body with a SQL-over-
  `paper_snapshots` summary sent via `~/scripts/notify.py`.
- Wrap the `paper_trade.sh` cron entry so a nonzero exit sends a Telegram
  message with the last 20 log lines.
- **Acceptance:** kill a dry run mid-way and receive the alert.

Steps 4.1 and 4.2 are independent; 4.3 and 4.5 may interleave anywhere;
4.4 depends on 4.2.

## 5. paper/ refactor behind a parity harness

Starts only after 4.1–4.3 have landed, so the parity baseline is the core
config.

### 5.1 Target shape

`PaperTrader.run()` becomes a short pipeline over a `RunContext` dataclass
(today, raw positions, corrected positions, portfolio value, cash,
signals, slot caps, errors). Each stage is a function in its own module
that takes and returns the context:

| Module | Responsibility (moved out of `run()`) |
|---|---|
| `positions.py` | fetch broker positions; apply split corrections and dividend accruals |
| `sizing.py` | slot caps, slots available, per-position notional |
| `execution.py` | sells → sweep sell → buys → sweep buy; order polling; pending-order reconcile |
| `reporting.py` | snapshot persistence; Telegram summary |

Broker access stays behind `alpaca_broker.py`. Existing small modules
(`cash_sweep`, `split_check`, `dividend_check`, `catastrophe_stop`,
`risk`, `overlay`, `feature_gate`, `persist`, `notifier`) are unchanged
except for import paths. Every DB call keeps the fail-soft pattern.

### 5.2 Parity harness

- `scripts/paper_parity.py`: runs the old `run()` and the new pipeline in
  dry-run against the same broker snapshot and asserts identical intended
  orders (symbol, side, notional to the cent) and identical snapshot
  fields.
- Must be green on three consecutive market days before merge.
- The old `run()` is deleted in the same PR that passes parity, so two
  live code paths never coexist.

### 5.3 Constraints

- No function in `src/ggTrader/paper/` over 100 lines after the refactor.
- Test count in `tests/paper/` does not drop; new stage functions get
  their own unit tests using the existing mock patterns.
- `ruff check .` and `ruff format .` clean.

## 6. Research track

### 6.0 Conventions for every run

- `--eval-start 2021-01-31 --eval-end 2026-04-30 --max-leverage 1.0`.
- Raw results to JSON in `docs/research/`, next to a dated report from
  `docs/research/TEMPLATE-research-report.md`.
- Judged against §2. Anything short is a written NO-GO.
- Regenerate `docs/research/RESEARCH_SNAPSHOT.md` with the
  `research-snapshot` skill after each report.

### 6.1 Step 0 — tape restore (blocks everything else)

- Add a benchmark/ETF symbol list (`SPY`, `TLT`, `GLD`, `DBC`, `IEF`) to
  the nightly path of `CachedYFinanceLoader` so they stay current.
- One-off backfill from the last good bar. Writes must use naive UTC
  (`data/live/cached_yfinance_loader.py:215-226`).
- **Acceptance:** `max(timestamp)` for each symbol equals the latest
  trading day; day-of-week census shows no Sat/Sun bars.

### 6.2 Step 1 — constrained `ensemble_ic`

- Add a `min_weight` floor parameter (default 0.10 per voter) to
  `EnsembleICSignal`; include it in the WFO sweep grid.
- Prior evidence: Sharpe 1.01, CAGR 19.6%, MaxDD -17.3%, 3/17 stability
  (vs phantom 1.12 baseline).
- **Kill criteria:** MaxDD worse than -10% or winner stable in fewer than
  8/17 folds.

### 6.3 Step 2 — `xs_momentum` and `dual_momentum`

- One pinned run each; no code changes expected. The only prior lab runs
  (2026-06-24) never finished (`status='running'`).

### 6.4 Step 3 — diversifying blend study

- Build `cross_asset_trend` (TLT/GLD/DBC, unleveraged, time-series
  momentum with inverse-vol scaling) as a registered weight strategy.
- Run `ggt lab --blend` with the core plus each sleeve in turn:
  `cross_asset_trend`, `idio_vol` (measured 0.447 correlation to core,
  MaxDD -17.2% vs SPY -22.1%), and whichever of 6.2/6.3 passed.
- Leverage 1.0 only. Blend GO requires improving on the core's Sharpe
  **and** meeting §2.

### 6.5 Step 4 — `fomc_drift` re-run

- Existing strategy on corrected tape, true pre-announcement bar. Trivial
  compute; done for completeness.

## 7. Prune

Runs after §6's candidate list is final so nothing a blend study still
needs is deleted.

### 7.1 Registry after prune (14 of 36)

Live: `ensemble`, `bb_reversion`, `rsi_reversion`, `ema_cross`,
`macd_divergence`, `volume_bb_reversion`.
Retry / study: `ensemble_ic`, `xs_momentum`, `dual_momentum`, `idio_vol`,
`fomc_drift`, `retail_attention`, `cross_asset_trend` (new).
Harness plumbing: `wfo_tournament`.

### 7.2 Removed (22)

`mtf_reversion`, `overnight_gap`, `ensemble_kelly`, `conviction_bb`,
`ensemble_conviction`, `congress_trades`, `index_deletion_fade`,
`insider_cluster_buy`, `max_effect`, `pairs_stat_arb`, `pead`,
`short_interest`, `short_volume_ratio`, `fx_hedge_overlay`,
`headline_sentiment`, `commodity_trend`, `treasury_curve`,
`leveraged_rotation_{sp500,nasdaq100,russell2000}`,
`leveraged_trend_{sp500,nasdaq100,russell2000}`.

Each removal takes its strategy module, data-pipeline module, backfill
script under `scripts/`, and tests together, so nothing orphaned remains.

### 7.3 Method

- Delete, do not archive: an archive package still has to import, lint,
  and test, which is the cost being shed.
- Git tag `pre-prune-2026-09` on the last commit with everything present;
  one line in `RESEARCH_SNAPSHOT.md` §5 records the tag.
- Research reports and JSON artifacts under `docs/research/` untouched.
- Database tables from the backfills stay (never-delete-data rule).
- Update `docs/cli_reference.md`, `docs/architecture.md`, `AGENTS.md`
  registry count, and `docs/changelog.md`.

## 8. Success criteria

- **Ops:** live account runs the SP500 core at 3.3% slots; MNST accounting
  correct; catastrophe stop armed; a crash pages Telegram.
- **Refactor:** no paper/ function over 100 lines; parity green three
  consecutive days; test count not reduced.
- **Prune:** registry at 14 entries; `ruff` clean; full suite green.
- **Research:** four dated reports with JSON artifacts. A GO meets §2. If
  nothing passes, the recorded outcome is "core plus sweep, no new
  sleeve."

## 9. Out of scope

- Crypto path (parked by decision).
- Any candidate in the do-not-retry register
  (`docs/research/2026-09-10-comprehensive-strategy-audit-and-retry-recommendations.md`
  §6), including `ensemble_kelly`.
- Leverage above 1.0.
- Restoring the 20.8 missing MNST shares broker-side (accept and document
  the paper-venue loss).
