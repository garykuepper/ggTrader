## 8. Trader rebuild

### 8.1 Situation (owner, 2026-11-06)

- The paper trader is **stopped**. The 12:45 PT `scripts/paper_trade.sh` cron line and the 06:00
  `daily_pnl_report.sh` cron line are commented out in the user crontab. Nothing else runs or
  auto-updates the old trader: the nightly updater does not scan `/home/flynn/ggTrader` and there
  is no watchtower.
- The owner closed the stock positions (ADM, ALB, AMZN sold at market). **AVB** could not be closed:
  the ticker is inactive after the EQR merger and the account still shows 4.05 shares (OD17).
  About 114 **SPY** shares remain.
- The intraday SPY observer timers and the Hermes "Intraday diary supervision" job are a separate
  research study that places no orders. They keep running and are not part of the trader.

Consequences for this program:
- With no live run to protect, **work merges to `main`**. There is no frozen lane, no long branch
  and no second image lane.
- Files shared with the old trader (`paper/*`, `lab.strategies.ensemble`, `strategies/indicators.py`,
  `lab.strategy`, `lab.data.fetch_stock_ohlcv`, `lab.persist.get_engine`, the loader chain,
  `pyproject.toml`, `Dockerfile`, `docker-compose.yaml`) **may be changed or deleted**.
- The old pandas indicators are **deleted once the new ones are validated** (end of P5, section
  5.1). Tag `legacy-engine-vbt0285` and the legacy venv lock (section 7.5) still reproduce closed
  research and the old trader's signals.
- The live path today calls no vectorbt function (`trader.py:519` -> `generate_core_signals` ->
  pandas indicators through `EnsembleSignal`; vectorbt is imported only via `overlay.py:12` ->
  `lab.simulate`). The rebuild is therefore new signal code, not an engine swap.

### 8.2 What is built (P7, after section 7 validation)

- `paper/signal_runner.py` calls the same `Ensemble.build()` used in research, with a
  one-combination grid of the live defaults (OD14), and extracts `buys`, `sells` and `last_exit`
  from the last bar. The live golden (7.3 item 4) is its acceptance test: buys, sells and
  `last_exit` over 120 sessions match the validated lab strategy on the same frame, and any
  difference from the legacy signals is pre-registered and explained.
- Order selection keeps the existing rules unless the owner changes them: buys sorted
  alphabetically (`signal_runner.py:100`) and the position cap `MAX_POSITIONS` from the environment
  (`risk.py:8-28`, code default 30; the changelog says `.env` sets 3; `.env` not read). One extra
  early-alphabet buy changes which names fill the slots, so the dry-run check compares intended
  orders after the cap, not only raw signals.
- `overlay.py` stops importing `lab.simulate` (removes the 2026-09-23 crash class).
- The loader chain moves to `data/live/` (section 5.7). Before any trader run that writes
  `ohlcv` (`_cache_to_db`, `refresh_benchmark_tape`), a scratch-Postgres round-trip test under
  pandas 3 proves timestamps keep their unit and stay naive UTC (UNVERIFIED today, S3). This
  protects the shared tape, not a live run, so it stays.
- AVB is handled per OD17; the trader must not try to trade it.

### 8.3 Safeguards kept (cheap engineering practice)

- **Fail closed.** `selfcheck()` runs before the data fetch; an input sanity gate (row count,
  universe size, NaN share on the last bar, duplicate or weekend days, absurd buy/sell counts;
  bounds ESTIMATE, tuned from the dry run) and any exception in signal generation mean **no orders**
  and a notifier message. There is no fallback engine and no legacy fallback path. If signals are
  ready after 12:55 PT, no orders (a DAY order after the close queues to the next open).
- **Image smoke test** in the Dockerfile and `docker-build.yml` before push: `selfcheck()` as root
  and as `appuser` asserts the Rust engine is dispatched (wrapped `simulate_from_signals_rs` call
  count), vectorbt and vectorbt-rust share major.minor, the mismatch warning is an error,
  `pip check`, `import ggTrader.paper.trader`, `ggt.py paper --help`.
- **Immutable image tags for rollback.** The trader runs an immutable sha tag, never trusts
  `:latest`; the previous tag is kept locally until the new one has run cleanly. Rollback is
  selecting the previous tag.
- **Cron stays disabled** until the owner re-enables it (OD18). No agent edits the crontab.

### 8.4 Dry run, then re-enable

1. After P7's tests and the live golden pass, run `ggt paper` **without `--live`** (the existing
   dry-run mode, which logs intended orders) for a short period the owner chooses, by hand or as the
   owner directs.
2. Before the first dry run, confirm what dry-run mode writes. A code read (`trader.py:497-527`,
   `:675-680`, `:1086-1095`) suggests it still initialises the paper schema, refreshes the
   benchmark tape (an `ohlcv` write), saves the peak value, writes a `paper_snapshots` row and sends
   Telegram messages labelled "DRY RUN" (UNVERIFIED end to end). Record which of these happen, so
   dry-run and live behaviour differ only in order submission (risk 14).
3. Each dry-run session: the run completes; the Rust probe passes; intended buys and sells after the
   cap equal the lab strategy's signals on the same frame; wall time and peak memory are logged.
4. The owner reviews the dry-run record and, only on the owner's explicit say-so, re-enables the
   cron lines. The first live run is checked with `daily-trader-check`. No date schedules this.

## 9. Phased plan

Effort is agent-hours, ESTIMATE (+/-40%); wall time for oracle runs is extra. Every phase merges to
`main`. No phase places orders or edits the crontab.

| Phase | Entry | Work | Exit criteria | Artefacts | Decides | Effort |
|---|---|---|---|---|---|---|
| **P0 Save research and legacy references** | spec approved | copy the S1-S8 reports, evidence scripts and prototypes into the repo (Appendix D Q5); legacy tag + `requirements-legacy.lock` + `legacy_env.sh`; frozen Parquet tape + sha; 17-fold and 2-fold reference re-run on legacy; per-strategy legacy references (Appendix D Q8); live golden (7.3 item 4); fix the changelog defect | 17-fold reproduces 0.592 / 17-of-17 or a documented fresh legacy baseline replaces it | `_legacy_reference_*.json`, tape sha, golden, saved reports | agent | 12 |
| **P1 Lock and pandas-3 stack** (pandas 3, numpy 2.5, vbt 1.2.0 + Rust, **legacy indicators**) | P0 | drop `pandas-ta` (OD15); `uv.lock`; `event_study` `.dropna()`; `.copy()` after concat; 44 `pct_change` sites (including `paper/feature_gate.py:50`, `paper/overlay.py:88`) with gap tests on 2.3.3 first; deprecations; `"1D"` | full suite green (S3: 1770/1770 after 2 fixes); signal diff A->B 0; 2-fold and 17-fold bit-for-bit except documented 1-ulp DSR; numba vs Rust bit-identical | `_p1_*.json`, signal report | agent | 14 |
| **P2 Engine adapter and Rust** | P1 | `engine/` config, selfcheck, provenance, `_compat`; presets, signals, portfolio (+chunker), stats (+guard, DSR), windows; boundary lint; L0-L3 tests; CI on the lock; image smoke step; watch job; Dependabot; runbook skill | all L0-L3 green on both engines; Rust dispatch probe passes; boundary test has only the legacy importers allow-listed | tests, workflow, skill | agent | 37 |
| **P3 Data and universe layer** | P2 | `data/bars.py`, `data/snapshot.py`; `lab.universe` (PIT mask, removal and terminal exits) | PIT mask 0 mismatches vs `eligible_at`; no research write to `ohlcv` | mask parity record | agent | 5 |
| **P4 Strategy port, `ensemble` first** | P3 | `vote`; new `ensemble` (OD1(a), OD3(a), OD14 control row); then the rest of the OD11 set; archive the others at the legacy tag | ensemble signal diff A->D within class (ii); each ported strategy reproduces its legacy result within class (ii) or carries an attributed diff | per-strategy JSON, tie audit | agent | 30 |
| **P5 WFO rewrite** (section 5.4: `generate_folds` mapped to bar positions, single-pass signals, `range_split` stacked windows for weights) | P4 | single-pass signal WFO; stacked weight WFO; per-window trades; compatibility shims (Appendix D Q3) removed at the end; delete legacy lab plumbing and the old pandas indicators once E1 is accepted | 17-fold "consistent" (7.4); peak RSS under 6 GB; wall time recorded | `_p5_*.json`, memory record | agent proposes; **owner accepts** (it defines E1) | 15 |
| **P6 Validation: definition changes and re-baseline** | P5 | one run per change: Sortino, CAGR, (BB ddof=0 optional), PIT refresh (OD5), terminal exits (OD7), exit-on-removal (OD6), trade-count definition (OD12); then final references, `ENGINE_PROVENANCE.json`, banners, snapshot Engine column, docs | each change attributed and accepted or rejected by the owner; owner accepts E2; any verdict change vs SPY has an attributed cause | `_p6_<change>_*.json`, engine-migration report | **owner per change** | 22 |
| **P7 Trader rebuild and dry run** | P4 ensemble validated + P5 E1 accepted (P6 changes the trader only if the owner accepts a change that affects the ensemble) | section 8.2-8.4: signal path on `Ensemble.build()`, fail-closed checks, sanity gate, image smoke, immutable tags, `ggtrader-deploy` and `daily-trader-check` updates, tests, dry-run review | live golden passes; every dry-run session meets 8.4 item 3; owner reviews the record | dry-run log, deploy record | **owner** (re-enabling cron, OD18) | 14 |
| **P8 First upgrade drill** | P2 | run the runbook on the next vectorbt release (1.2.1 is in upstream master) | ACCEPT/RE-BASELINE/REJECT recorded | upgrade log row | agent; owner if results change | 4 |

**Total ~153 agent-hours (ESTIMATE, +/-40%).** The earlier ~178 (rounded to ~180) less the hours
that existed only to protect a running trader: the shadow phase (30) and the staged cutover D0-D3
(11), 41 in all; plus the trader rebuild and dry run (14, of which about 11.5 are the parts of S7's
43.5-hour live plan that survive: selfcheck and sanity gate, tests, skills and docs, image and tag
work, cleanup) and 2 for saving the research artefacts in P0. S7's own breakdown puts about 32 of
its 43.5 hours on protection-only work (shadow harness, offline O-vs-N replay, forward shadow,
authorization asks, rollback pull-test, the D1/D2 probation steps and the engine flag/crosscheck);
the spec's shadow phase had been costed higher than S7's. Re-enabling the trader has no date.

**Stop conditions (any one halts the phase and goes to the owner):** P0 legacy reference does not
reproduce and no fresh legacy baseline can be documented; P1 any numeric difference beyond the
documented 1-ulp DSR; Rust probe returns 0 calls; P4 or P5 an unexplained post-burn-in flip or a
winner/gate flip in the configured migration; any run killed at the memory cap twice after
chunking; any write to `ohlcv` from a research path; any dry-run failure, or any unexplained
difference between dry-run intended orders and the lab signals on the same frame; a new vectorbt
release that cannot be locked without loosening an engine bound.

