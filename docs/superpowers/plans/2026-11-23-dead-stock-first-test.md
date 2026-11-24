# Dead-Stock First Honest Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, once and honestly, whether a plain momentum rule over every liquid NYSE and Nasdaq common stock, dead ones included, beats SPY after trading costs and taxes, and how much that answer moves when dying stocks return 100%, 50% or 0% of their last price.

**Architecture:** One new study folder (`research/stock-signals/dead-stock-first-test/`) reads the published combined release 1 (`load_bars(source="combined")`) plus a point-in-time eligibility table built from the raw CRSP rows (`shrcd`, `exchcd`, price, shares). A small monthly replay trades a fixed rule through a `TaxLotBook` ledger, applies the lab's own dead-stock exit and loss (`ggTrader.lab.dead_stocks`), and compares with SPY run through the identical ledger. Rule and numbers are frozen in a brief before any return is looked at; a development window is explored, a holdout window is run once.

**Tech Stack:** Python 3.12, pandas, psycopg2 (read-only via `connect_readonly`), `ggTrader.lab.dead_stocks`, `ggTrader.lab.strategies.xs_momentum.momentum_scores`, `research/paper-system/tax-lots/code/tax_lots.py` (`TaxLotBook`), `scripts/study.py`, `scripts/run_capped.sh`.

**Spec:** No separate spec. This plan is its own brief. Context: `docs/superpowers/specs/2026-11-22-kaggle-import-design.md`, `docs/kaggle_import_runbook.md` ("Known limits of version one"), `docs/superpowers/specs/2026-11-21-all-stocks-honest-test-bench-design.md` (the wider data programme; this plan scores one fixed rule and builds none of that programme's SEC universe).

## Global Constraints

- Free data only. No paid source. The paper trader stays stopped. Nothing is pushed, merged or deployed without the owner saying so.
- The four plain rules (`CLAUDE.md`, 2026-11-20): test on data the strategy never saw while tuned; use only what was known on each day and include stocks that later died; subtract costs and taxes and compare to SPY over the same dates; paper trade first. No old Part A bars, no GO/NO-GO labels.
- Every statement to the owner is in plain words. Define any term in the same sentence (for example "dead-stock loss: the share of its last price a holder loses when a stock stops trading").
- Pin both window ends. Development window: `2005-01-03` to `2019-12-31`. Holdout window: `2020-01-02` to `2024-12-31`. `2024-12-31` is the last CRSP day; `run_wfo` refuses later ends, and this study reads past it nowhere.
- Release: `release_id=1`, pinned in every call (never "newest"). Record `data_release` and `bars_sha256` in every JSON.
- Study code lives in the study's `code/` folder; nothing in `src/` imports it. The study never imports vectorbt. `bootstrap(__file__)` first in `run.py`; `main()`, `argparse`, `--help`, `__main__` guard, no work at import.
- Heavy runs only through `scripts/run_capped.sh` (3 GB cap, shared lock); tests only through `scripts/run_tests.sh <small paths>`; never two heavy jobs at once; exit 75 means busy, not failed. Never open `.env`; no builder connects to `ggtrader` except through `connect_readonly` in the scale probe and the driver, which are run by the orchestrator. Memory and OOM slice settings belong to the owner: do not change them.
- Ruff clean (`line-length = 100`). Raw results to JSON beside the report. Honest results are reported plainly whatever they say.

## Fixed choices (the pre-registration; Task 1 writes them into the brief unchanged)

| Item | Value |
|---|---|
| Rule | 12-1 momentum: score = close 21 trading days ago ÷ close 252 trading days ago − 1 (`momentum_scores(close, lookback=252, skip=21)`); hold the 50 highest scores, equal weight; decide on the last trading day of each month, trade at the **next** trading day's close. No parameter is tuned. |
| Universe A (primary) | common stock (`shrcd` in 10, 11) on NYSE or Nasdaq (`exchcd` in 1, 3), previous month-end absolute price ≥ $5, and previous month-end market value (price × shares) rank ≤ 1,000. |
| Universe B (stress, Task 8) | same, but price ≥ $1 and no rank limit. |
| Dead-stock loss | 1.0, 0.5, 0.0 (a stock that stops trading is sold on its last bar at last price × (1 − loss); price floor `SIM_PRICE_FLOOR`). Loss 1.0 is the headline, because it is the cautious reading. |
| Costs | per-side 6.4, 18.3 and 40 basis points (6.4 and 18.3 are the AQR library figures for large and small US stocks; 40 is a labelled assumption; set by Task 1) (a basis point is 0.01 percent) of traded value, charged on every buy and sell, dead exits included. These are **labelled assumptions**; Task 1 attaches the gg-library citations that bound them, and the owner may change them before Task 2. |
| Taxes | `TaxParams(st_rate=0.24, lt_rate=0.15)` (the registered standalone rates) and a high set `TaxParams(0.35, 0.20)`. Tax assessed each year-end and paid from the book's own cash; a dead-stock exit realises its loss. Both books are fully sold on the last day and the tax paid. |
| Benchmark | SPY bought on the first day and sold on the last, through the same book, costs and tax rates. |
| Start cash | 100,000 |
| Not modelled (stated in the report) | dividend tax (prices are total-return), margin or shorting, stocks that left before 1968, spreads wider than the cost grid. |

## Review Focus

Failure modes the fixed choices imply but no obvious task covers; each has a named test in the owning task.

1. A stock held when it is bought out or merged (the release marks `has_successor`): it must keep its price path, not be written down. (Task 3.)
2. CRSP prints a negative price when it is a bid-ask midpoint; market value and the $5 floor must use the absolute price. (Task 2.)
3. A company counted twice: a demoted Yahoo link and its CRSP placeholder are the same company; at most one column per company may be eligible on any day. (Task 2.)
4. A stock that changes exchange or share class mid-history must be eligible only on the days it was on NYSE/Nasdaq common stock. (Task 2.)
5. A held stock with no print on the rebalance day (halted) must be valued at its last print, and a dead stock's realised loss must reach the tax ledger in the year it died. (Tasks 3 and 4.)

---

### Task 1: Scaffold the study and freeze the brief

**Files:**
- Create: `research/stock-signals/dead-stock-first-test/` (via `scripts/study.py`), including `code/run.py`, `code/tests/`, the brief `2026-11-23-brief.md`
- Modify: `research/README.md` index row only if `study.py` prints one for this folder

**Interfaces:**
- Produces: the committed brief that Tasks 2 to 8 obey; `study.toml`.

- [ ] **Step 1:** Run `.venv/bin/python scripts/study.py new stock-signals dead-stock-first-test --kind screen --placement 9`.
- [ ] **Step 2:** Fill the brief from `research/TEMPLATE-brief.md` with the "Fixed choices" table above, verbatim values, the two windows, and the plain-words success statement: "A result is reported at every loss level and cost level beside SPY after the same costs and taxes; no single combination is picked afterwards." Add a "what this cannot tell you" section (the Not-modelled row, plus: CRSP has no volume, so real-world trading size limits are not tested).
- [ ] **Step 3:** Look up the cost basis in the library (`gg-library` MCP, subjects Trading and Mathematics-Statistics): quote one sourced figure each for large-company and small-company one-way costs with locators, and add them to the brief as the reason for the 5, 15 and 40 bps grid. If the library supports different numbers, put the library numbers in the grid and say so; do not leave the grid unexplained.
- [ ] **Step 4:** Run `.venv/bin/python scripts/study.py check research/stock-signals/dead-stock-first-test`. Expected: passes.
- [ ] **Step 5:** Commit `docs: dead-stock first test brief (frozen before any run)`. **Pause: show the owner the brief in plain words and wait for approval or changes before Task 2 starts.** The brief is not edited after this commit except by a dated amendment.

---

### Task 2: Point-in-time eligibility from the raw CRSP rows

**Files:**
- Create: `research/stock-signals/dead-stock-first-test/code/universe_frames.py`
- Test: `research/stock-signals/dead-stock-first-test/code/tests/test_universe_frames.py`

**Interfaces:**
- Consumes: `ggTrader.data.bars.connect_readonly(conn_str=None)`; `kaggle_crsp.daily_of(load_id)` (raw rows: `permno, date, shrcd, exchcd, prc, shrout`); the release spans frame from `Bars.spans` (columns `symbol, permno, spell, first_date, last_date, kind, possible_double`).
- Produces:
  - `month_end_fields(conn, load_id: int, start: str, end: str) -> pd.DataFrame` with columns `permno, month_end, shrcd, exchcd, abs_prc, cap` (cap = `abs_prc * shrout`; used for ranking only, so the file's share units do not matter). One row per permno per month-end trading day (the last row in that month on or before month-end).
  - `eligibility(fields: pd.DataFrame, spans: pd.DataFrame, *, universe: Literal["A", "B"], daily_index: pd.DatetimeIndex) -> pd.DataFrame`: boolean `daily_index x symbol` frame. A day is eligible when the **previous** month-end row passes the universe rule and the symbol's span covers the day. At most one symbol per permno is true on any day; when a demoted Yahoo link duplicates a `PERMNO-<n>` placeholder, the placeholder is kept.

- [ ] **Step 1: Write failing tests** in `test_universe_frames.py` on a hand-built fixture (6 permnos, 4 months): `test_negative_price_uses_absolute_value` (prc −7.0 passes the $5 floor), `test_rank_limit_keeps_top_n_by_cap_only` (N=2 of 4), `test_uses_previous_month_end_not_current` (a stock that crosses $5 on the 31st is not eligible until the next month's first day), `test_exchange_change_is_dated` (exchcd 2→1 on a known month: ineligible before, eligible after), `test_preferred_and_adr_codes_excluded` (shrcd 11 in, 73 out), `test_at_most_one_column_per_permno_per_day` (fixture with a flagged yahoo column and its placeholder; assert `eligible.T.groupby(permno_of).sum().max() <= 1` for every day and that the placeholder is the kept one), `test_dead_after_last_date_not_eligible`.
- [ ] **Step 2:** Run `scripts/run_tests.sh research/stock-signals/dead-stock-first-test/code/tests/test_universe_frames.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement `month_end_fields` (one grouped SQL read through `connect_readonly`, using `daily_of`) and `eligibility` (pandas only) as specified above.
- [ ] **Step 4:** Re-run the same command. Expected: all pass.
- [ ] **Step 5:** `ruff check` and commit `feat: point-in-time eligibility from CRSP rows for the dead-stock test`.

---

### Task 3: Monthly replay with costs and dead-stock exits (before tax)

**Files:**
- Create: `research/stock-signals/dead-stock-first-test/code/replay.py`
- Test: `research/stock-signals/dead-stock-first-test/code/tests/test_replay.py`

**Interfaces:**
- Consumes: `Bars.close`, `Bars.spans`, `eligibility(...)` frame (Task 2), `momentum_scores` (xs_momentum), `simulation_close(close, spans, loss)` and `forced_exit_frames(close, spans)` (`ggTrader.lab.dead_stocks`).
- Produces:
  - `@dataclass(frozen=True) class Trade: date: pd.Timestamp; symbol: str; qty: float; price: float; reason: Literal["rebalance", "dead_exit"]` (`qty` > 0 buy, < 0 sell; `price` is the execution price **before** cost; cost is charged by the ledger).
  - `plan_trades(close, eligible, spans, *, top_n: int, lookback: int, skip: int, dead_loss: float, start: str, end: str) -> list[TargetStep]` where `TargetStep(date, targets: dict[str, float])` holds equal weights chosen on a decision day and applied on the next trading day, plus dead-exit dates. Pure function: no cash, no tax. (The ledger in Task 4 sizes and executes.)
  - `dead_exit_dates(close, spans) -> pd.Series` (symbol → last bar date, only for series that end inside the window and have no successor).

- [ ] **Step 1: Write failing tests** on synthetic closes: `test_decision_day_prices_only` (changing every price after a decision day does not change that day's targets), `test_trade_is_next_day` (targets chosen on month-end `d` carry execution date = next trading day), `test_top_n_equal_weight_and_only_eligible` (a high-score ineligible stock is never chosen), `test_dead_stock_exit_on_last_bar_at_haircut` (loss 0.5 → exit price = 0.5 × last close; loss 1.0 → `SIM_PRICE_FLOOR`), `test_successor_stock_not_written_down` (`has_successor=True` span keeps its last price; Review Focus 1), `test_halted_stock_valued_at_last_print` (a NaN gap on a rebalance day does not drop the stock; Review Focus 5), `test_no_entry_after_death` (a dead stock's column never appears in targets after its last date).
- [ ] **Step 2:** Run `scripts/run_tests.sh research/stock-signals/dead-stock-first-test/code/tests/test_replay.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement the three functions above. Decision-day set: last trading day of each calendar month inside the window. Reuse `momentum_scores` and the lab's dead-stock helpers; do not re-implement the haircut.
- [ ] **Step 4:** Re-run. Expected: pass.
- [ ] **Step 5:** Cross-check against the lab engine on a survivors-only fixture: `test_matches_engine_without_dead_stocks` builds 30 synthetic always-alive stocks, runs the same rule through `ggTrader.lab.strategies.xs_momentum.XsMomentum().build` plus `ggTrader.engine.portfolio.run_weights`, and asserts the replay's pre-tax equity (zero cost, zero tax) equals the engine's within 1e-6 relative on every month-end. If it differs, find out why before going on; do not loosen the tolerance.
- [ ] **Step 6:** `ruff check`; commit `feat: monthly replay planner with dead-stock exits`.

---

### Task 4: Tax ledger execution and the SPY twin

**Files:**
- Create: `research/stock-signals/dead-stock-first-test/code/ledger_run.py`
- Test: `research/stock-signals/dead-stock-first-test/code/tests/test_ledger_run.py`

**Interfaces:**
- Consumes: `TaxLotBook`, `TaxParams` from `research/paper-system/tax-lots/code/tax_lots.py` (load by file path through `bootstrap` helpers, as other studies do; never copy the file); `TargetStep` and `dead_exit_dates` from Task 3.
- Produces:
  - `run_book(close: pd.DataFrame, steps: list[TargetStep], dead_exits: pd.Series, exit_close: pd.DataFrame, *, params: TaxParams, one_way_cost: float, start: str, end: str, start_cash: float = 100_000.0) -> BookResult`
  - `@dataclass(frozen=True) class BookResult: pretax_equity: pd.Series; after_tax_final: float; tax_paid: float; cost_paid: float; n_dead_exits: int; dead_exit_loss_realised: float; cagr_pretax: float; cagr_after_tax: float; trades: list[Trade]`
  - `run_spy(spy_close: pd.Series, *, params, one_way_cost, start, end, start_cash=100_000.0) -> BookResult` (buy on day one, sell on the last day, same costs and tax).
  - Sizing rule: each rebalance sizes on `marks + book.cash − book.unpaid_tax` (the owner's own tax bill is not spendable); each year-end calls `book.year_end(year, asof)` on the first trading day of the next year and pays it with `pay_tax` from book cash, selling pro rata first when cash is short; the last day sells everything and pays the final tax. Costs: every buy fills at `price × (1 + c)`, every sell at `price × (1 − c)`.

- [ ] **Step 1: Write failing tests:** `test_single_winner_taxed_at_long_term_rate` (buy, hold 400 days, double, liquidate: tax = 0.15 × gain within 1e-6), `test_short_term_gain_taxed_at_short_rate`, `test_costs_reduce_pretax_value_by_exact_amount` (one round trip at c = 0.0015 on a flat price loses exactly 0.003 of the traded value), `test_dead_exit_loss_reaches_ledger_in_death_year` (a held stock dies in 2012 at loss 1.0 with other gains that year: that year's assessed tax is lower by `st or lt rate × loss`; Review Focus 5), `test_dead_exit_loss_zero_realises_nothing_extra` (loss 0.0, flat price: no gain, no loss), `test_spy_twin_uses_same_costs_and_rates` (SPY run with c = 0 and zero rates equals buy-and-hold close ratio), `test_ledger_chain_verifies` (`book.verify_chain()` true after a run), `test_taxes_never_exceed_gains_when_all_losers` (all-loser fixture: tax_paid == 0).
- [ ] **Step 2:** `scripts/run_tests.sh research/stock-signals/dead-stock-first-test/code/tests/test_ledger_run.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement `run_book` and `run_spy` as specified. Where `TaxLotBook` forces a decision (year-end timing, shortfall), follow `research/paper-system/tax-lots/2026-11-02-after-tax-lot-engine-requirements.md` §2 R9, R12, R15; record the choice in the module docstring.
- [ ] **Step 4:** Re-run. Expected: pass.
- [ ] **Step 5:** `ruff check`; commit `feat: tax-ledger execution for the dead-stock test`.

---

### Task 5: Scale probe on the real release (orchestrator runs it)

**Files:**
- Create: `research/stock-signals/dead-stock-first-test/code/scale_probe.py`
- Test: `research/stock-signals/dead-stock-first-test/code/tests/test_scale_probe.py` (argument parsing and the sizing arithmetic only; no database)

**Interfaces:**
- Consumes: Tasks 2 and 3; `load_bars(symbols, start, end, source="combined", release_id=1)`.
- Produces: `results/dead_stock_first_test/scale_probe.json` with, for N = 500, 1,500 and all columns ever eligible in the development window: column count, seconds, peak memory in MB (`/proc/self/status` `VmHWM`), and `ok` (peak ≤ 2,500 MB). Decision rule written into the JSON: if the all-column read does not fit, the driver reads in date-bounded batches of columns and keeps only `close`.

- [ ] **Step 1:** Write `test_batches_cover_all_columns_once` and `test_peak_memory_reader_returns_mb` (monkeypatched `/proc` text). Run, expect FAIL.
- [ ] **Step 2:** Implement the probe. It reads only the columns ever eligible in `2004-01-01..2019-12-31` (warm-up year included), one batch at a time, and discards every field but `close` and `spans`.
- [ ] **Step 3:** Run tests. Expected: pass. Commit.
- [ ] **Step 4:** Orchestrator runs `scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-first-test/code/scale_probe.py --release-id 1 --out results/dead_stock_first_test/scale_probe.json`. Expected: finishes, JSON written. Read it myself. If the all-column read fails the 2,500 MB bar, stop and report to the owner in plain words before changing the plan.

---

### Task 6: Development-window driver and sanity checks

**Files:**
- Create: `research/stock-signals/dead-stock-first-test/code/run.py` (fill the scaffold)
- Test: `research/stock-signals/dead-stock-first-test/code/tests/test_run.py`

**Interfaces:**
- Consumes: Tasks 2 to 5.
- Produces: `run.py --window {dev,holdout} --universe {A,B} --release-id 1 --out PATH`, writing JSON with, for each loss in (1.0, 0.5, 0.0), cost in (6.4, 18.3, 40 bps) and tax set (standard, high): the strategy `BookResult` fields, the SPY twin's, and the difference in `cagr_after_tax`. Also a `coverage` block per year: companies eligible, companies that died that year, how many the rule held at death, the realised loss. A `provenance` block: release id, `bars_sha256`, brief commit SHA, git SHA, window, universe, `crsp_end`. `--window holdout` refuses to run if `results/dead_stock_first_test/holdout-<universe>.json` already exists.

- [ ] **Step 1: Write failing tests:** `test_holdout_runs_once` (second invocation raises `SystemExit` naming the existing file), `test_output_has_every_grid_cell` (3 × 3 × 2 cells, none missing), `test_window_dates_pinned` (dev ends 2019-12-31, holdout starts 2020-01-02 and ends 2024-12-31), `test_loss_changes_only_dead_exit_prices` (with no deaths in the fixture all three loss rows are identical).
- [ ] **Step 2:** Run `scripts/run_tests.sh research/stock-signals/dead-stock-first-test/code/tests/test_run.py -q`. Expected: FAIL, then implement `run.py`, re-run, expect pass. `scripts/study.py check` on the folder must pass.
- [ ] **Step 3:** Orchestrator runs the development window, universe A: `scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-first-test/code/run.py --window dev --universe A --release-id 1 --out results/dead_stock_first_test/dev-A.json`.
- [ ] **Step 4: Sanity checks (all read by me, not trusted from a builder):** (a) the SPY twin's before-tax growth over the window matches SPY's own close ratio within 0.1 percent at zero cost; (b) loss 0.0 gives a strictly higher or equal result than loss 1.0 in every cell; (c) the number of held-at-death stocks is greater than zero in the development window, otherwise the test says nothing about dead stocks and the report must say so; (d) no PERMNO is ever held in two columns (assert over the trade list). Any failure stops the plan and goes to the owner.
- [ ] **Step 5:** Commit code and `dev-A.json`'s summary (the raw JSON stays under `results/`, git-ignored, and is also copied into the study folder's `results/` as the cited artifact).

---

### Task 7: Reviews, then the holdout run once

**Files:**
- Create: `research/stock-signals/dead-stock-first-test/2026-11-23-review-sol.md`, `…-review-opus.md`

- [ ] **Step 1:** Delegate a fresh-context review of `code/` and the brief to Codex Sol (builders here are Claude or Codex Terra; Codex output also gets Opus). Give it the code diff and the brief only, not my conclusions. Focus: the five Review Focus lines, look-ahead (any use of a price after the decision day), the tax sizing rule, the dead-exit path.
- [ ] **Step 2:** Delegate a second fresh review to Opus on the same material, plus the `dev-A.json` numbers against the code's own definitions.
- [ ] **Step 3:** Fix every real finding myself or by a builder, re-run the affected tests, re-run the development window only if code changed, and say in the review files what changed. Do not look at holdout data.
- [ ] **Step 4:** Run the holdout once: `scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-first-test/code/run.py --window holdout --universe A --release-id 1 --out results/dead_stock_first_test/holdout-A.json`. Expected: JSON written. If it errors, fix the error and say so in the report; a rerun is allowed only for a crash that produced no result, and the report states it.
- [ ] **Step 5:** Commit reviews and the holdout summary.

---

### Task 8: Stress universe B, report and owner readout

**Files:**
- Create: `research/stock-signals/dead-stock-first-test/<date>-dead-stock-first-test.md` (from `research/TEMPLATE-research-report.md`), updates to `research/RESEARCH_INDEX.md` and `docs/changelog.md`

- [ ] **Step 1:** Run universe B for both windows (`--universe B`, outputs `dev-B.json`, `holdout-B.json`); same checks as Task 6 Step 4. B is a stress test: report it beside A, never instead of it. If memory fails the probe's bar for B, report that as a finding and do not shrink B quietly.
- [ ] **Step 2:** Write the report in plain words and in the Wikipedia-objective voice used for long reports: question, rule, data, results table (loss × cost × tax, strategy vs SPY, development and holdout side by side), dead-stock exposure (how many held stocks died, realised loss, tax effect), what it cannot tell you, and the verdict in one plain sentence per loss level. Every number is checked against the JSON by me before it is written. No GO/NO-GO labels and no old evidence bars.
- [ ] **Step 3:** Run `scripts/study.py check`, `scripts/study.py close` (writes `PIN.json`), `ruff check .`, and `scripts/run_tests.sh --studies` once, alone. Expected: all pass.
- [ ] **Step 4:** Commit and report to the owner with token usage for the phase. Ask before pushing.

---

## Self-review

- **Coverage:** loss 0/50/100 (Tasks 3, 4, 6); universe including dead stocks from CRSP (Task 2); costs and taxes both applied and compared with SPY on the same dates and ledger (Task 4); data the rule never saw (fixed rule, holdout run once, Task 7); plain words (Task 8, constraints); nothing pushed or deployed.
- **Names:** `eligibility`, `month_end_fields`, `plan_trades`, `dead_exit_dates`, `run_book`, `run_spy`, `BookResult`, `TargetStep` are defined once and used identically in later tasks.
- **Proportion:** eight tasks, each ending in a checkable artifact; the first and the scale probe are the cheap places to stop if the owner or the data says no.
- **Open for the owner:** the cost grid and tax sets (Task 1 pause), and whether universe B is wanted after A is read.
