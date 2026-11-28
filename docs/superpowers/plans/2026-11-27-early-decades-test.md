# Early-Decades Stock Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Score three versions of the first test's recent-winner rule once on 1969-02-03 to 1998-07-31, the only stock-level window nobody has scored, and report the result against SPY after costs and taxes.

**Architecture:** A new study folder `research/stock-signals/early-decades/` holds `ed_`-prefixed copies of the first test's engine (the pattern the momentum-deaths study used), plus a comparison-series reader and three arms. Everything is built and tested on synthetic data and on an already-used window first. The real window is scored once, after a written registration is committed and the owner has settled the comparison.

**Tech Stack:** Python, pandas, the `ggTrader` lab and `load_bars(source="combined")`, `scripts/study.py`, `scripts/run_tests.sh`.

**Spec:** Agenda candidates 1–3 in `research/programme/next-research-agenda-2026-11-26.md`; window limits in `research/programme/window-use-ledger.md`. Owner's go-ahead 2026-11-27 ("do the early decades test then the SPY comparison").

## Global Constraints

- Free data only, taxable account, information known on each date, stocks that later died stay in. The paper trader stays stopped; no orders.
- Pin `--eval-start 1969-02-03` and `--eval-end 1998-07-31` on every cited result; save raw JSON beside the report.
- Plain words in every owner-facing text; define each technical term where it first appears. Do not cite the retired old evidence rules. No GO/NO-GO labels.
- One scored run of the real window. If it trails the comparison, keep the result and do not tune rules on this window.
- Heavy jobs only through `scripts/run_capped.sh`, one at a time, overnight; tests only through `scripts/run_tests.sh` with targeted paths.
- Reports and registration are saved before scoring; nothing merged or pushed without the owner's say.

## Decisions made here (my calls; the owner can change any before Task 6)

1. **Rule:** the first test's fixed recent-winner rule exactly as written in its section 2 (copied, not re-tuned): each month buy the top 50 stocks by return over the past year skipping the latest month, from the 1,000 largest eligible stocks (universe A, price ≥ $5, positive shares).
2. **Three arms, all reported:** (A) monthly trading, the control; (B) yearly rebalancing with a one-year hold (long-term tax rate when held over a year); (C) monthly, but skip any stock with a one-day rise of 20% or more in the prior 252 trading days. Three arms means a lucky result is likelier; the report says so.
3. **Costs:** 0.183% per purchase or sale as in the first test, plus a 0.5% per side sensitivity, because 1970s–1990s trading was dearer than the 2009–2016 figure it comes from. Taxes: the first test's lower rates (24% short-term, 15% long-term).
4. **Dead stocks:** same three recovery readings as the first test (loss 0.0, middle, worst), headline reading loss 0.0 as the owner chose.
5. **Comparison (settled in Task 6 with the owner):** proposed SPY for 1993-01-29 to 1998-07-31 (frozen table `market.spy_daily_yahoo_1993_1999`), and for 1969-02-03 to 1992-12-31 the CRSP value-weighted market return `vwretd` (all stocks weighted by size; not a fund, so labelled a stand-in). Results for the two sub-periods are shown separately and together.
6. **Known data limits, to be stated in the report:** no trading volume, no final return when a stock dies, many early prices are quote midpoints (2,968 of 5,404 rows on 1973-01-02), only 1,203 NYSE common stocks qualify on 1969-02-03.

## Review Focus

- A one-day rise of 20% in arm C uses only data before the pick date (no lookahead).
- Midpoint prices and stale prices must not create a fake 20% jump or fake momentum.
- Stocks with a missing month, relisting or ticker/permno change must not leak a later return into an earlier pick.
- The yearly arm must apply the long-term rate only when the holding period actually exceeds a year.
- The comparison series and strategy cover identical dates; SPY and `vwretd` sub-periods must join without gap or overlap on 1993-01-29.

---

### Task 1: Scaffold the study and copy the engine

**Files:**
- Create: `research/stock-signals/early-decades/` (via `scripts/study.py new stock-signals early-decades --kind study --placement <row>`), `code/ed_*.py` copies of `dead-stock-momentum-deaths/code/dm_*.py` plus `frozen_copies.json` listing source paths and hashes
- Test: `research/stock-signals/early-decades/code/tests/test_ed_copies.py`

**Interfaces:** Produces `ed_dead_stock_first_test`, `ed_bars_reader`, `ed_ledger_run`, `ed_replay`, `ed_universe_frames` with the same public functions as their `dm_` sources.

- [ ] Write `test_ed_copies_match_sources_by_hash` (each copy's hash equals the recorded value after the documented renaming); run, see it fail.
- [ ] Scaffold, copy, record hashes; run the test, see it pass.
- [ ] Run `scripts/study.py check`; commit.

### Task 2: Regression on an already-used window

**Files:** `code/ed_run.py` (driver, `--window`, `--arm`, `--eval-start`, `--eval-end`, `--out`), test `code/tests/test_ed_regression.py`.

- [ ] Write `test_arm_a_reproduces_first_test_2020_2024_cells`: arm A over 2020-01-02..2024-12-31 at 0.183% and loss 0.0 equals the first test's saved cells (use `_holdout-A.json`) exactly.
- [ ] Implement the driver by reusing the copied engine; run the test; it must pass before anything touches 1969+.
- [ ] Commit.

### Task 3: Comparison-series reader

**Files:** `code/ed_comparison.py`, test `code/tests/test_ed_comparison.py`.

**Interfaces:** Produces `comparison_returns(start: str, end: str) -> pd.Series` (daily total return), `comparison_label(start, end) -> str` returning "SPY" or "CRSP market return (stand-in)".

- [ ] Tests: join at 1993-01-29 has no gap and no double day; whole range 1969-02-03..1998-07-31 has no missing trading day; `vwretd` and SPY are never mixed within a sub-period; the label is correct for each date.
- [ ] Implement with read-only queries against `kaggle_crsp.daily_of(1)` (`vwretd`) and `market.spy_daily_yahoo_1993_1999`.
- [ ] Apply the same cost-free buy-and-hold with tax on final sale to the comparison; test it on a toy series. Commit.

### Task 4: Arms B and C

**Files:** `code/ed_arms.py`, test `code/tests/test_ed_arms.py`.

**Interfaces:** `arm_b_schedule(rebalance_dates) -> list` (yearly dates), `jump_excluded(returns: pd.DataFrame, asof: pd.Timestamp, threshold: float = 0.20, lookback: int = 252) -> set[int]` (permnos to skip).

- [ ] Tests: yearly arm trades once a year and a 400-day holding gets the long-term rate while a 300-day holding gets short-term; `jump_excluded` ignores a jump dated after `asof`; a midpoint-to-midpoint change on a flat price does not trigger; a stock with a gap month is handled without error.
- [ ] Implement; run tests; commit.

### Task 5: Registration (nothing scored yet)

**Files:** `research/stock-signals/early-decades/2026-11-27-early-decades-registration.md`

- [ ] Write the registration in plain words: rule, three arms, dates, universe, costs, tax rates, recovery readings, comparison, count of variants (3 arms × 2 cost levels × 3 recovery readings), the data limits, and the rule that the result is reported as found.
- [ ] Commit it (the commit hash is the proof it came before scoring).

### Task 6: Owner settles the comparison

- [ ] Show the owner the proposed comparison (decision 5) and the alternative of starting the test in 1993 (SPY only). Edit the registration if they choose differently; commit.

### Task 7: The one scored run (overnight)

- [ ] Run `scripts/run_capped.sh .venv/bin/python research/stock-signals/early-decades/code/ed_run.py --arm {A,B,C} --eval-start 1969-02-03 --eval-end 1998-07-31 --out <raw json>` for each arm, one after another, detached so a tool timeout cannot kill it; reading prices from `thesix` is a steady read, fine overnight.
- [ ] Verify: three raw JSON files exist, each records start, end, release id, rule hash and code commit.

### Task 8: Report and review

**Files:** `research/stock-signals/early-decades/2026-11-2x-early-decades.md` (template `research/TEMPLATE-research-report.md`), changelog, `research/programme/window-use-ledger.md`, roster via the `research-snapshot` skill.

- [ ] Report all 18 cells per arm against the comparison, by sub-period, with the data limits; say plainly if the rule trails.
- [ ] Mark the window spent in the ledger; update the roster and `next_steps`.
- [ ] Two independent reviews from different vendors (Gemini Neon and Claude, not the builder's vendor); check every figure against the raw JSON.
- [ ] Commit; merge and push only on the owner's say.

## Self-Review

- Coverage: arms 1–3 (Tasks 2–4), registration before scoring (5), comparison decision (6), single scored run (7), report and review (8).
- Order matches the owner's request: the test is built first, then the SPY comparison is settled, and only then is the real window scored.
- Open risk: the window is one-shot, so arm-level surprises can't be fixed afterwards; Task 2's regression is the guard.
