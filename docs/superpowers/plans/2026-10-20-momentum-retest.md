# Momentum Re-Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Phases 0–2 of the pre-registered momentum re-test:
- fix the record;
- screen momentum on 100 years of Ken French data;
- if large caps pass, measure what costs and taxes take from our exact rules on the 2021–2026 S&P 500 tape.

**Architecture:** New modules under `src/ggTrader/lab/`, with thin drivers in `scripts/`:
- `french_data.py` and `momentum_screen.py` do the Phase 1 screen.
- `taxlots.py`, `raw_history.py` and `momentum_replay.py` do the Phase 2 replay.

Every driver writes a JSON artifact under `docs/research/`. Phase 2 tasks run only if Task 5 records that **Big** passed. Phases 3–5 get their own plans after their gates.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, yfinance, and the existing lab helpers (`significance.newey_west_se`, `event_study`, `data.load_ohlcv`, `index_constituents.sp500_members_asof`). For the T-bill series, coverage and seam scans, reuse the helpers in `scripts/production_replica_audit.py`.

**Spec:** `docs/research/briefs/2026-10-20-momentum-retest.md` (committed `eb7e4b7`). It is binding: never edit it. Any change needs a new dated brief and the owner's approval.

## Global Constraints

**Live paper trader.** It is production and frozen until about 2026-12-29.
- Never edit `src/ggTrader/paper/`, `.env`, `docker-compose.yaml` or `Dockerfile`.
- Never run `ggt paper` or `scripts/paper_trade.sh`, any `docker compose up/pull/restart`, or a deploy.
- Never call an Alpaca tool that places, cancels or closes orders.
- The database is SELECT-only; the loader's own cache writes are allowed.
- Never print secrets from `.env`.

**Lab code imports.** It must not import `ggTrader.paper.*` at all (none is needed here), and must not modify `src/ggTrader/data/live/`.

**Tests and lint.**
- Run tests only through `scripts/run_tests.sh <args>`. Exit 137 means a hang, so investigate it rather than raising the cap.
- Run `ruff check` and `ruff format` on changed files.

**Git.**
- Branch `research/momentum-retest`.
- Stage only the files each task lists, and never the untracked `.claude/skills/*-workspace/` dirs.
- Never push. Never merge to `main`; the owner merges.

**Fixed values from the brief.** Copy them exactly:

| Item | Value |
|---|---|
| Phase 1 NW lags | 12 |
| Holm family level | one-sided p = 0.00135 |
| Base annual cost haircut | Big 0.010, Small 0.030 |
| Stress annual cost haircut | Big 0.020, Small 0.060 |
| Full window | 1927-01 → 2026-04 |
| Post-publication window | 1994-01 → 2026-04 |
| Post-publication bar | ≥ +1.0 pt/yr |
| Diagnostic subperiods | 1927–1993, 1994–2012, 2013–2026 |
| Phase 2 window | 2021-01-31 → 2026-04-30, 420 calendar days warm-up |
| Selection | top 50; rank by P(t−21)/P(t−252) − 1; at least 253 bars |
| Trading cost | stocks 5 bp/side; benchmarks 1 bp/side (the brief doesn't set the benchmark cost; 1 bp/side matches the replica audit's SPY cost and is the planner's call) |
| Notional | $100,000 |
| Tax rates | 24% / 15% primary, 40.8% / 23.8% sensitivity |
| Re-entry lock | 31 days |
| Coverage floor | 0.95 |
| Seam threshold | 0.40 |
| Taint window | 5 bars |
| Kill rule | Rule T after-tax CAGR − SPY after-tax CAGR ≥ +1.0 pt/yr |

**Anti-goal.** Never change a rule, window or threshold after seeing a result.

**Long drivers.** Any driver that can run longer than about 5 minutes checkpoints its JSON after each unit of work and resumes from it.

## Review Focus

1. **French file format drift.** Changed headers, extra footnote lines, or the `-99.99` missing code inside the window must make the parser fail loudly, never silently mis-parse.
   Tests: `test_load_size_momentum_raises_on_missing_column` and `test_parse_marks_missing_codes_nan` (Task 3), plus `test_screen_raises_on_nan_in_window` (Task 4).
2. **Month alignment between the two French files.** The portfolios start in 1927-01 and the factors in 1926-07. The screen must inner-join months and refuse to run if either window isn't fully covered.
   Test: `test_screen_window_coverage_raises` (Task 4).
3. **Wash sales on partial replacement, including purchases in the 30 days *before* a loss sale.** Only the replaced fraction of the loss is disallowed.
   Tests: `test_wash_sale_partial_replacement_disallows_fraction` and `test_wash_sale_purchase_before_loss_sale_counts` (Task 6).
4. **Splits and dividends on held lots.** Share counts and per-share basis must adjust while the holding period stays the same, and dividends must be classified per lot.
   Tests: `test_split_preserves_basis_and_holding_period` and `test_dividend_qualified_61_day_rule` (Task 6).
5. **Year-end tax shortfall and delisting while locked.** When the tax due exceeds cash, sell HIFO from the largest position without going negative. A locked Rule T name that delists is sold at its last valid close.
   Tests: `test_year_end_tax_shortfall_sells_hifo` and `test_delisted_locked_name_sold_at_last_close` (Task 8).

---

## Phase 0: fix the record

### Task 1: Day-clustered FOMC event statistic

**Files:**
- Modify: `src/ggTrader/lab/event_study.py`, adding one function after `split_event_returns`.
- Create: `scripts/fomc_clustered_event_study.py`
- Test: `tests/lab/test_event_study.py`
- Output: `docs/research/_fomc_clustered_event_study_20261020.json`

**Interfaces:**
- Consumes, from `scripts/audit_reruns_wfo.py`:
  - constants `FOMC_EVAL_START`, `FOMC_EVAL_END`, `FOMC_WARMUP_DAYS`, `FOMC_UNIVERSE`, `FOMC_ROUNDTRIP_COST`;
  - the same `sys.path` insertion it uses to import from `scripts/`.
- Also consumes `ggTrader.lab.strategies.fomc_drift._event_signals`, `FomcDriftStrategy._load_events`, and `event_study_test`.
- Produces `split_event_returns_by_date(returns: pd.DataFrame, mask: pd.DataFrame) -> tuple[pd.Series, pd.Series]`:
  - **Event series:** one value per date on which at least one symbol is in position, equal to the mean return of the in-position symbols.
  - **Non-event series:** one value per date on which no symbol is in position, equal to the mean return of all symbols.
  - NaN returns are dropped before averaging.

- [ ] **Step 1: Write the failing test.** `test_split_event_returns_by_date_counts_dates_not_symbol_days`:
  - Use a 3-symbol × 6-date returns frame with its first row NaN, and a mask that is True for all three symbols on dates 2 and 4 only.
  - Assert the event series has length 2 and each value equals that row's mean.
  - Assert the non-event series has length 3, covering dates 1, 3 and 5; date 0 is NaN.
- [ ] **Step 2: Run** `scripts/run_tests.sh tests/lab/test_event_study.py -q`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement `split_event_returns_by_date`** in `event_study.py`.
- [ ] **Step 4: Run** the same command. Expected: PASS, with the existing event-study tests still passing.
- [ ] **Step 5: Write `scripts/fomc_clustered_event_study.py`.**
  - It has a `main()` and an argparse `--out`, defaulting to the output path above.
  - Load data the same way `run_fomc_drift_rerun` does: the universe, `load_ohlcv`, and the evaluation-window slice.
  - Build `_event_signals` for the primary pair only, `entry_offset_days=1, hold_days=0`.
  - **Gross:** call `event_study_test` on the series from `split_event_returns_by_date(close_eval.pct_change(), mask)`.
  - **Net:** subtract `FOMC_ROUNDTRIP_COST` from each symbol's first in-position day, as the rerun does, *before* averaging by date.
  - Write JSON with these keys:
    - `universe`, `eval_start`, `eval_end`, `pair`;
    - `n_event_dates`;
    - `gross` and `net`, as `EventStudyResult._asdict()`;
    - `unclustered_reference`, read from `docs/research/_audit_reruns_20261019.json` at `fomc_drift.primary_event_study`. Do not hardcode it.
- [ ] **Step 6: Run** `.venv/bin/python scripts/fomc_clustered_event_study.py`. Expected: the JSON is written, with `n_event_dates` between 100 and 130; it should be about 113.
- [ ] **Step 7: Commit.**

```bash
git add src/ggTrader/lab/event_study.py tests/lab/test_event_study.py scripts/fomc_clustered_event_study.py docs/research/_fomc_clustered_event_study_20261020.json
git commit -m "research(fomc): day-clustered event t for the three-ETF universe"
```

### Task 2: Record fixes (docs only)

**Files:**
- Modify: `docs/research/2026-10-19-production-replica-audit-nogo.md`. Append a dated `## Correction (2026-10-20)` section and leave the existing body unchanged.
- Modify: `docs/research/RESEARCH_SNAPSHOT.md`: the `dual_momentum` row, the `fomc_drift` row, and the DSR note in its methodology section.
- Modify: `docs/next_steps.md`. Add a DSR power note, and make the momentum re-test the current step with a link to the brief.
- Modify: `docs/research/prompts/web-strategy-research-prompt.md` and `docs/research/prompts/local-implementation-prompt-TEMPLATE.md`. Regenerate their ggTrader-context sections by following `.claude/skills/research-snapshot/SKILL.md`.
- Modify: `docs/changelog.md`, adding a 2026-10-20 entry.

**Interfaces:**
- Consumes: Task 1's JSON (`gross.t_stat`, `net.t_stat`, `n_event_dates`).

**Required statements.** Paraphrasing is fine; the numbers must come from the artifacts.
- **dual_momentum:** it is identical to `xs_momentum` on the 2026-10-19 replay, because at top_n 50 its negative-momentum filter never removed a pick. It is not an independent test.
- **DSR:**
  - With one-year training folds and a 12-combo grid, the corrected gate needs an in-sample Sharpe of roughly 2.5 to pass, so walk-forward OOS results are mostly per-fold anchor results.
  - Gate rejections stand. Low pass counts reflect low power, not proof that there is no edge.
- **FOMC:** the day-clustered gross and net t, and the number of event dates, from Task 1. The unclustered 3.619 / 1.572 counted each meeting day three times.
- **Prompt templates:** no remaining claim of KEEP 0.80 vs 0.78 as current, "gates 17/17", or 1.12/16.3 as current. They should reflect the audit NO-GO and point to the momentum re-test brief.

- [ ] **Step 1:** Make the edits above.
- [ ] **Step 2: Verify.** `grep -nE "17/17|0\.80 vs|1\.12 / 16\.3" docs/research/prompts/*.md` should return only lines that explicitly label those figures historical or void.
- [ ] **Step 3: Commit.**

```bash
git add docs/research/2026-10-19-production-replica-audit-nogo.md docs/research/RESEARCH_SNAPSHOT.md docs/next_steps.md docs/research/prompts/web-strategy-research-prompt.md docs/research/prompts/local-implementation-prompt-TEMPLATE.md docs/changelog.md
git commit -m "docs(research): record fixes — dual momentum duplicate, DSR power, clustered FOMC t, prompts"
```

---

## Phase 1: is momentum real, and in which company size?

### Task 3: French data loader with frozen files

**Files:**
- Create: `src/ggTrader/lab/french_data.py`
- Test: `tests/lab/test_french_data.py`, using inline fixture strings and no network access.
- Data, written by Task 4's download step: `data/external/french/` holds the two zips plus `MANIFEST.json`.

**Interfaces:** produces the following.

| Name | Definition |
|---|---|
| `FRENCH_BASE_URL` | `"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"` |
| `SIZE_MOM_FILE` | `"6_Portfolios_ME_Prior_12_2_CSV.zip"` |
| `FACTORS_FILE` | `"F-F_Research_Data_Factors_CSV.zip"` |
| `FRENCH_DIR` | `PROJECT_ROOT / "data" / "external" / "french"` |

- `download_and_freeze(files: list[str], dest: Path = FRENCH_DIR, http_get: Callable[[str], bytes] | None = None, force: bool = False) -> dict`
  - Writes each zip and returns `MANIFEST.json`, which holds `{file: {url, sha256, bytes, downloaded_utc}}`.
  - Raises `FileExistsError` if a file already exists, unless `force` is set.
- `verify_manifest(dest: Path = FRENCH_DIR) -> dict`: raises `ValueError` on a missing file or a checksum mismatch, and returns the manifest.
- `parse_french_monthly(text: str, section_title: str | None) -> pd.DataFrame`
  - Parses the monthly table that follows `section_title`. If `section_title` is None, it parses the first table and stops at a blank line or a line containing "Annual".
  - The index is month-end `Timestamp`s parsed from `YYYYMM`.
  - Columns are stripped headers. Values are converted from percent to decimal.
  - `-99.99` and `-999` become NaN.
- `load_size_momentum_vw(dest: Path = FRENCH_DIR) -> pd.DataFrame`
  - Uses the section titled `"Average Value Weighted Returns -- Monthly"`.
  - Columns are renamed as follows; any missing header raises `KeyError`.

    | Source header | Column |
    |---|---|
    | `SMALL LoPRIOR` | `small_lo` |
    | `ME1 PRIOR2` | `small_mid` |
    | `SMALL HiPRIOR` | `small_hi` |
    | `BIG LoPRIOR` | `big_lo` |
    | `ME2 PRIOR2` | `big_mid` |
    | `BIG HiPRIOR` | `big_hi` |

  - Header matching is case-insensitive and ignores whitespace.
- `load_market_total_return(dest: Path = FRENCH_DIR) -> pd.DataFrame` returns columns `mkt` (= `Mkt-RF` + `RF`) and `rf`, both in decimals.

The exact header strings must be confirmed against the real file at download time. If they differ, update the rename map and the test fixture to match the actual file, and note it in the commit message.

- [ ] **Step 1: Write the failing tests:**
  - `test_parse_value_weighted_section_only`: the fixture has both a VW and an EW section. Assert that only the VW values are returned, divided by 100, with a month-end index.
  - `test_parse_marks_missing_codes_nan`
  - `test_load_size_momentum_raises_on_missing_column`
  - `test_market_total_return_adds_rf`: Mkt-RF 1.00 and RF 0.30 give `mkt` 0.013.
  - `test_verify_manifest_detects_tamper`, using `tmp_path`.
  - `test_download_refuses_overwrite`, using a fake `http_get`.
- [ ] **Step 2: Run** `scripts/run_tests.sh tests/lab/test_french_data.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `french_data.py`. Read zip members with `zipfile`; the default `http_get` is `urllib.request`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit.**

```bash
git add src/ggTrader/lab/french_data.py tests/lab/test_french_data.py
git commit -m "feat(lab): Ken French loader with frozen, checksummed files"
```

### Task 4: Screen statistics and driver

**Files:**
- Create: `src/ggTrader/lab/momentum_screen.py`
- Create: `scripts/momentum_french_screen.py`
- Test: `tests/lab/test_momentum_screen.py`
- Output: `docs/research/_momentum_french_screen_results.json`, plus the committed `data/external/french/*`.

**Interfaces:**
- Consumes: Task 3 loaders, and `ggTrader.lab.significance.newey_west_se`.
- Produces these constants, with values copied from the Global Constraints:

| Constant | Value |
|---|---|
| `FULL` | `("1927-01", "2026-04")` |
| `POST` | `("1994-01", "2026-04")` |
| `DIAG` | `[("1927-01", "1993-12"), ("1994-01", "2012-12"), ("2013-01", "2026-04")]` |
| `BASE_COST` | `{"big": 0.010, "small": 0.030}` |
| `STRESS_COST` | `{"big": 0.020, "small": 0.060}` |
| `NW_LAGS_MONTHLY` | `12` |
| `HOLM_ALPHA` | `0.00135` |
| `POST_BAR_PP` | `1.0` |

- And these functions:
  - `net_excess(winners: pd.Series, market: pd.Series, annual_cost: float) -> pd.Series`: returns the aligned monthly `winners − market − annual_cost/12`, and raises `ValueError` on NaN.
  - `nw_t(x: pd.Series, lags: int = NW_LAGS_MONTHLY) -> float`
  - `one_sided_p(t: float) -> float`, computed as `scipy.stats.norm.sf(t)`.
  - `holm(pvals: dict[str, float], alpha: float = HOLM_ALPHA) -> dict[str, bool]`: a step-down procedure; with two hypotheses the first threshold is α/2 and the second is α.
  - `cagr_diff_pp(winners: pd.Series, market: pd.Series, annual_cost: float) -> float`: the CAGR of `winners − annual_cost/12` minus the CAGR of `market`, each compounded monthly over the series' span, in percentage points.
  - `capm(win_ex_rf: pd.Series, mkt_ex_rf: pd.Series, lags: int = 12) -> dict`, with keys `alpha_ann_pct`, `alpha_t` and `beta`. Use OLS, with an NW t on the alpha residual mean.
  - `worst_rolling_relative(winners: pd.Series, market: pd.Series, months: int = 12) -> dict`, with keys `start`, `end` and `relative_pct`. Compound both series over the window and take the minimum of the ratio minus 1.
  - `screen(sm: pd.DataFrame, mk: pd.DataFrame) -> dict`, returning:
    - `groups.{big,small}`, each holding `full_t`, `full_p`, `post_cagr_diff_pp`, `diag`, `stress`, `capm`, `size_matched` and `worst_12m`;
    - `holm`;
    - `pass.{big,small}`;
    - `next`, holding the brief §5 table's row text.
  - The size-matched comparison is the winners minus the mean of the same size's three portfolios.
- **Pass rule.** `pass[g]` = Holm rejects g at base cost over `FULL`, **and** `post_cagr_diff_pp[g] ≥ POST_BAR_PP` at base cost.

- [ ] **Step 1: Write the failing tests:**
  - `test_net_excess_subtracts_monthly_cost`: winners 0.02, market 0.01 and cost 0.012 give 0.009 for every month.
  - `test_holm_two_hypotheses`:
    - `{"big": 0.0005, "small": 0.0010}` → both True;
    - `{"big": 0.0009, "small": 0.0010}` → both False;
    - `{"big": 0.0001, "small": 0.0020}` → big True, small False.
  - `test_cagr_diff_pp_constant_returns`: 120 months of winners at 0.01 and market at 0.005, with cost 0, gives `(1.01**12 - 1.005**12) * 100` within 1e-9.
  - `test_screen_window_coverage_raises`: if the data ends in 2026-03, raise `ValueError`.
  - `test_screen_raises_on_nan_in_window`
  - `test_screen_pass_requires_both_tests`: a synthetic series that is strongly positive over 1927–1993 and negative from 1994 gives `pass.big` False even though `full_t ≥ 3.2`.
- [ ] **Step 2: Run** `scripts/run_tests.sh tests/lab/test_momentum_screen.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `momentum_screen.py`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Write `scripts/momentum_french_screen.py`.**
  - It has a `main()` with argparse:
    - `--download` calls `download_and_freeze` for both files, and never uses force;
    - `--out` sets the output path.
  - By default it calls `verify_manifest`, loads the files, runs `screen` and writes the JSON. The JSON adds `provenance` (the manifest), `brief` (the path), and `runtime_sec`.
- [ ] **Step 6: Download once, then run offline.**
  - `.venv/bin/python scripts/momentum_french_screen.py --download`
  - `.venv/bin/python scripts/momentum_french_screen.py`

  Expected: the JSON is written and the manifest verifies.
- [ ] **Step 7: Commit.**

```bash
git add src/ggTrader/lab/momentum_screen.py tests/lab/test_momentum_screen.py scripts/momentum_french_screen.py data/external/french/ docs/research/_momentum_french_screen_results.json
git commit -m "research(momentum): Phase 1 French size x momentum screen"
```

### Task 5: Phase 1 report and gate decision

**Files:**
- Create: `docs/research/<YYYY-MM-DD>-momentum-french-screen[-nogo].md`, following `docs/research/TEMPLATE-research-report.md`.
- Modify: `docs/research/RESEARCH_SNAPSHOT.md` (the `xs_momentum` row), `docs/next_steps.md` and `docs/changelog.md`.

**Required content:**
- Every number comes from the Task 4 JSON, cited by key:
  - Big and Small `full_t`, with the Holm decision;
  - `post_cagr_diff_pp`;
  - the three diagnostic subperiods and stress costs;
  - CAPM alpha and beta;
  - the size-matched comparison;
  - the worst 12 months.
- The verdict follows the brief §5 table exactly.
- State that a pass authorizes more research only.
- Label any claim from published research as unverified.

- [ ] **Step 1:** Write the report.
- [ ] **Step 2: Second read.** The senior quant applies `.claude/skills/research-report-review/SKILL.md`, and you apply the corrections.
- [ ] **Step 3: Gate.**
  - If `pass.big` is False, Phase 2 is **not executed** and this plan ends after the commit. If only Small passed, the report says that Phase 3 needs paid small-cap data and that this is the owner's decision.
  - If `pass.big` is True, continue to Task 6.
- [ ] **Step 4: Commit.**

```bash
git add docs/research/*momentum-french-screen*.md docs/research/RESEARCH_SNAPSHOT.md docs/next_steps.md docs/changelog.md
git commit -m "research(momentum): Phase 1 report and gate decision"
```

---

## Phase 2: our version after costs and taxes. Run only if Task 5 records `pass.big = True`

### Task 6: Multi-lot tax engine

**Files:**
- Create: `src/ggTrader/lab/taxlots.py`
- Test: `tests/lab/test_taxlots.py`

**Interfaces:** produces the following.
- `@dataclass Lot`, with fields:
  - `symbol: str`, `shares: float`, `basis: float` (total dollars);
  - `holding_start: pd.Timestamp`, which wash-sale tacking may move earlier.
- `class TaxBook(st_rate: float, lt_rate: float, credit_ordinary_offset: bool = False)`, with methods:
  - `buy(symbol, shares, price, date, cost) -> None`
  - `sell(symbol, shares, price, date, cost) -> float`: returns net proceeds. Lots are chosen HIFO, and realized gains are classified short-term (held 365 days or less) or long-term.
  - `dividend(symbol, per_share, ex_date) -> float`: returns cash. The dividend is qualified for lots held at least 61 days by the ex-date, and ordinary otherwise.
  - `split(symbol, ratio, date) -> None`: share count × ratio; total basis and holding start unchanged.
  - `interest(amount) -> None`: ordinary income.
  - `close_year(year: int) -> float`: returns the net tax for the year. A negative value is possible only when `credit_ordinary_offset` is set, and is −min(3000, net loss) × `st_rate`.
  - `deferred_tax(prices: dict[str, float], date) -> float`
  - `positions() -> dict[str, float]`, `lots(symbol) -> list[Lot]`, and the properties `carryforward_st` and `carryforward_lt`.
- **Wash-sale rule.**
  - For a loss sale of q shares on date d, replacement shares are shares of the same symbol bought in [d−30, d+30] that are not part of the lots sold. They are matched by purchase date.
  - The disallowed fraction is min(q, replacement)/q. The disallowed loss is added pro rata to the replacement lots' basis, and their `holding_start` moves earlier by the sold lot's holding duration.
  - Replacements bought *after* d are matched when they're bought, so the book keeps pending loss sales for 30 days.
- **Yearly netting** in `close_year`:
  - Net short-term and long-term gains separately, then apply carryforwards with their character kept.
  - Ordinary items (interest and non-qualified dividends) are taxed at `st_rate`. Qualified dividends and net long-term gains are taxed at `lt_rate`.
  - A net loss carries forward. The $3,000 ordinary offset applies only with `credit_ordinary_offset`.

- [ ] **Step 1: Write the failing tests:**
  - `test_hifo_partial_sale_picks_highest_basis`
  - `test_st_vs_lt_classification_365_vs_366_days`
  - `test_wash_sale_buy_within_30_days_after_disallows_loss_and_adds_basis`
  - `test_wash_sale_purchase_before_loss_sale_counts`
  - `test_wash_sale_partial_replacement_disallows_fraction`: sell 100 shares at a $10/share loss and buy 40 within 30 days. Assert $400 is disallowed and $600 allowed.
  - `test_wash_sale_buy_31_days_later_allows_loss`
  - `test_carryforward_keeps_character`:
    - Year 1 has a net short-term loss of 5,000 with no credit, so the short-term carryforward is 5,000.
    - Year 2 has a short-term gain of 3,000, so the tax is 0 and the carryforward is 2,000.
  - `test_ordinary_offset_credit_only_when_flagged`: the credit is −720.0 at 24% with the flag, and 0 without it.
  - `test_dividend_qualified_61_day_rule`
  - `test_split_preserves_basis_and_holding_period`
  - `test_deferred_tax_matches_liquidation_without_costs`
- [ ] **Step 2: Run** `scripts/run_tests.sh tests/lab/test_taxlots.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `taxlots.py`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit.**

```bash
git add src/ggTrader/lab/taxlots.py tests/lab/test_taxlots.py
git commit -m "feat(lab): multi-lot tax engine with wash sales and carryforwards"
```

### Task 7: Raw prices, dividends and splits, with reconciliation

**Files:**
- Create: `src/ggTrader/lab/raw_history.py`
- Test: `tests/lab/test_raw_history.py`
- Modify: `.gitignore`, adding `results/cache/` if it isn't already ignored.

**Interfaces:** produces the following.
- `@dataclass RawHistory`, with three frames, each dates × symbols:
  - `close`: raw closes;
  - `dividends`: cash per share on the ex-date, 0.0 elsewhere;
  - `splits`: the ratio on the ex-date, 1.0 elsewhere.
- `RAW_CACHE_DIR = PROJECT_ROOT / "results" / "cache" / "raw_history"`
- `load_raw_history(symbols: list[str], start: str, end: str, fetch: Callable[[str, str, str], pd.DataFrame] | None = None) -> tuple[RawHistory, list[str]]`
  - Returns the history and the missing symbols.
  - The default fetch is `yf.Ticker(s).history(start=start, end=end, auto_adjust=False, actions=True)`, using the columns `Close`, `Dividends` and `Stock Splits`. A ratio of 0 means no split.
  - Results are cached per symbol under `RAW_CACHE_DIR`.
  - There is no silent fallback: absent symbols are returned in the missing list.
- `reconcile(raw: RawHistory, adjusted_close: pd.DataFrame, tol: float = 0.005) -> list[dict]`
  - For each symbol and calendar month, compare the total-return factor rebuilt from raw closes, split ratios and reinvested dividends against `adjusted_close[last] / adjusted_close[first]`.
  - Return mismatches above `tol` as `{"symbol", "month", "raw_tr", "adj_tr"}`.

- [ ] **Step 1: Write the failing tests:**
  - `test_reconcile_split_and_dividend_match`: a synthetic 2:1 split plus a $1 dividend, with a consistent adjusted series, gives `[]`.
  - `test_reconcile_flags_tampered_month`: gives one mismatch.
  - `test_load_raw_history_reports_missing_symbols`: uses a fake fetch that raises or returns empty for one symbol.
- [ ] **Step 2: Run** `scripts/run_tests.sh tests/lab/test_raw_history.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `raw_history.py`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit.**

```bash
git add src/ggTrader/lab/raw_history.py tests/lab/test_raw_history.py .gitignore
git commit -m "feat(lab): raw price/dividend/split history with adjusted-tape reconciliation"
```

### Task 8: Momentum replay engine (Rules B and T) and buy-and-hold benchmarks

**Files:**
- Create: `src/ggTrader/lab/momentum_replay.py`
- Test: `tests/lab/test_momentum_replay.py`

**Interfaces:**
- Consumes:
  - `TaxBook` (Task 6) and `RawHistory` (Task 7);
  - a daily T-bill simple-rate `pd.Series`, as produced by `tbill_daily` in `scripts/production_replica_audit.py`;
  - `members_fn: Callable[[pd.Timestamp], list[str]]`, which in the driver is `sp500_members_asof`;
  - `data/universe/sp500_sectors.json`, for the concentration diagnostic.
- Produces `@dataclass(frozen=True) ReplayConfig`, with fields:

| Field | Default |
|---|---|
| `rule: str` | `"B"` or `"T"` |
| `top_n: int` | 50 |
| `lookback: int` | 252 |
| `skip: int` | 21 |
| `min_bars: int` | 253 |
| `cost_bps: float` | 5.0 |
| `start_cash: float` | 100_000.0 |
| `st_rate: float` | 0.24 |
| `lt_rate: float` | 0.15 |
| `reentry_lock_days: int` | 31 |
| `credit_ordinary_offset: bool` | False |
| `value_unused_carryforward: bool` | False |

- And `@dataclass ReplayResult`, with fields:
  - `nav_pretax: pd.Series`
  - `after_tax_cagr_pct: float`: the primary endpoint, after terminal liquidation.
  - `deferred_basis_cagr_pct: float`: the secondary endpoint.
  - `trades: pd.DataFrame`, with columns `date, symbol, side, notional, price, reason, lot_term, realized`.
  - `taxes_by_year: dict[int, float]`
  - `diagnostics: dict`, with keys:
    - `turnover_annual`, `beta_spy`, `sector_max_share`;
    - `worst_rel_drawdown_pct`, `recovery_days`;
    - `rebound_2022_23_rel_pct`: the return relative to SPY from SPY's 2022 trough to its next high;
    - `tax_drag_breakdown`, with keys `st`, `lt`, `div` and `cash`;
    - `locked_slot_share`, for Rule T only.
- And these functions:
  - `rank_momentum(adj_close, asof, eligible, lookback=252, skip=21, min_bars=253) -> list[str]`: sorts by `P(t−21)/P(t−252) − 1` from the adjusted closes on `asof`, descending, with ties broken by ticker.
  - `decision_dates(index: pd.DatetimeIndex, rule: str) -> list[pd.Timestamp]`: the last trading day of each month for B, and of each March, June, September and December for T.
  - `run_momentum_replay(adj_close, raw, tbill, eval_start, eval_end, cfg, members_fn) -> ReplayResult`
  - `run_buy_and_hold(symbol, raw, tbill, eval_start, eval_end, st_rate, lt_rate, cost_bps=1.0, start_cash=100_000.0) -> ReplayResult`: buys at the first session's close and reinvests dividends at the next close. It is used for SPY and MTUM.

**Daily order of operations** in `run_momentum_replay`:
1. **Corporate actions and cash.** Apply splits, then dividends (the cash goes to sleeve cash), then that day's T-bill interest on cash.
2. **Delistings.** A held symbol with no raw close from today onward is sold at its last valid close, with reason `"delisted"`.
3. **Trades.** If the previous session was a decision date, trade at today's raw close, 5 bp per side, buying only with available cash.
   - **Rule B:**
     - Sell every non-target name in full.
     - Trim or add continuing names to an equal 1/50 share of NAV.
     - Buy new names with the remaining cash at equal weight.
   - **Rule T:**
     - Keep a held non-target only if one of its lots both shows an unrealized gain and has been held 365 days or less. Sell every other non-target in full.
     - Fill `top_n −` the held count with the highest-ranked names that are neither held nor sold at a loss within 31 days.
     - Split the cash equally among the additions. No trims.
4. **Year-end taxes.** On the last session of each year, call `tax = book.close_year(year)` and pay it from cash. If cash is short, sell HIFO from the largest position.
5. **Terminal liquidation.** On `eval_end`, sell everything at that close and call `close_year`. If `value_unused_carryforward` is set, add the remaining carryforward × `lt_rate`.

**Steps:**
- [ ] **Step 1: Write the failing tests**, all on synthetic frames:
  - `test_rank_uses_skip_and_lookback`
  - `test_decision_dates_month_and_quarter_ends`
  - `test_rule_t_keeps_short_term_winner_outside_top_n`, which then expects a sale at the first quarter-end after the lot turns 366 days old.
  - `test_rule_t_sells_locked_name_when_gain_turns_to_loss`
  - `test_reentry_lock_blocks_rebuy_within_31_days`
  - `test_cash_never_negative`
  - `test_year_end_tax_shortfall_sells_hifo`
  - `test_delisted_locked_name_sold_at_last_close`
  - `test_buy_and_hold_reinvests_dividends_and_taxes_them`
- [ ] **Step 2: Run** `scripts/run_tests.sh tests/lab/test_momentum_replay.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `momentum_replay.py`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit.**

```bash
git add src/ggTrader/lab/momentum_replay.py tests/lab/test_momentum_replay.py
git commit -m "feat(lab): tax-aware momentum replay (rules B/T) and taxed buy-and-hold"
```

### Task 9: Phase 2 driver, data gates, run and report

**Files:**
- Create: `scripts/momentum_retest_phase2.py`
- Test: `tests/scripts/test_momentum_retest_phase2.py`
- Output: `docs/research/_momentum_phase2_results.json` and `docs/research/_momentum_phase2_curves.csv`.
- Create: `docs/research/<YYYY-MM-DD>-momentum-phase2[-nogo].md`
- Modify: `docs/research/RESEARCH_SNAPSHOT.md`, `docs/next_steps.md` and `docs/changelog.md`.

**Interfaces:**
- Consumes:
  - Tasks 6–8;
  - `load_ohlcv`, `sp500_members_asof` and `all_members_between`;
  - from `scripts/production_replica_audit.py`, through the same `sys.path` import pattern: `tbill_daily`, `seam_scan`, `coverage`, `tainted_trades` and `SEAM_THRESHOLD`.
- Produces two functions:
  - `data_gates(coverage_mean: float, reconcile_mismatches: list[dict], missing: list[str], traded: set[str], tainted: list[dict]) -> list[str]`: returns the stop flags. It flags coverage below 0.95, any mismatch or missing symbol inside `traded`, and any tainted trade.
  - `phase2_verdict(t_cagr: float, spy_cagr: float, mtum_cagr: float, flags: list[str]) -> dict`: returns `{"verdict", "reason", "mtum_note"}`.
    - Any flag gives `INCONCLUSIVE`.
    - Otherwise, `t_cagr − spy_cagr ≥ 1.0` gives `SURVIVES_PHASE2`; anything less gives `NO-GO`.
    - `mtum_note` is `"recommend MTUM"` when T survives but `t_cagr < mtum_cagr`.

**What the driver does:**
1. **Load data.** The PIT universe over the window, plus adjusted closes from `load_ohlcv` with the 420-day warm-up.
2. **Fetch raw history.** First take the union of the top-100 ranked names at every decision date. Fetch `load_raw_history` for that union plus SPY and MTUM.
3. **Run the arms:**
   - Rules B and T at 24/15;
   - Rule T at 40.8/23.8;
   - Rule T with `credit_ordinary_offset` and `value_unused_carryforward` set;
   - SPY and MTUM buy-and-hold at both brackets.
4. **Report pre-tax evidence.** The `excess_stats` NW t of each rule's daily pre-tax returns against SPY.
5. **Check data quality.** Verify seams against fresh yfinance, as the replica audit's `_verify_seams` does. Build the `traded` set from both rules' trades, then call `data_gates`.
6. **Save as it goes.** Checkpoint the JSON after each arm.

**Steps:**
- [ ] **Step 1: Write the failing tests:**
  - `test_data_gates_flags_low_coverage_and_traded_mismatch`
  - `test_data_gates_ignores_untraded_mismatch`
  - `test_phase2_verdict_mapping`: covers the boundary at exactly 1.0 and the MTUM note.
- [ ] **Step 2: Run** `scripts/run_tests.sh tests/scripts/test_momentum_retest_phase2.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the driver: a `main()` with argparse `--out`, `--curves` and `--force`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Run the study.** `.venv/bin/python scripts/momentum_retest_phase2.py`. Expected: the JSON is written with every arm, and `verdict` is present.
- [ ] **Step 6: Report.**
  - Follow the template, citing every figure by its JSON key.
  - Include all the brief §10 diagnostics.
  - State that this window was already seen, so the result can only kill.
  - The senior quant gives it a second read with `research-report-review`.
- [ ] **Step 7: Commit.**

```bash
git add scripts/momentum_retest_phase2.py tests/scripts/test_momentum_retest_phase2.py docs/research/_momentum_phase2_results.json docs/research/_momentum_phase2_curves.csv docs/research/*momentum-phase2*.md docs/research/RESEARCH_SNAPSHOT.md docs/next_steps.md docs/changelog.md
git commit -m "research(momentum): Phase 2 after-tax replay vs SPY/MTUM"
```

---

**After this plan:**
- If Phase 2 returns `SURVIVES_PHASE2`, the next plans are:
  - the paid-data vendor choice and Phase 3 replay, which needs owner approval of the purchase;
  - the Phase 4 live paper path, built after the freeze lift through `ggtrader-deploy`.
- If it returns `NO-GO`, momentum is closed, and the freeze lift retires the stock picks for the SPY sweep alone.
