# Dead-Stock Data Integrity and Payout Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out how clean our combined CRSP + Yahoo price data is for the stocks the first test held, and fill the biggest gap in it: why each dead stock stopped trading and what its holders got, using free SEC filings.

**Architecture:** A new study in `research/stock-signals/` with two phases. Phase 1 reads only our own database and result files: it lists the 141 dead stocks the rule held, names them, classifies their last month of prices, and runs data-quality checks over the CRSP rows. Phase 2 (after the owner's go) finds each stock's SEC delisting notice and 8-K filings, and writes a payout table (evidence class per stock, plus a suggested loss figure where the evidence supports one). No strategy rerun happens in this plan; the table is the input to a later plan.

**Tech Stack:** Python 3.12, pandas, psycopg (via the repo's existing DB helpers), `urllib` for SEC downloads, existing readers `research/stock-signals/_lib/sec_master_index.py`, `sec_form25.py`; pytest through `scripts/run_tests.sh`.

**Spec:** No separate design doc. The questions and limits come from the first test's report, [2026-11-24-dead-stock-first-test.md](../../../research/stock-signals/dead-stock-first-test/2026-11-24-dead-stock-first-test.md) sections 4 and 6, and the owner's instruction of 2026-11-24 ("focus on the data integrity and filling in gaps we may have"). Task 1 writes the frozen brief that stands in for a spec.

## Global Constraints

- Free data only (owner, 2026-10-23). SEC EDGAR is free and public.
- The first test's holdout (2020-2024) is spent for the 12-1 momentum rule and its claim files stay. This plan does not rerun any strategy and does not touch `results/dead_stock_first_test/*.claim`.
- The closed study `research/stock-signals/dead-stock-first-test/` is read as data only (its `_dev-A.json`, `_holdout-A.json` and trade files). Nothing in it is edited, and no code is imported from its `code/` folder (study isolation rule).
- Release pins: `release_id = 1`, CRSP load `1`, McGill load `2` (read through `kaggle_crsp.daily_of(1)` and `kaggle_chars.mcgill_of(2)`).
- SEC fair access: at most 10 requests a second (the plan uses at most 4), and every request carries a User-Agent that names who is asking. **Owner ruling of 2026-11-24: use the owner's email on file.** The string is `ggTrader research <that email>`, exported as `SEC_USER_AGENT` in the shell of the run command. The code reads only that variable and has no default; the email is never written into the repository, a result file, a report or a delegate brief (the SEC cache's `.meta.json` records the URL and hash, not the header).
- Raw SEC downloads go to `/media/thesix/ggTrader-data/sec-delistings/` with a `SHA256SUMS.txt`, never into the repository.
- Tests run only as `scripts/run_tests.sh <small targeted paths>`. Heavy jobs run only through `scripts/run_capped.sh`. One heavy job at a time. Exit 75 means busy, not failed.
- Plain words in every owner-facing text: no unexplained "bps", "PIT", "CIK", "8-K" without a one-sentence meaning in the same place.
- Nothing is merged, pushed or deployed without the owner's say. The trader stays stopped. Memory and OOM slice settings are not changed.
- Ruff must pass (`line-length = 100`). Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Delegation and usage (CLAUDE.md, 2026-11-08)

Usage on 2026-11-24 (owner screenshot): Codex weekly 85% left, Claude session 90% left, Claude weekly 32% left and ahead of pace. Claude weekly is the scarce meter, so reviews go to Codex and some building goes to Antigravity (owner's go of 2026-11-24: "give agy some work too"; this re-enables Antigravity Gemini for this plan, and the owner decides whether it stays on). Phase 1 is Tasks 1-4, then a pause and a usage report; Phase 2 is Tasks 5-8, then a pause and a usage report. If Claude weekly is under about 20% at a pause, Phase 2 waits for the reset.

| Task | Builder | Task reviewer |
| :--- | :--- | :--- |
| 1 roster, brief | Claude Sonnet (needs the database and the first study's JSON) | Codex Sol |
| 2 price labels | Antigravity Gemini 4 Neon | Codex Sol |
| 3 integrity checks | Antigravity Gemini 4 Neon | Codex Sol |
| 4 driver, run, findings | Claude Sonnet | Codex Sol |
| 5 SEC downloader | Antigravity Gemini 4 Neon | Codex Sol |
| 6 Form 25 matching | Claude Sonnet | Codex Sol |
| 7 8-K evidence | Antigravity Gemini 4 Neon | Codex Sol |
| 8 payout table, report | Claude Sonnet | Codex Sol |

Whole-study reviews: Codex Sol and Claude Opus (once, at the end of each phase), plus Antigravity Gemini 4 Argon as a third independent reader of the final report. A reviewer never reviews its own model family. The orchestrator re-reads every diff and reruns the checks itself. Never put the owner's email in any delegate brief.

## Review Focus

Failure modes the spec implies but a straight reading would miss, most likely first. Each is pinned by a test in the task named.

1. **Same-name wrong company** (two firms named alike, a company and its subsidiary): two Form 25 filers with the same normalized name inside the window must give `ambiguous`, never a pick. Task 6.
2. **Form 25 for something other than the common stock** (preferred shares, warrants, notes): only notices whose security description names common or ordinary shares count. Task 6.
3. **Name missing or stale** (8 of the 138 permno-keyed stocks have no McGill name; 3 are Yahoo tickers `ARG`, `FRX`, `HSP`): they must show up as `name_unknown` with a reason, not drop out of the count. Task 1.
4. **A download that is not data** (SEC returns 403/429 or an HTML error page with status 200): never cached, never hashed as evidence, and the run resumes cleanly. Task 5.
5. **A filing date that is far from the last trading day** (notice filed before the last bar, or months after): the match window is fixed in the brief (30 days before to 180 days after the last bar) and tested at both edges. Task 6.

## File Structure

New study folder `research/stock-signals/dead-stock-data-integrity/` (scaffolded in Task 1):

| File | One job |
| :--- | :--- |
| `code/roster.py` | The list of held dead stocks with names. |
| `code/price_path.py` | Labels each dead stock's last month from prices alone. |
| `code/integrity_checks.py` | Data-quality checks on CRSP daily rows and the link to the first test's holdings. |
| `code/sec_client.py` | Polite, caching SEC downloader. |
| `code/sec_filings.py` | Name matching and Form 25 / 8-K evidence. |
| `code/payout_table.py` | Evidence classes, suggested losses, coverage numbers. |
| `code/run.py` | Driver: `--phase integrity` and `--phase sec`. |
| `code/tests/test_*.py` | One test file per module above. |

Also touched: `research/RESEARCH_INDEX.md`, `docs/changelog.md`, the memory note (Task 8).

Study placement: row 9 of `research/README.md` (assessing candidate data availability) rather than row 3 (audits), because the study needs `research/stock-signals/_lib` (the SEC readers) and a study may only import its own category's `_lib`. The orchestrator records this as a ruling in the report.

---

### Task 1: Scaffold, frozen brief, and the roster of held dead stocks

**Files:**
- Create (via scaffold): `research/stock-signals/dead-stock-data-integrity/` with `README.md`, `study.toml`, `code/run.py`, `code/tests/`
- Create: `research/stock-signals/dead-stock-data-integrity/2026-11-24-dead-stock-data-integrity-brief.md`
- Create: `research/stock-signals/dead-stock-data-integrity/code/roster.py`
- Test: `research/stock-signals/dead-stock-data-integrity/code/tests/test_roster.py`

**Interfaces:**
- Produces: `parse_exits(result: dict, window: str, universe: str) -> pd.DataFrame` (columns `symbol, permno, window, universe, exit_date, last_close`; `permno` is a nullable Int64, empty for Yahoo tickers); `attach_names(conn, roster, mcgill_load_id: int, release_id: int) -> pd.DataFrame` (adds `stock_name, ticker, cusip, exchcd, name_source`, where `name_source` is `"mcgill"` or `"none"`, and fills `permno` for Yahoo tickers from `ref.symbol_map`); `build_roster(conn, study_root: Path, mcgill_load_id: int = 2, release_id: int = 1) -> pd.DataFrame` (reads `_dev-A.json` and `_holdout-A.json` from `study_root.parent / "dead-stock-first-test"`, key path `result.death_price_paths.exits`).

- [ ] **Step 1: Scaffold the study.** Run `.venv/bin/python scripts/study.py new stock-signals dead-stock-data-integrity --kind audit --placement 9 --date 2026-11-24`. Expected: folder created, `study.py check` prints `OK`.
- [ ] **Step 2: Write the brief** (the frozen rules; amendments are dated additions, never edits). It must contain, copied verbatim: the question ("For the dead stocks the first test held, what do our CRSP rows and SEC filings say about why they stopped trading, and how clean are the price rows around them?"); the roster definition (the 141 exits in `_dev-A.json` and `_holdout-A.json`, 117 + 24; the 3 Yahoo-ticker exits and the 8 permno-keyed stocks without a McGill name are kept and labelled); the price-path rules of Task 2; the integrity checks and thresholds of Task 3; the SEC match window (last bar minus 30 days to last bar plus 180 days, exact normalized-name match, no fuzzy matching in version one) and evidence-class table of Tasks 6-7; the suggested-loss table of Task 8; and the statement "no strategy is rerun by this study". The brief is committed before any real run.
- [ ] **Step 3: Write the failing tests** in `test_roster.py`:

```python
def test_parse_exits_reads_permno_and_ticker_rows():
    result = {"result": {"death_price_paths": {"exits": [
        {"symbol": "PERMNO-89260", "exit_date": "2005-06-24", "last_close": 74.98},
        {"symbol": "ARG", "exit_date": "2010-05-14", "last_close": 40.0}]}}}
    out = parse_exits(result, window="dev", universe="A")
    assert out["permno"].iloc[0] == 89260
    assert out["permno"].isna().tolist() == [False, True]
    assert out["symbol"].tolist() == ["PERMNO-89260", "ARG"]

def test_parse_exits_rejects_missing_exits_key():
    with pytest.raises(KeyError):
        parse_exits({"result": {}}, "dev", "A")

def test_attach_names_marks_unnamed_and_fills_ticker_permno(fake_conn):
    # fake_conn: McGill has permno 89260 -> comp_name "ALPHA CORP"; symbol_map ARG -> 11111; 11111 not in McGill
    out = attach_names(fake_conn, roster, mcgill_load_id=2, release_id=1)
    assert out.set_index("symbol").loc["PERMNO-89260", "name_source"] == "mcgill"
    assert out.set_index("symbol").loc["ARG", "permno"] == 11111
    assert out.set_index("symbol").loc["ARG", "name_source"] == "none"
```

- [ ] **Step 4: Run to see them fail.** `scripts/run_tests.sh research/stock-signals/dead-stock-data-integrity/code/tests/test_roster.py -v` expects FAIL (module missing).
- [ ] **Step 5: Implement** `parse_exits`, `attach_names`, `build_roster` in `roster.py`. McGill name rule: the row with the latest `date` on or before the exit date for that permno; if the permno has only later rows, use its earliest row and set `name_source = "mcgill_later"`. Use parametrised SQL on `kaggle_chars.mcgill_of(%s)`; no string-built SQL.
- [ ] **Step 6: Run the tests; expect PASS.** Then a real-data check (read-only), printed by a throwaway call: `build_roster(...)` has 141 rows; the count with `name_source` in `{"mcgill","mcgill_later"}` is 130 permno-keyed stocks of 138 (measured 2026-11-24: 130 of 138 permnos appear in McGill). If the counts differ, stop and report; do not adjust the test.
- [ ] **Step 7: Commit** brief, scaffold, code and tests: `git commit -m "research: scaffold dead-stock data-integrity study, brief and roster"`.

---

### Task 2: Label each dead stock's last month from prices alone

**Files:**
- Create: `research/stock-signals/dead-stock-data-integrity/code/price_path.py`
- Test: `research/stock-signals/dead-stock-data-integrity/code/tests/test_price_path.py`

**Interfaces:**
- Produces: `classify_path(close: pd.Series) -> dict` with keys `ratio` (last close divided by the close 21 valid closes earlier), `daily_std` (standard deviation of the last 21 daily simple returns), `label`. Constants at module top: `WINDOW = 21`, `CALM_STD = 0.01`, `COLLAPSE_BELOW = 0.5`, `DRIFT_BELOW = 0.9`, `CALM_BAND = (0.9, 1.1)`. Labels: `too_short` (fewer than 22 valid closes), `collapse_like` (ratio < 0.5), `drifting_down` (0.5 <= ratio < 0.9), `calm_like_deal` (ratio in the calm band and `daily_std <= CALM_STD`), else `unclear`.
- The thresholds repeat the first report's buckets and are assumptions written into the brief before any run. They are descriptive labels only: they cannot set a payout.

- [ ] **Step 1: Write the failing tests.**

```python
def test_flat_prices_are_calm_like_deal():
    close = pd.Series([50.0, 50.01] * 15)
    assert classify_path(close)["label"] == "calm_like_deal"

def test_halving_is_collapse_like():
    close = pd.Series([100.0] * 10 + list(np.linspace(100, 40, 22)))
    assert classify_path(close)["label"] == "collapse_like"

def test_volatile_flat_ratio_is_unclear():
    close = pd.Series([100.0, 112.0] * 15 + [100.0])
    assert classify_path(close)["label"] == "unclear"

def test_short_series_is_too_short():
    assert classify_path(pd.Series([10.0] * 21))["label"] == "too_short"

def test_nan_closes_are_dropped_before_counting():
    close = pd.Series([50.0] * 5 + [np.nan] * 3 + [50.0] * 20)
    assert classify_path(close)["label"] == "calm_like_deal"
```

- [ ] **Step 2: Run to see FAIL.** `scripts/run_tests.sh .../test_price_path.py -v`.
- [ ] **Step 3: Implement `classify_path`**, dropping NaN closes first (the "own valid closes" rule of the first test).
- [ ] **Step 4: Run the tests; expect 5 passed.**
- [ ] **Step 5: Commit** `git commit -m "research: price-path labels for dead stocks (descriptive)"`.

---

### Task 3: Data-quality checks on CRSP rows, tied to the first test's holdings

**Files:**
- Create: `research/stock-signals/dead-stock-data-integrity/code/integrity_checks.py`
- Test: `research/stock-signals/dead-stock-data-integrity/code/tests/test_integrity_checks.py`

**Interfaces:**
- Input frame for every check: `rows` with columns `permno` (int), `date` (datetime64), `prc` (float, CRSP's sign convention: negative means a bid-ask midpoint), `ret` (float), `shrout` (float). Each check returns `pd.DataFrame[check, permno, date, detail]` (empty frame with those columns when nothing is flagged).
- Produces:
  - `check_duplicates(rows) -> DataFrame` (same permno and date twice).
  - `check_bad_price(rows) -> DataFrame` (`prc` null or zero while `ret` is not null).
  - `check_spikes(rows, threshold: float = 0.5, reverse_band: tuple = (0.9, 1.1)) -> DataFrame` (a one-day return beyond ±`threshold` whose compounded two-day return stays inside `reverse_band`; flags the spike day).
  - `check_gaps(rows, calendar: pd.DatetimeIndex, min_days: int = 5) -> DataFrame` (trading days missing inside a stock's own first-to-last span, in a run of at least `min_days`; flags the first missing day; `detail` = "N days missing").
  - `check_stale(rows, run: int = 10) -> DataFrame` (the same absolute price on `run` rows in a row; flags the first date of the run).
  - `check_shares(rows) -> DataFrame` (the last row of a calendar month has a valid price but `shrout` null or <= 0; market cap, and so the size ranking, would be unknown).
  - `read_trades(path: Path) -> pd.DataFrame` (parses the first test's trade files; the `qty` column is text like `np.float64(337.4789998835294)` and must become a float).
  - `holding_intervals(trades: pd.DataFrame, cell: str) -> pd.DataFrame[permno, start, end]` (a position opens when the running quantity of a `PERMNO-<n>` symbol first exceeds 1e-9 and closes when it returns to zero within 1e-9).
  - `flags_inside_holdings(flags: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame` (flags whose date lies inside an open interval for the same permno).
  - `REFERENCE_CELL = "loss0_cost18.3bps_taxlow"` (matches the key style in `holdings_audit`; the builder confirms the exact cell name in the trade file before using it).
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write the failing tests** (exact values):

```python
def test_duplicates_flag_second_copy():
    rows = frame([(1, "2020-01-02", 10.0, 0.0, 5.0), (1, "2020-01-02", 10.0, 0.0, 5.0)])
    assert len(check_duplicates(rows)) == 1

def test_spike_that_reverses_is_flagged_and_real_jump_is_not():
    up_down = frame_returns(1, [0.0, 0.8, -0.44])      # 1.8 * 0.56 = 1.008
    up_stay = frame_returns(2, [0.0, 0.8, 0.0])        # stays up
    flagged = check_spikes(pd.concat([up_down, up_stay]))
    assert flagged["permno"].tolist() == [1]

def test_gap_of_five_flagged_gap_of_four_not():
    cal = pd.bdate_range("2020-01-01", periods=30)
    five = rows_missing(cal, permno=1, missing=slice(10, 15))
    four = rows_missing(cal, permno=2, missing=slice(10, 14))
    out = check_gaps(pd.concat([five, four]), cal)
    assert out["permno"].tolist() == [1]
    assert out.iloc[0]["detail"] == "5 days missing"

def test_stale_run_of_ten_flagged_nine_not():
    assert len(check_stale(constant_prices(1, 10))) == 1
    assert len(check_stale(constant_prices(2, 9))) == 0

def test_negative_price_is_not_a_bad_price():
    rows = frame([(1, "2020-01-02", -10.0, 0.01, 5.0)])
    assert check_bad_price(rows).empty

def test_zero_price_with_a_return_is_flagged():
    rows = frame([(1, "2020-01-02", 0.0, 0.01, 5.0)])
    assert len(check_bad_price(rows)) == 1

def test_missing_shares_on_month_end_flagged():
    rows = frame([(1, "2020-01-30", 10.0, 0.0, np.nan), (1, "2020-01-31", 10.0, 0.0, 0.0)])
    assert check_shares(rows)["date"].dt.day.tolist() == [31]

def test_read_trades_parses_numpy_text(tmp_path):
    path = write_gz(tmp_path, "cell,date,symbol,qty,price,reason\n"
                              "c,2020-01-02,PERMNO-7,np.float64(10.5),3.0,rebalance\n")
    assert read_trades(path)["qty"].iloc[0] == 10.5

def test_holding_interval_and_flag_inside():
    trades = trades_frame([("c", "2020-01-02", "PERMNO-7", 10.0), ("c", "2020-02-03", "PERMNO-7", -10.0)])
    iv = holding_intervals(trades, "c")
    assert iv.iloc[0]["start"] == pd.Timestamp("2020-01-02") and iv.iloc[0]["end"] == pd.Timestamp("2020-02-03")
    inside = flags_inside_holdings(flag(7, "2020-01-15"), iv)
    outside = flags_inside_holdings(flag(7, "2020-03-01"), iv)
    assert len(inside) == 1 and outside.empty
```

- [ ] **Step 2: Run to see FAIL.**
- [ ] **Step 3: Implement all functions above** in `integrity_checks.py`; all checks work on `abs(prc)` where the sign matters, and none mutate the input.
- [ ] **Step 4: Run the tests; expect PASS** (9 tests).
- [ ] **Step 5: Commit** `git commit -m "research: CRSP row checks and holding-interval overlap"`.

---

### Task 4: Phase 1 driver, real run, and the integrity findings (pause for the owner)

**Files:**
- Modify: `research/stock-signals/dead-stock-data-integrity/code/run.py`
- Create (by the run): `research/stock-signals/dead-stock-data-integrity/_integrity.json`, `_roster.csv`
- Create: `research/stock-signals/dead-stock-data-integrity/2026-11-24-dead-stock-integrity-findings.md`
- Test: `research/stock-signals/dead-stock-data-integrity/code/tests/test_run.py`

**Interfaces:**
- Consumes: Tasks 1-3 (`build_roster`, `classify_path`, the six checks, `read_trades`, `holding_intervals`, `flags_inside_holdings`).
- Produces: `run.py --phase integrity --release-id 1 --crsp-load 1 --mcgill-load 2 --eval-end 2024-12-31 --out DIR`. It writes `_roster.csv` (the roster plus `ratio, daily_std, price_label`), and `_integrity.json` with, per check: number of rows flagged, number of stocks flagged, 20 example rows, and `flags_inside_holdings` counts for each of the four trade files (`dev-A`, `dev-B`, `holdout-A`, `holdout-B`) in `results/dead_stock_first_test/`. Atomic writes (`ggTrader.utils.io`). The CRSP scan reads rows from `kaggle_crsp.daily_of(%s)` in batches of 500 permnos (shares `shrcd` 10/11, `exchcd` 1/3, dates from 2004-01-01) so memory stays under the 4 GB cap; the calendar is the dates on which at least 1,000 stocks have a row.

- [ ] **Step 1: Write the failing driver tests** (small fixtures, no database): `test_integrity_phase_writes_json_with_every_check_name` asserts the keys `{"duplicates","bad_price","spikes","gaps","stale","shares"}` and `"holdings_overlap"`; `test_batches_are_500_permnos_each` asserts the batching helper `permno_batches(permnos, size=500)` splits 1,203 ids into 500, 500, 203; `test_refuses_eval_end_after_crsp_end` asserts an error for `--eval-end 2025-01-02`.
- [ ] **Step 2: Run to see FAIL; implement; run to see PASS.**
- [ ] **Step 3: Real run.** `scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-data-integrity/code/run.py --phase integrity --release-id 1 --crsp-load 1 --mcgill-load 2 --eval-end 2024-12-31 --out research/stock-signals/dead-stock-data-integrity`. Expected: exit 0 and two files written. Exit 75 means wait, 137/143 means report the kill, do not retry with a bigger cap.
- [ ] **Step 4: Write the findings note** in plain words: how many of the 141 have a name; the price-label counts; each check's counts and how many flags fall inside periods the rule actually held the stock; the three examples with the largest effect. If a flag sits inside a holding, say which result it could move and by how much only if the data shows it. Do not rerun any strategy.
- [ ] **Step 5: Run `scripts/study.py check research/stock-signals/dead-stock-data-integrity`; expect OK.** Run `scripts/run_tests.sh research/stock-signals/dead-stock-data-integrity/code/tests -v`; expect all pass.
- [ ] **Step 6: Commit** `git commit -m "research: phase 1 integrity run and findings"`.
- [ ] **Step 7: PAUSE (owner gate A).** Report usage so far and show the findings. Ask for a go for Phase 2. The SEC contact is already settled (see Global Constraints), so only the go is needed.

---

### Task 5: Polite, caching SEC downloader

**Files:**
- Create: `research/stock-signals/dead-stock-data-integrity/code/sec_client.py`
- Test: `research/stock-signals/dead-stock-data-integrity/code/tests/test_sec_client.py`

**Interfaces:**
- Produces: `class SecUserAgentMissing(RuntimeError)`, `class SecFetchError(RuntimeError)`, `load_user_agent(environ: Mapping[str, str]) -> str` (raises `SecUserAgentMissing` when `SEC_USER_AGENT` is absent or blank; there is no default), `class SecClient` with `__init__(self, user_agent: str, cache_dir: Path, min_interval: float = 0.25, opener: Callable = urllib.request.urlopen, sleep: Callable = time.sleep, clock: Callable = time.monotonic)` and `get(self, url: str) -> bytes`, plus `manifest(self) -> dict[str, str]` (url to SHA-256 of every cached body). Cache layout: `<cache_dir>/<sha256(url)>.body` and `<sha256(url)>.meta.json` (`url`, `sha256`, `fetched_at`, `status`). Constant `ALLOWED_HOSTS = {"www.sec.gov", "data.sec.gov"}`.

- [ ] **Step 1: Write the failing tests** with a fake opener:

```python
def test_missing_user_agent_raises():
    with pytest.raises(SecUserAgentMissing):
        load_user_agent({})
    with pytest.raises(SecUserAgentMissing):
        load_user_agent({"SEC_USER_AGENT": "  "})

def test_second_get_comes_from_cache(tmp_path):
    client, opener = make_client(tmp_path, body=b"abc")
    assert client.get(URL) == b"abc" and client.get(URL) == b"abc"
    assert opener.calls == 1

def test_rate_limit_status_is_not_cached(tmp_path):
    client, opener = make_client(tmp_path, status=429, body=b"slow down")
    with pytest.raises(SecFetchError):
        client.get(URL)
    assert list(tmp_path.glob("*.body")) == []

def test_html_error_page_with_200_is_not_cached(tmp_path):
    client, _ = make_client(tmp_path, status=200, body=b"<html>Request Rate Threshold Exceeded</html>")
    with pytest.raises(SecFetchError):
        client.get(URL)
    assert list(tmp_path.glob("*.body")) == []

def test_interval_between_requests_is_enforced(tmp_path):
    client, opener = make_client(tmp_path, body=b"x", min_interval=0.25)
    client.get(URL + "1"); client.get(URL + "2")
    assert opener.sleeps and opener.sleeps[0] == pytest.approx(0.25, abs=0.01)

def test_other_host_is_refused(tmp_path):
    client, _ = make_client(tmp_path, body=b"x")
    with pytest.raises(ValueError):
        client.get("https://example.com/a")

def test_user_agent_header_is_sent(tmp_path):
    client, opener = make_client(tmp_path, body=b"x", user_agent="Project Name contact@example.test")
    client.get(URL)
    assert opener.last_headers["User-Agent"] == "Project Name contact@example.test"
```

(An HTML body is refused only when the URL is not an `.htm`/`.html`/index page; the master-index ZIP, submissions JSON and `primary_doc.xml` are the only URLs the study asks for, so the rule is "a body starting with `<html` or `<!doctype html` is refused unless the URL ends in `.htm`, `.html` or `/`".)

- [ ] **Step 2: Run to see FAIL; implement; run to see PASS** (7 tests). Failed fetches retry at most 3 times with a doubling wait starting at 2 seconds, then raise `SecFetchError`.
- [ ] **Step 3: Check disk before any real download:** `df -h /media/thesix` must show at least 5 GB free; create `/media/thesix/ggTrader-data/sec-delistings/` with a short `README.md` (what it is, source, date, the User-Agent policy without the contact string).
- [ ] **Step 4: Commit** `git commit -m "research: caching SEC downloader with fair-access guards"`.

---

### Task 6: Find each stock's Form 25 notice

**Files:**
- Create: `research/stock-signals/dead-stock-data-integrity/code/sec_filings.py`
- Test: `research/stock-signals/dead-stock-data-integrity/code/tests/test_sec_filings.py`

**Interfaces:**
- Consumes: `SecClient.get`; `parse_listing_archive(content: bytes) -> list[dict]` from `research/stock-signals/_lib/sec_master_index.py` (each record has `accession, form, filing_date, parties[{cik, name, filename}]`).
- Produces:
  - `master_index_url(year: int, quarter: int) -> str` = `https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.zip`.
  - `quarters_covering(start: date, end: date) -> list[tuple[int, int]]`.
  - `form25_candidates(client: SecClient, start: date, end: date) -> list[dict]` (keeps forms `25`, `25/A`, `25-NSE`, `25-NSE/A`; drops `8-A12B`; a quarter that fails to download or parse is logged in the returned `skipped` list and the run goes on: return type `tuple[list[dict], list[dict]]`).
  - `normalize_name(name: str) -> str` (uppercase, punctuation to spaces, collapse spaces, drop the trailing words `INC`, `CORP`, `CORPORATION`, `CO`, `COMPANY`, `LTD`, `LLC`, `LP`, `PLC`).
  - `match_roster_row(row: Mapping, candidates: list[dict], before_days: int = 30, after_days: int = 180) -> dict` returning `{"status": "matched"|"ambiguous"|"none", "ciks": list[str], "accessions": list[str]}`. A candidate counts when any of its parties has the same `normalize_name` as `row["stock_name"]` and its `filing_date` lies in `[exit_date - before_days, exit_date + after_days]`. One distinct CIK is `matched`; two or more is `ambiguous`; zero (or a row whose `name_source` is `"none"`) is `none`.
  - `notice_is_common_stock(security_description: str) -> bool` (true when the text, uppercased, contains `COMMON` or `ORDINARY SHARES` and does not contain `PREFERRED`, `WARRANT`, `NOTE`, `DEBENTURE` or `RIGHTS`).
  - `read_notice(client: SecClient, accession: str, cik: str) -> dict` (fetches `https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/` index and `primary_doc.xml`, records both SHA-256s, and calls `parse_bound_exchange_notice` from `_lib/sec_form25.py`; on a notice the parser does not accept (issuer-filed HTML Form 25) return `{"status": "unreadable", ...}` instead of raising).
- The exact-match decision stands for version one: no fuzzy name matching. After the pilot below, the report states the exact-match rate and, if it is under half of the 130 named stocks, lists name-variant examples for the owner to decide on a second, documented matching rule. Do not add one silently.

- [ ] **Step 1: Write the failing tests:**

```python
def test_normalize_name_drops_corporate_suffix():
    assert normalize_name("SiRF Technology Holdings, Inc.") == "SIRF TECHNOLOGY HOLDINGS"

def test_match_requires_window_edges():
    row = {"stock_name": "ALPHA CORP", "name_source": "mcgill", "exit_date": "2010-06-30"}
    on_edge_before = cand("ALPHA CORP", "2010-05-31", cik="0000000001")
    too_early = cand("ALPHA CORP", "2010-05-30", cik="0000000001")
    on_edge_after = cand("ALPHA CORP", "2010-12-27", cik="0000000001")
    too_late = cand("ALPHA CORP", "2010-12-28", cik="0000000001")
    assert match_roster_row(row, [on_edge_before])["status"] == "matched"
    assert match_roster_row(row, [too_early])["status"] == "none"
    assert match_roster_row(row, [on_edge_after])["status"] == "matched"
    assert match_roster_row(row, [too_late])["status"] == "none"

def test_two_filers_with_the_same_name_are_ambiguous():
    row = {"stock_name": "ALPHA CORP", "name_source": "mcgill", "exit_date": "2010-06-30"}
    out = match_roster_row(row, [cand("ALPHA CORP", "2010-07-01", cik="0000000001"),
                                 cand("Alpha Corp.", "2010-07-02", cik="0000000002")])
    assert out["status"] == "ambiguous" and len(out["ciks"]) == 2

def test_same_cik_filing_twice_is_one_match():
    row = {"stock_name": "ALPHA CORP", "name_source": "mcgill", "exit_date": "2010-06-30"}
    out = match_roster_row(row, [cand("ALPHA CORP", "2010-07-01", cik="0000000001"),
                                 cand("ALPHA CORP", "2010-07-09", cik="0000000001", form="25/A")])
    assert out["status"] == "matched" and out["ciks"] == ["0000000001"]

def test_unnamed_row_never_matches():
    row = {"stock_name": None, "name_source": "none", "exit_date": "2010-06-30"}
    assert match_roster_row(row, [cand("ALPHA CORP", "2010-07-01", cik="0000000001")])["status"] == "none"

def test_notice_is_common_stock():
    assert notice_is_common_stock("Common Stock, par value $0.01")
    assert not notice_is_common_stock("Warrants to purchase Common Stock")
    assert not notice_is_common_stock("Series A Preferred Stock")

def test_quarters_covering_spans_year_boundary():
    assert quarters_covering(date(2009, 11, 15), date(2010, 2, 1)) == [(2009, 4), (2010, 1)]

def test_a_failing_quarter_is_skipped_not_fatal():
    candidates, skipped = form25_candidates(client_failing_q2, date(2010, 1, 1), date(2010, 9, 30))
    assert [s["quarter"] for s in skipped] == [(2010, 2)] and candidates
```

- [ ] **Step 2: Run to see FAIL; implement; run to see PASS** (8 tests).
- [ ] **Step 3: Real pilot.** With `SEC_USER_AGENT` exported by the owner's go, run the matching for the 130 named stocks only (quarters from 2005 Q1 to 2024 Q4 that cover an exit date; at most about 80 downloads). Print: stocks matched, ambiguous, none; and for ten matched stocks, the Form 25 security description and rule provision from `read_notice`. Stop and report if more than 10 quarters are skipped.
- [ ] **Step 4: Commit** `git commit -m "research: match dead stocks to SEC Form 25 notices"`.

---

### Task 7: Read the 8-K filings around each matched stock's exit

**Files:**
- Modify: `research/stock-signals/dead-stock-data-integrity/code/sec_filings.py`
- Test: `research/stock-signals/dead-stock-data-integrity/code/tests/test_sec_events.py`

**Interfaces:**
- Consumes: `SecClient.get`; the matched `ciks` from Task 6.
- Produces:
  - `submissions_url(cik: str) -> str` = `https://data.sec.gov/submissions/CIK{int(cik):010d}.json`.
  - `extract_8k_events(submissions: dict, pages: list[dict], around: date, days: int = 90) -> list[dict]` (reads the parallel arrays `form`, `filingDate`, `items`, `accessionNumber`, `primaryDocument` from `submissions["filings"]["recent"]` and from each page of older filings; keeps forms `8-K` and `8-K/A` filed within `days` of `around`; each event is `{"accession", "filing_date", "items": list[str], "primary_document"}` with `items` split on commas).
  - `older_page_urls(submissions: dict) -> list[str]` (from `filings.files[].name`, as `https://data.sec.gov/submissions/{name}`).
  - `evidence_class(events: list[dict]) -> str` with this table, fixed in the brief and applied in order: any item `1.03` gives `bankruptcy_filed`; else any of `2.01`, `3.03`, `5.01` gives `acquired_or_merged`; else item `3.01` gives `delisted_other`; else `no_8k_found`. Item meanings (1.03 bankruptcy or receivership, 2.01 completion of an acquisition or disposition of assets, 3.01 notice of delisting, 3.03 material change to the rights of shareholders, 5.01 change in control) are checked by the builder against the SEC's Form 8-K instructions, fetched through `SecClient` and hashed; if any meaning differs, the builder stops and reports before pinning the table.
  - `fetch_events(client, cik: str, around: date, days: int = 90) -> list[dict]`.
- Item codes exist in the SEC data only for filings after the 2004 form revision. The roster's exits all lie in 2005-2020, so this is not expected to bite, but any event whose `items` field is empty is kept and counted in the coverage numbers as `items_missing`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_extract_events_filters_form_and_window():
    subs = recent_fixture([("8-K", "2010-07-01", "2.01,3.03"), ("10-Q", "2010-07-02", ""),
                           ("8-K", "2010-12-01", "5.02"), ("8-K/A", "2010-07-10", "2.01")])
    out = extract_8k_events(subs, pages=[], around=date(2010, 7, 5), days=90)
    assert [e["filing_date"] for e in out] == ["2010-07-01", "2010-07-10"]
    assert out[0]["items"] == ["2.01", "3.03"]

def test_older_pages_are_read():
    subs = recent_fixture([])
    page = page_fixture([("8-K", "2010-07-01", "1.03")])
    assert extract_8k_events(subs, pages=[page], around=date(2010, 7, 5), days=30)[0]["items"] == ["1.03"]

def test_evidence_class_order():
    assert evidence_class([{"items": ["1.03", "2.01"]}]) == "bankruptcy_filed"
    assert evidence_class([{"items": ["3.03"]}]) == "acquired_or_merged"
    assert evidence_class([{"items": ["3.01"]}]) == "delisted_other"
    assert evidence_class([{"items": ["8.01"]}]) == "no_8k_found"
    assert evidence_class([]) == "no_8k_found"

def test_older_page_urls():
    subs = {"filings": {"files": [{"name": "CIK0000000001-submissions-001.json"}]}}
    assert older_page_urls(subs) == ["https://data.sec.gov/submissions/CIK0000000001-submissions-001.json"]

def test_submissions_url_pads_cik():
    assert submissions_url("0000000042") == "https://data.sec.gov/submissions/CIK0000000042.json"

def test_empty_items_are_kept_and_flagged():
    subs = recent_fixture([("8-K", "2010-07-01", "")])
    out = extract_8k_events(subs, pages=[], around=date(2010, 7, 5), days=30)
    assert out[0]["items"] == []
```

- [ ] **Step 2: Run to see FAIL; implement; run to see PASS** (6 tests).
- [ ] **Step 3: Real pilot** for the stocks Task 6 matched uniquely: print the evidence-class counts and, for five stocks per class, the filing dates and items. Stop and report if more than a third of events have empty `items`.
- [ ] **Step 4: Commit** `git commit -m "research: 8-K item evidence for dead stocks"`.

---

### Task 8: Payout table, coverage, report, close (pause for the owner)

**Files:**
- Create: `research/stock-signals/dead-stock-data-integrity/code/payout_table.py`
- Modify: `research/stock-signals/dead-stock-data-integrity/code/run.py` (adds `--phase sec`)
- Create (by the run): `_payouts.csv`, `_payouts.json`
- Create: `research/stock-signals/dead-stock-data-integrity/2026-11-25-dead-stock-data-integrity.md` (date set to the day it is written; file name `YYYY-MM-DD-slug.md`)
- Modify: `research/RESEARCH_INDEX.md`, `docs/changelog.md`, `research/stock-signals/dead-stock-data-integrity/README.md`, `study.toml`
- Test: `research/stock-signals/dead-stock-data-integrity/code/tests/test_payout_table.py`

**Interfaces:**
- Consumes: the roster (Task 1), price labels (Task 2), `match_roster_row` / `read_notice` (Task 6), `fetch_events` / `evidence_class` (Task 7).
- Produces:
  - `EVIDENCE_CLASSES = ("bankruptcy_filed", "acquired_or_merged", "delisted_other", "no_8k_found", "no_filing_match", "name_unknown")`.
  - `suggested_loss(evidence_class: str, exchcd: int | None) -> tuple[float | None, str]` returning `(loss, basis)`. `acquired_or_merged` gives `(0.0, "filing_evidence")`. `bankruptcy_filed` gives `(0.30, "literature_fill")` for exchange codes 1 and 2 (NYSE, AMEX) and `(0.55, "literature_fill")` for exchange code 3 (Nasdaq); these are the missing-delisting-return fills from Shumway (1997) and Shumway & Warther (1999), which the builder finds in `gg-library` (`search_sources`) and cites with locators in the report. Any other class gives `(None, "none")` and the later rerun keeps all three readings (1.0, 0.5, 0.0) for that stock.
  - `build_payout_table(roster: pd.DataFrame, matches: dict, notices: dict, events: dict) -> pd.DataFrame` (one row per roster row: `permno, symbol, window, universe, exit_date, last_close, stock_name, price_label, evidence_class, form25_accession, form25_provision, form25_security, event_accessions, suggested_loss, basis, cross_check`). `cross_check` is `"conflict"` when a `bankruptcy_filed` stock has price label `calm_like_deal` or an `acquired_or_merged` stock has `collapse_like`, else `"agrees"` or `"not_comparable"`.
  - `coverage_summary(table: pd.DataFrame) -> dict` (counts per evidence class, number with a suggested loss, number of conflicts, `items_missing`, quarters skipped).
- Writes via `ggTrader.utils.io` atomic writers; `_payouts.json` also holds the `SecClient.manifest()` hashes and the skipped quarters.

- [ ] **Step 1: Write the failing tests:**

```python
def test_suggested_loss_table():
    assert suggested_loss("acquired_or_merged", 1) == (0.0, "filing_evidence")
    assert suggested_loss("bankruptcy_filed", 1) == (0.30, "literature_fill")
    assert suggested_loss("bankruptcy_filed", 2) == (0.30, "literature_fill")
    assert suggested_loss("bankruptcy_filed", 3) == (0.55, "literature_fill")
    assert suggested_loss("delisted_other", 1) == (None, "none")
    assert suggested_loss("no_filing_match", 3) == (None, "none")

def test_cross_check_flags_conflicts_both_ways():
    table = build_payout_table(*conflict_fixture())
    assert table.set_index("permno")["cross_check"].to_dict() == {1: "conflict", 2: "conflict", 3: "agrees"}

def test_every_roster_row_survives_into_the_table():
    roster = roster_fixture(n=5)
    table = build_payout_table(roster, matches={}, notices={}, events={})
    assert len(table) == 5 and set(table["evidence_class"]) <= set(EVIDENCE_CLASSES)

def test_unnamed_stock_is_name_unknown():
    table = build_payout_table(roster_fixture(unnamed=True), {}, {}, {})
    assert table["evidence_class"].iloc[0] == "name_unknown"

def test_coverage_summary_counts():
    cov = coverage_summary(table_fixture())
    assert cov["with_suggested_loss"] == 2 and cov["conflicts"] == 1
```

- [ ] **Step 2: Run to see FAIL; implement; run to see PASS** (5 tests).
- [ ] **Step 3: Real run.** `scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-data-integrity/code/run.py --phase sec --release-id 1 --mcgill-load 2 --cache-dir /media/thesix/ggTrader-data/sec-delistings --out research/stock-signals/dead-stock-data-integrity` (with `SEC_USER_AGENT` exported). The run is resumable: the cache means a restart does not repeat finished downloads. Expected: `_payouts.csv` has 141 rows. Write `SHA256SUMS.txt` in the cache folder from the manifest.
- [ ] **Step 4: Write the report** (`research/TEMPLATE-research-report.md` shape, plain words, summary first): how many of the 141 got documented evidence, per class; how many got a suggested loss and on what basis; the conflicts and what they suggest; which gaps remain and why (no volume in the CRSP file; stocks with no matching notice; 2025-2026 rows exist only for stocks still trading; Yahoo-only history for linked stocks); what the table could change in the first test's result if applied (state it as a question for the rerun plan, not as a number; do not compute a rerun here); the exact-match rate and name-variant examples from Task 6; every decision the orchestrator made, with what it costs if wrong; a words-used list. Report no GO/NO-GO label.
- [ ] **Step 5: Wire up the records:** add the row to `research/RESEARCH_INDEX.md`, an entry to `docs/changelog.md`, set `study.toml` `status = "closed"` and `verdict = "UNRESOLVED"` only if the study's `check` accepts it (otherwise follow the first test's close procedure: `scripts/study.py close`), update `README.md`, and write the memory note `project_dead_stock_data_integrity_<date>.md` with an index line.
- [ ] **Step 6: Final checks:** `scripts/study.py check research/stock-signals/dead-stock-data-integrity` prints OK; `scripts/run_tests.sh research/stock-signals/dead-stock-data-integrity/code/tests -v` all pass; `ruff check` and `ruff format --check` clean on the study folder; then `scripts/run_tests.sh --studies` as the single heavy job (expect the previous 1,385 passes plus this study's new tests, no failures).
- [ ] **Step 7: Reviews and commit.** Sol reviews the diff of this task and the whole study; Opus checks every number in the report against `_payouts.json` and `_integrity.json`. Fix findings, then `git commit -m "research: dead-stock payout table, coverage and report"`.
- [ ] **Step 8: PAUSE (owner gate B).** Report usage, the coverage numbers and the conflicts. Ask whether to write the rerun plan (re-run the first test's cells with documented outcomes for the covered stocks and the three readings for the rest) and whether to merge and push. Do neither without the owner.

---

## Self-review

- **Spec coverage:** the owner's request (data integrity and gap filling) maps to Tasks 3-4 (integrity of our rows and their effect on actual holdings) and Tasks 5-8 (filling the payout gap, the largest identified uncertainty). Gaps that free data cannot fill (volume, 2025 onward for dead stocks) are named in the report, not worked around.
- **Not in this plan, by design:** the strategy rerun with documented outcomes (needs coverage numbers first; it would also need the closed study's replay code moved into `_lib`, which changes that study's pinned hashes and needs its own plan); fuzzy name matching; reading 8-K text for deal prices; McGill characteristics wiring.
- **Type consistency:** `roster` columns written in Task 1 (`permno, symbol, window, universe, exit_date, last_close, stock_name, ticker, cusip, exchcd, name_source`) are what Tasks 4, 6 and 8 read; `match_roster_row` returns `status` / `ciks` / `accessions`, which Task 8 consumes as `matches`; `evidence_class` strings are defined once (Task 7 and `EVIDENCE_CLASSES` in Task 8).
- **Proportion:** eight tasks, tests as code, bodies left to the implementer.
- **Unverified assumptions the builders must check and report, not paper over:** the EDGAR `master.zip` quarterly files exist and fit the parser's 30 MB limit; Form 8-K item meanings; the McGill company name is the name at the time of exit (otherwise exact matching will under-match); the trade-file cell name for `REFERENCE_CELL`.
