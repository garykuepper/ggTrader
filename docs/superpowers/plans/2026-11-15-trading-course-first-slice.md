# Trading Course Site, First Slice: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the site engine for `/home/flynn/ggTraderLearn` (a new private learning site
copied from ggPackMind's study engine) plus lesson 5 (costs) and lesson 13 (the two testing
traps), served on the tailnet, then stop for the owner's review.

**Architecture:** A static site built by one Python script from `study/topics.toml` and one
Markdown file per lesson. Interactive pieces are ES modules that compute only through
`study/js/trademath.js`, which a Python reference (`tests/ref_trademath.py`) mirrors function
for function. Node runs the JS in tests. The build checks everything before writing and
writes nothing on any error.

**Tech Stack:** Python 3 standard library + `python-markdown` (system package
`python3-markdown`), `tomllib`, plain ES modules, MathJax 2.7.9 from
`/usr/share/javascript/mathjax`, Node for tests, `python3 -m http.server`, systemd user unit,
Tailscale Serve.

**Spec:** `/home/flynn/ggTrader/docs/superpowers/specs/2026-11-15-trading-course-site-design.md`
(read it first; section 6 lists which ggPackMind checks are kept, replaced or dropped).

**Source engine:** `/home/flynn/ggPackMind` (read `AGENTS.md`, Study section). Copy from it.
Never edit it and never import from it.

## Global Constraints

- New repo at `/home/flynn/ggTraderLearn`, its own `git init`. Nothing in ggPackMind or
  ggTrader changes except this plan and the spec.
- Tailnet only. Local server binds `127.0.0.1:8097`. Tailscale Serve publishes HTTPS port
  `8452` (8443 and 8446 to 8451 are taken on 2026-11-15; re-check with `ss -ltn` first).
- Free sources only. Rules that change carry an "as of" date and a source URL, verified on
  the day the lesson is written.
- Every number is (a) from a cited source, (b) from a ggTrader report with its window and
  commit, or (c) labeled "illustrative".
- Facts are stated plainly with sources. The course never tells the owner what to do with
  their own trading, never recommends stopping trading, retiring the stock sleeve, ending
  paper trading or the project, and never mentions IRAs or retirement accounts.
- ggTrader's results are daily-bar or longer. Never present them as day-trading evidence.
- No links to ggTrader (repo or reports). External links open in a new tab with
  `rel="noopener noreferrer"`.
- Plain words. Every term defined on first use. The owner's private account figures never
  appear, and delegate prompts never contain them.
- Theme: `gg-theme` skill, default palette. Charts: `dataviz` skill rules plus ggPackMind's
  (direct labels, second series dashed, "How to read it" box after every chart).
- Delegation (ggTrader `CLAUDE.md`, 2026-11-08): builders Sonnet, Haiku, Codex Terra, Codex
  Luna; reviewers Codex Sol and Claude Opus, never the builder's own family. The orchestrator
  reruns every check itself. Every figure Haiku extracts is checked by a stronger model.

## Review Focus

1. Calculator inputs at the edges (win rate 0 or 1, average loss 0, zero trades, negative
   cost): the reader expects a clear message, never `NaN`, `Infinity` or a blank. Test in
   Task 5 and Task 6.
2. Best-of-N with N = 1 and very large N (10,000): `inverse normal(0)` is minus infinity, so
   the expected best Sharpe of one trial must be exactly 0 and large N must stay finite. Test
   in Task 3.
3. JS and Python agreeing in the tails (`normInv(1e-9)`, `normInv(1 - 1e-9)`): a lazy
   approximation drifts and the mirror test would hide it at mid-range inputs. Test in
   Task 3.
4. A case study missing its holding period, window, report or commit: the reader would take
   a daily result as day-trading evidence. The build must fail. Test in Task 4.
5. Reading on a phone with no network: charts and drills must work offline and external
   links simply fail to open. Check in Task 7 at 375 px width.

---

### Task 1: Copy the engine and strip the thermal course

**Files:**
- Create: `/home/flynn/ggTraderLearn/` with `site.py`, `study/build_study.py`,
  `study/check_wiki.py`, `study/topics.toml` (empty list), `study/content/`,
  `study/diagrams/`, `study/js/{calc,chart,drill,diagram,typeset,validation}.js`,
  `study/js/topics/`, `theme/`, `tests/node_eval.mjs`, `tests/test_study_build.py`,
  `AGENTS.md`, `README.md`, `.gitignore` (`public/`, `__pycache__/`)
- Do not copy: `build.py` (the paper library), `papers.toml`, `physics.js`, `units.js`,
  `data737.js`, `ref_physics.py`, `ref_737.py`, `tests/test_units.py`, any topic or content
  file, `public/library/`

**Interfaces:**
- Produces: `site.py` with `SITE: str` ("Trading fundamentals"), `head_links(up: str) -> str`,
  `topnav(up: str, on: str, cheat: bool = True) -> str`, `esc(s: str) -> str`, taken from
  ggPackMind `build.py` with the library nav item removed. `study/build_study.py` imports
  these from `site` instead of `build`. `main(root: Path, mathjax: Path) -> int` keeps its
  signature (0 on success, 1 on any error, nothing written on error).
- The theme uses the gg-theme default palette ("ggbsite" teal and amber), not `tokens-rose`.

- [ ] **Step 1:** Copy the files listed, `git init`, first commit "chore: copy study engine
  from ggPackMind".
- [ ] **Step 2: Write the failing test** `tests/test_study_build.py::test_empty_course_builds`:
  with an empty `topics.toml` and an empty `content/`, `main(tmp_root, MATHJAX)` returns 0
  and writes `public/index.html` and `public/study/index.html`, and neither page contains
  "737", "Boeing", "ECS" or "library.html".
- [ ] **Step 3:** Run `python3 -m unittest tests.test_study_build -v`. Expected: FAIL.
- [ ] **Step 4:** Remove from `build_study.py`: the Boeing `FOOTER` (new footer: "Numbers are
  cited, from ggTrader's research, or labeled illustrative."), the 737 checks, `si_first`,
  the kelvin check, the em-dash and semicolon voice rule (Task 4 replaces it), and the paper
  ID check (Task 4 replaces it). Keep the math balance, missing-module, broken-link, wiki-link,
  diagram-lint and "How to read it" checks, and their existing tests.
- [ ] **Step 5:** Run `python3 -m unittest -v`. Expected: all copied tests that still apply
  pass. Delete tests that only cover removed rules, and list each deleted test in the
  commit message.
- [ ] **Step 6:** Commit "feat: strip thermal course, rename header module".

### Task 2: Number formats

**Files:**
- Create: `study/js/formats.js`, `tests/test_formats.py`

**Interfaces:**
- Produces: `fmt(value: number, kind: "usd"|"pct"|"bps"|"count"|"years"|"ratio",
  digits?: number) -> string` and `parseInput(text: string, kind) -> number|null`. `pct`
  takes a fraction (0.0125 → "1.25%"). `bps` takes a fraction (0.0005 → "5 bps"). `usd` uses
  a thousands separator and a leading minus ("-$1,250.00"). Non-finite input returns "n/a".
  `calc.js` and `drill.js` call these where they called `units.js`.

- [ ] **Step 1: Write the failing test** `test_formats_match_table`, using Node through
  `tests/node_eval.mjs`, with these exact cases: `fmt(0.0125,"pct")=="1.25%"`,
  `fmt(0.0005,"bps")=="5 bps"`, `fmt(-1250,"usd")=="-$1,250.00"`, `fmt(1234567,"count")=="1,234,567"`,
  `fmt(NaN,"pct")=="n/a"`, `fmt(Infinity,"usd")=="n/a"`, `parseInput("1.25%","pct")==0.0125`,
  `parseInput("abc","usd") is None`.
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement `formats.js`. Rewire `calc.js` and `drill.js` off `units.js`
  (drill answers are compared as numbers with a relative tolerance of 1%, as now).
- [ ] **Step 4:** Run `python3 -m unittest -v`. Expected: PASS.
- [ ] **Step 5:** Commit "feat: money, percent and count formats".

### Task 3: Trading-math module and its Python mirror

**Files:**
- Create: `study/js/trademath.js`, `tests/ref_trademath.py`, `tests/test_trademath.py`

**Interfaces (all rates are fractions, Sharpe ratios are annual, `years` are of daily data):**
- Costs (lesson 5):
  - `roundTripCost(spread, slippage, feesUsd, notional) -> number`: spread + 2·slippage +
    feesUsd/notional, as a fraction of notional. `spread` is the full bid-ask spread as a
    fraction of price; crossing it in and out costs one full spread in total.
  - `breakEvenWinRate(avgWin, avgLoss, cost) -> number|null`: the win rate p where
    p·(avgWin − cost) − (1 − p)·(avgLoss + cost) = 0. Returns `null` when
    avgWin + avgLoss ≤ 0.
  - `expectancy(winRate, avgWin, avgLoss, cost) -> number`: average profit per trade after
    cost.
  - `annualCostDrag(cost, tradesPerYear) -> number`: 1 − (1 − cost)^tradesPerYear.
- Statistics (lesson 13):
  - `normCdf(x)`, `normInv(p)`: `normInv` uses Acklam's rational approximation with one
    Halley refinement step, so it agrees with Python's `statistics.NormalDist().inv_cdf`
    to 1e-9 relative over p in [1e-9, 1 − 1e-9]. The Python reference uses `NormalDist`.
  - `sharpeStdErr(sr, years) -> number`: sqrt((1 + sr²/2) / years) (Lo 2002, normal returns).
  - `minDetectableSharpe(years, alpha = 0.05, power = 0.8) -> number`:
    (z_{1−alpha} + z_{power}) / sqrt(years), one-sided, small-Sharpe approximation.
  - `powerToDetect(trueSr, years, alpha = 0.05) -> number`:
    1 − normCdf(z_{1−alpha} − trueSr·sqrt(years)).
  - `chanceOfFalseWinner(nTrials, alpha = 0.05) -> number`: 1 − (1 − alpha)^nTrials.
  - `effectiveTrials(nTrials, avgCorr) -> number`: nTrials·(1 − avgCorr) + avgCorr, clamped
    to [1, nTrials]. Label it in the lesson as a rough rule, not a formula from the paper.
  - `expectedMaxSharpe(nTrials, years) -> number`: Bailey and López de Prado's expected
    maximum of nTrials null Sharpe estimates with variance 1/years:
    sqrt(1/years)·((1 − γ)·normInv(1 − 1/N) + γ·normInv(1 − 1/(N·e))), γ = 0.5772156649.
    Returns exactly 0 for N = 1.
  - `deflatedSharpe(sr, srStar, years, skew = 0, kurt = 3) -> number`: probability that the
    true Sharpe exceeds srStar, with T = 252·years daily observations and per-period Sharpe
    sr/sqrt(252) (Bailey and López de Prado 2014, equation for PSR evaluated at srStar).
  - `holm(pValues: number[], alpha = 0.05) -> boolean[]`: Holm step-down rejections, in the
    input order.
  - `mulberry32(seed) -> () => number` and `nullBestSharpes(nTrials, years, runs, seed) ->
    number[]`: simulated best-of-N Sharpe under no edge, for the chart. The Python reference
    implements the same `mulberry32` bit for bit and Box-Muller normals.

- [ ] **Step 1: Write the failing test** `test_every_function_matches_python` in the style
  of ggPackMind's `tests/test_study_physics.py`: a `CASES` list covering every function at
  ordinary inputs, plus these pinned cases: `expectedMaxSharpe(1, 5) == 0`;
  `expectedMaxSharpe(10000, 5)` finite and > `expectedMaxSharpe(100, 5)`;
  `normInv(1e-9)` and `normInv(1 - 1e-9)` match `NormalDist` within 1e-9 relative;
  `chanceOfFalseWinner(72) ≈ 0.9751` (to 4 places); `minDetectableSharpe(5) ≈ 1.112`
  (to 3 places); `breakEvenWinRate(0.01, 0.01, 0) == 0.5`;
  `breakEvenWinRate(0.01, 0.01, 0.002) == 0.6`; `breakEvenWinRate(0, 0, 0) is None`;
  `holm([0.01, 0.04, 0.03], 0.05) == [True, False, False]`;
  `nullBestSharpes(10, 5, 3, 42)` identical in JS and Python.
- [ ] **Step 2:** Run `python3 -m unittest tests.test_trademath -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `ref_trademath.py` first, then `trademath.js` to match.
- [ ] **Step 4:** Run it. Expected: PASS.
- [ ] **Step 5:** Commit "feat: trading-math module mirrored in Python".

### Task 4: Sources, case studies and the plain-words check

**Files:**
- Create: `study/sources.toml`, `study/casestudies.toml`, `tools/check_reading_edition.py`
  (copied from ggTrader `scripts/check_reading_edition.py`, with its import of
  `check_lens_report._check_paths` copied alongside as `tools/check_lens_report.py` or
  inlined), `tests/test_sources.py`
- Modify: `study/build_study.py`

**Interfaces:**
- `sources.toml`: `[[source]]` with `id` (`S` + digits), `title`, `author`, `year`, and
  either `library_id` + `locator` (gg-library) or `url` + `as_of` (ISO date, web source).
  Rule sources (any `kind = "rule"`) must have `url` and `as_of`.
- Citation syntax in prose: `[S12]`. Renders as a link to `sources.html#S12`, a page of
  source cards (title, author, year, locator or URL, as-of date). No files are served.
- `casestudies.toml`: `[[case]]` with `id` (`C` + digits), `report` (title), `report_path`
  (path inside ggTrader, shown as text, never linked), `report_date`, `window` (start and end
  dates), `holding` (`"intraday"|"daily"|"monthly"|"multi-year"`), `commit` (ggTrader SHA),
  `engine` (`"legacy"` before 2026-11-06, else `"vectorbt-1.2"`), and `values` (a table of
  named numbers).
- Case block in prose: `<div class="case" data-id="C3">…</div>`. The build renders a label
  line from the TOML ("Daily holding · 2021-01-31 to 2026-06-01 · legacy engine") above it.
- `source_errors(slug, text)` now returns errors for: an unknown `[S..]` ID; an unknown case
  ID; a case missing any required field; `holding = "daily"` or longer inside a paragraph
  that contains the words "day trading" or "intraday" without the word "not"; the plain-words
  check (em dashes, undefined abbreviations) from `tools/check_reading_edition.py`.
- Worked examples keep ggPackMind's section check with the sections
  `**Inputs`, `**What we're solving`, `**Steps`, `**Answer`, `**Assumptions`.

- [ ] **Step 1: Write the failing tests** in `tests/test_sources.py`:
  `test_unknown_source_fails`, `test_rule_source_needs_as_of`,
  `test_case_missing_commit_fails`, `test_case_missing_holding_fails`,
  `test_case_label_rendered` (asserts the label text above), `test_citation_links_to_card`,
  `test_em_dash_fails`, `test_example_missing_assumptions_fails`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement the TOML loaders and checks in `build_study.py`, and the
  `sources.html` page.
- [ ] **Step 4:** Run `python3 -m unittest -v`. Expected: PASS.
- [ ] **Step 5:** Commit "feat: sources, case studies and plain-words check".

### Task 5: Lesson 5, Costs

**Files:**
- Create: `study/content/costs.md`, `study/js/topics/costs.js`, `study/diagrams/orderbook.svg`
- Modify: `study/topics.toml`, `study/sources.toml`, `tests/test_study_topics.py` (new cases)

**Interfaces:**
- Consumes: `roundTripCost`, `breakEvenWinRate`, `expectancy`, `annualCostDrag` (Task 3);
  `fmt` (Task 2); citation and case syntax (Task 4).
- Produces: topic `slug = "costs"`, `title = "Costs: what a trade must earn to break even"`,
  `calc` (inputs: price, shares, spread in cents, slippage in bps, fees in dollars, average
  win, average loss, win rate, trades per year), `drills` (at least 3), chart `breakeven`
  (break-even win rate against cost per round trip, for 1:1 and 2:1 win:loss sizes), diagram
  `orderbook` (bid, ask, spread, and the cost of crossing it, live from the calculator).

Content requirements (the builder writes the prose):
- Opens with "What this is and why you care". Defines spread, slippage, market impact,
  commission, regulatory fees and round trip on first use.
- One worked example with illustrative numbers.
- Regulatory fees (SEC Section 31 fee, FINRA trading activity fee) and Alpaca's commission
  and fee schedule, each cited with URL and an "as of" date verified on the writing day.
- Lesson 5 has no ggTrader case study unless one honestly fits; costs in ggTrader's daily
  research may be shown only labeled "daily holding".

- [ ] **Step 1: Write the failing tests:** `test_costs_calc_matches_reference` (calculator
  outputs equal `ref_trademath` at the default inputs and at three more input sets),
  `test_costs_edge_inputs` (win rate 0 and 1, average loss 0, zero trades per year: every
  output is a finite number or the message "Not defined for these inputs", never NaN),
  `test_costs_drills_answers` (each drill's answer equals the reference over 20 seeds).
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Write `costs.js`, `orderbook.svg` and `costs.md`; add sources.
- [ ] **Step 4:** Run `python3 -m unittest -v` and `python3 study/build_study.py`.
  Expected: PASS and a clean build.
- [ ] **Step 5:** Commit "feat: lesson 5, costs".

### Task 6: Lesson 13, The two testing traps

**Files:**
- Create: `study/content/two-traps.md`, `study/js/topics/two-traps.js`
- Modify: `study/topics.toml`, `study/sources.toml`, `study/casestudies.toml`,
  `tests/test_study_topics.py`

**Interfaces:**
- Consumes: Task 3 statistics functions, Task 4 citation and case syntax.
- Produces: topic `slug = "two-traps"`, `title = "Two testing traps: false winners and too
  little data"`, `calc` (inputs: number of ideas tried, average correlation between them,
  years of data, best Sharpe found; outputs: effective number of tests, chance at least one
  looks good by luck, expected best Sharpe with no edge, deflated Sharpe probability,
  smallest Sharpe the data could detect), `drills` (at least 4), charts `bestofn`
  (simulated best Sharpe under no edge against number of ideas, from `nullBestSharpes`, with
  `expectedMaxSharpe` drawn over it) and `power` (chance of detecting a true Sharpe of 0.3,
  0.5 and 1.0 against years of data).

Case-study figures (extract, then verify; never quote from memory):
- **C1 False winners:** the ensemble's once-headline Sharpe of 1.12 and what it became after
  the SPY duplicate-row fix and the point-in-time membership fix (expected about 0.97, then
  0.59; confirm). Start from `research/RESEARCH_SNAPSHOT.md` and the reports it links for
  the 2026-08-19 baseline re-run and the 2026-09-23 point-in-time fix. Note that the 1.12's
  window is unrecoverable.
- **C2 Too little data:** the root-cause audit's 72 measured tests, none reliable, none
  showing an edge, none ruling out 1 to 4 percentage points a year:
  `research/audits/root-cause-2026-11-04/2026-11-05-executive-summary.md` and its findings.
- **C3 Counting ideas:** the research-process audit (2026-11-01): 38 registered names
  include voters, variants and shared machinery, and a NO-GO count cannot say how many
  independent ideas failed.
- **C4 Corrections in practice:** the factor screen's real Holm and deflated-Sharpe
  thresholds and its best t-statistic:
  `research/stock-signals/factor-screen/2026-10-21-factor-screen-nogo.md` and
  `_factor_screen_results.json`.
- All four are `holding = "daily"` or longer. The lesson says plainly that they do not
  measure day trading.

Content requirements:
- Both traps, base rates (few ideas are true beforehand), and the arithmetic of active
  management after costs, each cited (gg-library first: search `Trading` and
  `Mathematics-Statistics` for Bailey and López de Prado, Harvey and Liu, Lo 2002, Sharpe's
  "arithmetic of active management").
- States plainly what ggTrader's evidence does and does not show. No advice on what to do.

- [ ] **Step 1: Extract** the C1 to C4 figures (Haiku is fine), each with file, line and
  commit (`git -C /home/flynn/ggTrader rev-parse HEAD`). A stronger model re-reads every
  figure at its source before it goes into `casestudies.toml`.
- [ ] **Step 2: Write the failing tests:** `test_two_traps_calc_matches_reference` (default
  and three more input sets), `test_two_traps_edge_inputs` (1 idea, 10,000 ideas,
  correlation 0 and 1, 1 year of data: finite outputs or the "Not defined" message),
  `test_two_traps_drills_answers` (20 seeds), `test_bestofn_chart_seeded` (chart points
  equal `ref_trademath.nullBestSharpes` with the same seed).
- [ ] **Step 3:** Run them. Expected: FAIL.
- [ ] **Step 4:** Write `two-traps.js` and `two-traps.md`; add sources and cases.
- [ ] **Step 5:** Run `python3 -m unittest -v` and `python3 study/build_study.py`.
  Expected: PASS and a clean build.
- [ ] **Step 6:** Commit "feat: lesson 13, the two testing traps".

### Task 7: Serve it on the tailnet and hand over for review

**Files:**
- Create: `~/.config/systemd/user/trading-course.service` (copy of
  `research-library.service`, port 8097, directory `/home/flynn/ggTraderLearn/public`),
  `AGENTS.md` (build, test, serve, the rules above), `README.md`

- [ ] **Step 1:** Re-check ports with `ss -ltn`. Then `systemctl --user daemon-reload &&
  systemctl --user enable --now trading-course.service` and
  `tailscale serve --bg --https=8452 http://127.0.0.1:8097`.
- [ ] **Step 2:** Verify: `curl -s http://127.0.0.1:8097/study/costs.html | grep -c "How to
  read it"` is at least 1, and `tailscale serve status` lists `:8452 (tailnet only)`.
- [ ] **Step 3:** Open both lessons at 375 px width with the network cut off (browser
  offline mode). Charts, calculators and drills work, and nothing scrolls sideways.
- [ ] **Step 4:** Final review: Codex Sol reviews the Claude-built tasks and Claude Opus
  reviews the Codex-built ones, each with a fresh context, the spec and the diff. Opus also
  checks every case-study figure against its source. Fix findings, rerun
  `python3 -m unittest -v`.
- [ ] **Step 5:** Commit "docs: agent guide and readme", then **stop**. Send the owner the
  URL `https://encom.tail918610.ts.net:8452/` and ask for their review of the format before
  lessons 1 to 4, 6 to 12 and 14 to 19 are planned.
