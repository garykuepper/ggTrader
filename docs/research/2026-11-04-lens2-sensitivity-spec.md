# Lens 2 sensitivity table: fixed specification

**Date:** 2026-11-04 · **Status:** fixed before use · **Investigation:**
`docs/superpowers/specs/2026-11-04-root-cause-investigation-design.md` (sections 1, 2, 5
Lens 2, and 11 items 5 and 7) · **Inventory:** `docs/research/_investigation_inventory.json`

## 1. Purpose and status

Lens 2 (verdict audit) fills one sensitivity table covering every material result in
the inventory, NO-GO and GO alike. This document fixes, in advance, what the table
contains and how each cell is computed.

- **Exploratory only.** The table shows how each result looks against two significance
  levels (t of at least 3 and t of at least 2), with the effect in raw terms. It is
  **not a new pass bar**.
- **No verdict changes on this table alone.** The original pre-registered bar remains
  the verdict for every result. A row that clears t of at least 2 but failed its
  original bar is still a fail under that bar. Any verdict change needs separate
  evidence from another lens (for example, a tape or code defect confirmed by Lens 1
  or Lens 6), and that evidence, not this table, is what changes the verdict.
- **Raw total return is the effect measure** (spec section 1 and item 7): the owner's
  target is raw gains, measured as total return (price plus dividends). Pre-tax is
  primary; after-tax is shown beside it when the artifact has it.
- **Commit rule.** This specification is committed to the investigation branch before
  any Lens 2 work opens a result value. The commit hash of this file must be older
  than every Lens 2 commit and every Lens 2 working note. Lens 2 cites that hash at the
  top of its table. Any later change to this document is a new, dated commit that says
  whether results had been opened when it was made; a change made after opening is
  labelled post hoc in the table header and does not rewrite rows already filled.
- **Authoring rule.** While writing this document, results JSONs were opened only to
  list their key names and value types (section 9). Selections in section 9 were taken
  from the pass-bar, decision-rule and key-naming text of briefs and reports. Some
  result values were visible in report lines read for that purpose; none was used to
  make a selection or shape a rule.

## 2. Unit of the table

- One row per inventory row (`id`), split into sub-rows only by the selection rule
  below.
- Rows are never dropped, merged or reordered by their values. Order follows the
  inventory.

### 2.1 Selection rule (arm, group, hypothesis, bracket, window)

When an artifact holds more than one arm, group, hypothesis, tax bracket or window,
Lens 2 selects with this precedence, decided by text, never by values:

1. **Named by the verdict.** The arm, group, hypothesis, bracket or window that the
   original verdict was judged on, as named in the pre-registration brief or, if the
   brief names none, the source report. Section 9 records the result of this step for
   each listed artifact, with the brief and section.
2. **Literal primary.** If nothing is named: the artifact's key literally called
   `primary`, or the arm referenced by the artifact's top-level `verdict` block.
3. **Do not choose.** If more than one candidate still qualifies: emit one sub-row per
   candidate, labelled `<id>/<key>`, in the artifact's key order.

Windows follow the same precedence. A window the original verdict used in addition to
the primary (for example the production replica's holdout) is reported as **its own
sub-row** labelled with its key (for example `<id>/holdout`), never substituted for the
primary window. If the artifact marks that window as not completed or not citable (for
example a `citable` flag, or a closure field stating the holdout did not complete),
the sub-row's column 15 records that and quotes the field.

### 2.2 One window per cell group

Column 5 (raw CAGR gap) and columns 6, 7, 8, 9 and 11 (CI, t, verdicts, power) must
come from the **same window**. If the pre-registered economic test (CAGR gap) and the
pre-registered statistical test (t) use different windows, emit one sub-row per window
(labelled for example `<id>/<g>/A1` and `<id>/<g>/A2`). In each sub-row, a cell whose
only available key belongs to the other window reads
**"not computable (window mismatch: <window A> vs <window B>)"**. Never pair a gap from
one window with a t from another.
- Rows whose inventory `raw_artifact` is null are kept. Every computed cell in such a
  row reads **"no raw artifact"**. Columns 12 (original verdict and bar) and 14
  (exposure) are still filled from the source report where it states them.

## 3. Fixed columns, in order

| # | Column | Units |
|---|---|---|
| 1 | Result id | inventory `id` (plus `/<key>` sub-row labels per sections 2.1 and 2.2) |
| 2 | Window used | `start..end`, strings copied exactly from the artifact |
| 3 | Strategy raw CAGR | % per year |
| 4 | SPY total-return CAGR over the same window | % per year |
| 5 | Raw CAGR gap vs SPY | percentage points per year |
| 6 | 95% confidence interval of the gap | percentage points per year, with the quantity named |
| 7 | t-statistic of the gap | dimensionless |
| 8 | Verdict at t >= 3 | "clears" / "does not clear" |
| 9 | Verdict at t >= 2 | "clears" / "does not clear" |
| 10 | After-tax gap, if the artifact has it | percentage points per year |
| 11 | Power note | smallest detectable gap at 80% power, pp per year |
| 12 | Original pre-registered verdict and bar | text |
| 13 | Artifact path and JSON key paths used | path plus key paths |
| 14 | Exposure and max-leverage settings | text, as recorded |
| 15 | Standard-error method and tested quantity | text |

Columns 1 to 13 are the required columns in the required order. Columns 14 and 15 are
appended so that the first thirteen keep their order; they are mandatory in every row.

## 4. How each column is computed

General rules for every column:

- Read values from the raw artifact named in the inventory row. Do not read numbers
  from report prose, memory files or chat for any computed column.
- Prefer a key the artifact already reports over a derived value. Derive only with the
  formulas below, and only from fields in the same artifact, same window and same arm.
- Record every key path read in column 13, for example
  `windows.primary.arms.primary.excess_vs_spy.t_stat`.
- Round only for display: two decimals for percentages and t-statistics. Carry full
  precision through derivations.

### Column 2: window used

Read the start and end strings of the window the row covers (for example
`eval_start` / `eval_end`, `window`, `start` / `end`). Copy them exactly. If the
artifact has no window field for that result but the pre-registration brief fixes the
window of the key being read (section 9 says where this applies), copy the brief's
window and append "(window from brief <file> §<n>; not stored in artifact)". Otherwise
write "not computable from artifact (missing: window)". Do not take a window from report
prose or infer one from fold dates.

### Column 3: strategy raw CAGR

1. If the artifact reports a CAGR for the arm and window (for example `cagr_pct`,
   `pre_tax_cagr`), use it.
2. Else, if it reports a total return and the window, derive
   `CAGR = (1 + total_return) ** (1 / years) - 1`, with
   `years = (end - start).days / 365.25` (the convention in
   `src/ggTrader/lab/metrics.py`).
3. Else write "not computable from artifact (missing: <key>)".

### Column 4: SPY total-return CAGR over the same window

- Same two-step rule as column 3, applied to the artifact's SPY series for the same
  window and arm.
- **The comparator must be SPY total return (dividends included).** Establish the
  basis from, in this order: a field in the artifact; the Lens 1 finding on that tape;
  the driver code at the commit that wrote the artifact. Cite which source settled it.
- If the comparator is SPY **price** return: write "price-return comparator: not
  comparable" in column 4, and the same phrase in columns 5 to 11.
- If the basis cannot be established from those three sources: write "comparator basis
  unknown: not comparable" in column 4 and columns 5 to 11.
- If the artifact's comparator is not SPY but a total-return market series (for example
  the Fama-French market return used by the long-history factor screens): fill columns
  4 to 11 from the artifact as reported and prefix column 4 with
  "comparator: <name>, not SPY". Do not swap in SPY data.
- If the artifact has no comparator CAGR or total return for that window: "not
  computable from artifact (missing: <key>)".

### Column 5: raw CAGR gap vs SPY

1. If the artifact reports the CAGR difference directly (for example `cagr_diff_pp`),
   use it.
2. Else `gap_pp = CAGR_strategy - CAGR_SPY` (columns 3 and 4, in percentage points
   per year). This is a simple difference of CAGRs, not a ratio.
3. If column 3 or 4 is not computable, column 5 is "not computable from artifact"
   citing the missing key.

### Column 7: t-statistic of the gap

- **Primary method: Newey-West**, when the artifact provides a Newey-West t-statistic
  (for example `excess_vs_spy.t_stat` from `src/ggTrader/lab/significance.py`, or a
  monthly screen's `t` / `full_t` from `src/ggTrader/lab/momentum_screen.py`). Record
  the lag count and sampling frequency in column 15 from the artifact (for example
  `inputs.nw_lags`) or, if absent there, from the code that wrote it.
- **Else the artifact's own reported statistic**, named in column 15 exactly as the
  artifact or its driver defines it. When this fallback is used, column 15 must also
  record whether the observations behind it overlap (for example overlapping folds or
  rolling windows) or are fold-dependent, as stated in the artifact or its driver code;
  if neither states it, write "overlap/fold dependence: not stated".
- If the artifact reports no t-statistic: "not computable from artifact (missing:
  t-statistic)". Do not compute a t-statistic from equity curves, and do not convert a
  p-value or a Sharpe ratio into one.
- **Never mix methods inside one row.** The t-statistic, the standard error in column
  6 and the power note in column 11 all use the one method named in column 15.
- The t-statistic in these artifacts tests the **mean excess return** (arithmetic,
  usually net of the artifact's costs), not the CAGR gap itself. Column 15 names the
  tested quantity. Column 7 is recorded as reported; it is not rescaled to the CAGR gap.
- Arithmetic mean excess minus roughly half the difference in return variances
  approximates the log-growth (CAGR) gap, so for volatile or levered rows the CI and t
  (columns 6 and 7) and the CAGR gap (column 5) can differ in size and even in sign.
  Never read the column 6 interval as an interval on the CAGR gap.

### Column 6: 95% confidence interval

The interval is on the quantity the t-statistic tests, because that is the only
quantity with a standard error in the artifact.

1. Standard error of the annualized mean excess:
   - if the artifact reports it, use it;
   - else, if the artifact reports both the annualized mean excess (for example
     `ann_excess_pct`) and the t-statistic of column 7, `SE = ann_excess / t`
     (same method as column 7);
   - else "not computable from artifact (missing: <key>)".
2. `CI = ann_excess +/- 1.96 x SE`, written as `[low, high] pp/yr on <tested quantity>`.
3. Do not build an interval around the CAGR gap of column 5 from this standard error,
   and do not estimate a standard error from a CAGR gap and a t-statistic alone. If
   the artifact gives only a CAGR gap and a t-statistic, column 6 is "not computable
   from artifact (missing: annualized mean excess)".
4. If t is zero or infinite, the standard error is not derivable: "not computable from
   artifact (t is <value>)".

### Columns 8 and 9: verdict at t >= 3 and t >= 2

- "clears" if column 7 is greater than or equal to 3 (column 8) or 2 (column 9);
  otherwise "does not clear". A negative t never clears.
- Only the t-statistic decides these columns. No economic threshold (CAGR gap,
  Sharpe, drawdown) is added to them.
- If column 7 is not computable, both columns repeat its "not computable" text.
- These columns are labelled **exploratory** in the table header.

### Column 10: after-tax gap

- Fill only if the artifact itself reports after-tax figures for both the strategy and
  SPY in the same window (for example `after_tax_cagr`). `after_tax_gap_pp =
  after_tax_CAGR_strategy - after_tax_CAGR_SPY`.
- If the artifact reports several tax brackets or wash-sale variants, list each, named
  by its key (for example `24/15: <value>; 40.8/23.8: <value>`). Do not pick one.
- Otherwise write "not computable from artifact (no after-tax fields)". Do not apply a
  tax model.

### Column 11: power note

Smallest annualized mean excess detectable with 80% power, from the artifact's own
sample and noise, at each bar:

- `MDE(bar) = (bar + 0.8416) x SE`, for bar = 3 and bar = 2, where 0.8416 is the
  80th percentile of the standard normal. Write both, for example
  `MDE t>=3: <value> pp/yr; t>=2: <value> pp/yr`.
- SE is the column 6 standard error, with the same method as columns 6 and 7. One
  standard-error method per row, with no exceptions: no tracking-error or
  independent-returns substitute.
- If the column 6 standard error is not computable, column 11 reads "not computable
  from artifact" with the same missing key cited.
- Lens 9 owns the full power analysis; this column is a note only.

### Column 12: original pre-registered verdict and bar

- Verdict: the inventory row's `verdict`.
- Bar: copied with the source cited, taking sources in this precedence:
  pre-registration brief, then source report, then the code constants that encode it
  (such as `T_BAR` and `CAGR_BAR_PP` in `src/ggTrader/lab/significance.py`). If two
  sources disagree, use the higher-precedence one and record the disagreement in
  column 15; Lens 2 does not resolve it.
- If no bar was written before the result: "no pre-registered bar found (looked in:
  <sources>)".

### Column 13: artifact path and JSON key paths used

The inventory `raw_artifact` path, then every key path read for columns 2 to 11 and
14, separated by semicolons. Rows with a null artifact: "no raw artifact".

### Column 14: exposure and max-leverage settings

- Record what the artifact states about exposure and leverage for that arm (for
  example mean stock, SPY and cash exposure; position size; reserve; max leverage;
  target leverage such as 1.3x). Copy key names and values.
- If the artifact records none: "not recorded in artifact".

### Column 15: standard-error method and tested quantity

For example: "Newey-West, 5 lags, daily; tested quantity: mean daily strategy minus
SPY return, annualized x252". Or "Newey-West, 12 lags, monthly, net of 1%/yr cost;
tested quantity: mean monthly net excess vs market". Or "artifact's own statistic:
<name>". Or "none reported".

## 5. Missing data

- Missing field: write exactly **"not computable from artifact"** and cite the missing
  key in parentheses. Never estimate, interpolate, read the value from prose, or
  recompute it from other runs.
- Inventory `raw_artifact` null: write exactly **"no raw artifact"** in every computed
  column.
- Artifact path listed but file absent or unreadable: "no raw artifact (listed path
  missing: <path>)".
- A missing value is never a reason to drop or reorder a row.

## 6. Leverage and exposure

- The gap is reported **as the artifact reports it**. A sleeve that holds 10% stocks,
  a book swept into SPY, and a levered book are not comparable without column 14, which
  is why column 14 is mandatory.
- Rows are **not** re-leveraged, exposure-matched or rescaled. If the artifact itself
  reports an exposure-matched comparison (for example `excess_vs_exposure_matched`),
  it may be quoted in column 14 as a note, but columns 3 to 11 stay on the
  SPY comparison.
- Blend results computed at a max leverage different from production (the lab default
  is 2.0, production is 1.0) carry that setting in column 14 verbatim.

## 7. Worked example (placeholders only)

Table header line Lens 2 must use:
"Exploratory sensitivity table. Not a pass bar. Original pre-registered bars decide
verdicts. Spec commit: `<hash>`."

| id | window | strat CAGR | SPY TR CAGR | gap pp/yr | 95% CI | t | t>=3 | t>=2 | after-tax gap | power note | original verdict and bar | artifact and keys | exposure / leverage | SE method |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `<id>/primary` | `<start>..<end>` | `<value>` | `<value>` | `<value>` | `[<low>, <high>] on <quantity>` | `<value>` | `<clears or not>` | `<clears or not>` | not computable from artifact (no after-tax fields) | `MDE t>=3: <value>; t>=2: <value>` | `<verdict>`; `<bar>` (source `<doc>`) | `<path>`; `<key paths>` | `<keys and values>` | Newey-West, `<lags>`, `<frequency>`; `<quantity>` |
| `<id>` | not computable from artifact (missing: window) | `<value>` | price-return comparator: not comparable | price-return comparator: not comparable | price-return comparator: not comparable | price-return comparator: not comparable | price-return comparator: not comparable | price-return comparator: not comparable | price-return comparator: not comparable | price-return comparator: not comparable | `<verdict>`; `<bar>` | `<path>`; `<key paths>` | not recorded in artifact | none reported |
| `<id>` | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | no raw artifact | `<verdict>`; `<bar>` | no raw artifact | `<from report, or not recorded>` | none reported |

## 8. Not allowed

- No thresholds beyond t >= 3 and t >= 2. No economic cut-offs, no Sharpe cut-offs, no
  "near miss" category.
- No dropping, merging or reordering rows because of their values or because they look
  bad (or good).
- No changing the SPY comparator: no substituting another benchmark, no swapping a
  price-return comparator for total return by fetching new data, no exposure or
  leverage matching.
- No verdict change from this table alone.
- No recommendations in or beside the table: no strategy, parameter, deployment,
  sizing or account suggestions, and no advice to stop, retire or shut down anything.
- No new runs to fill a cell. A cell the artifact cannot fill is "not computable from
  artifact"; any re-run is a Phase 2 item that needs the owner's approval.
- No changing this specification after results are opened, except by a dated,
  labelled post hoc commit as in section 1.

## 9. Appendix: field map and fixed selections for six artifacts

Key paths below were listed with a keys-and-types dump. `<w>` is a window key, `<arm>`
an arm key, `<g>` a group or hypothesis key, `<b>` a bracket key. Each artifact lists
the inventory rows that use it, the selection made under section 2.1 (with the
step that decided it), and the exact key that feeds each column. Selections were
made from brief and report text (pass-bar sections, decision rules, key names), not
from result values.

### `docs/research/_production_replica_results.json`

- Inventory rows: `stocks-production-replica`, `stocks-production-replica-holdout-2010-2020`.
- Selection (step 1): `docs/research/briefs/2026-10-19-production-replica-audit.md` §4
  judges GO "on both windows, in the primary arm". Arm: `primary` only (lag0, cost3,
  cost10 and cash0 are not rows). Windows: `stocks-production-replica` uses
  `windows.primary`; `stocks-production-replica-holdout-2010-2020` uses
  `windows.holdout` as its own row, labelled holdout and never substituted for the
  primary. If `windows.holdout.citable` is false or `closure.holdout` states that the
  holdout did not complete, column 15 of the holdout row quotes that field.
- Column 2: `windows.<w>.eval_start`, `windows.<w>.eval_end`.
- Column 3: `windows.<w>.arms.primary.stats.cagr_pct`.
- Column 4: `windows.<w>.arms.primary.spy_stats.cagr_pct`. Total-return basis is not
  stated in the JSON; establish it per column 4.
- Column 5: `windows.<w>.arms.primary.excess_vs_spy.cagr_diff_pp` (same window as t).
- Column 7: `windows.<w>.arms.primary.excess_vs_spy.t_stat` (Newey-West, daily; lag
  count from `NW_LAGS` in `src/ggTrader/lab/significance.py`, matching brief §4's
  "Newey-West (5 lags)"; not stored in the JSON).
- Column 6 and 11: SE = `excess_vs_spy.ann_excess_pct` / `excess_vs_spy.t_stat`.
  Sample length: `excess_vs_spy.n_obs`.
- `years_for_t`: null in `windows.primary.arms.*`, but a float in
  `windows.holdout.arms.primary`. It is not used by any column.
- Column 10: no numeric field (`closure.after_tax` is a text string). Brief §4 names
  the 24%/15% bracket for the after-tax bar.
- Column 14: `windows.<w>.arms.primary.mean_exposure.{stocks,spy,cash}`,
  `windows.<w>.arms.primary.config.risk.position_pct`,
  `windows.<w>.arms.primary.config.reserve_pct` (config keys sit under each arm, not at
  top level); `windows.<w>.arms.primary.excess_vs_exposure_matched.*` may be quoted as a
  note only.
- Column 12: brief §4 (t >= 3.0 NW(5) and CAGR gap >= +1.0 pt/yr before tax, both
  windows, primary arm; after-tax >= +1.0 pt/yr at 24/15 only if those pass). Verdict
  fields: `verdict.verdict`, `closure.verdict`, `windows.<w>.arms.primary.pretax_pass`.

### `docs/research/_momentum_french_screen_results.json`

- Inventory row: `stocks-momentum-french-screen`.
- Selection (step 1): `docs/research/briefs/2026-10-20-momentum-retest.md` §5
  pre-registers two hypotheses, Big and Small. Groups: one sub-row each for `big` and
  `small`. Not rows: `size_matched.*`, `diag[i]` (subperiods), `stress.*` (all listed
  as diagnostics, not pass criteria, in §5).
- Windows (section 2.2 applies): the statistical test A1 uses 1927-01 → 2026-04 and the
  economic test A2 uses 1994-01 → 2026-04. Emit sub-rows `<id>/<g>/A1` and
  `<id>/<g>/A2`, four in total. The JSON stores neither window; column 2 copies them
  from the brief §5 with the note required by column 2.
- `<id>/<g>/A1`: column 7 = `groups.<g>.full_t` (Newey-West, 12 lags, monthly, base
  cost; `NW_LAGS_MONTHLY` in `src/ggTrader/lab/momentum_screen.py`). Columns 3, 4, 5:
  "not computable (window mismatch: 1927-01..2026-04 vs 1994-01..2026-04)" for
  column 5; columns 3 and 4 "not computable from artifact (missing: CAGR fields)".
  Column 6 and 11: "not computable from artifact (missing: annualized mean excess)".
- `<id>/<g>/A2`: column 5 = `groups.<g>.post_cagr_diff_pp`. Columns 6 to 9 and 11:
  "not computable (window mismatch: 1994-01..2026-04 vs 1927-01..2026-04)". Columns 3
  and 4: "not computable from artifact (missing: CAGR fields)".
- Comparator: Fama-French market (`Mkt-RF + RF`, brief §5), not SPY; column 4 carries
  the "comparator: <name>, not SPY" prefix.
- Column 10, 14: no fields.
- Column 12: brief §3 and §5 (A1 Holm at family-wise one-sided p = 0.00135; A2 >= +1.0
  pt/yr; pass needs both). Verdict fields: `pass.<g>`, `holm.<g>`.

### `docs/research/_factor_screen_results.json`

- Inventory row: `stocks-factor-screen-h1-h4`.
- Selection (step 1): `docs/research/briefs/2026-10-20-factor-screen.md` §4
  pre-registers exactly four hypotheses, all Big. One sub-row per hypothesis H1, H2,
  H3, H4. `small_cap_diagnostics.*` is excluded from the family (brief §4) and is not a
  row.
- Windows (section 2.2 applies): A1 uses `inputs.common_window` and A2 uses
  `inputs.A2_window` (brief §6). Emit `<id>/<g>/A1` and `<id>/<g>/A2` per hypothesis,
  eight in total.
- `<id>/<g>/A1`: column 2 = `inputs.common_window.start` .. `.end`; column 7 =
  `hypotheses.<g>.A1.t` (Newey-West, monthly, lags at `inputs.nw_lags`, base cost).
  Column 5: "not computable (window mismatch: common_window vs A2_window)". Columns 6
  and 11: "not computable from artifact (missing: annualized mean excess)".
- `<id>/<g>/A2`: column 2 = `inputs.A2_window.start` .. `.end`; column 5 =
  `hypotheses.<g>.A2.cagr_diff_pp`. Columns 6 to 9 and 11: "not computable (window
  mismatch: A2_window vs common_window)".
- Columns 3 and 4: no fields ("not computable from artifact (missing: CAGR fields)").
- Comparator: Fama-French market, not SPY; prefix column 4 accordingly.
- Not used: `hypotheses.<g>.diagnostics.*` (subperiods, stress, CAPM).
- Column 10, 14: no fields.
- Column 12: brief §6 (A1 Holm across four at alpha = 0.00135 with ordered thresholds;
  A2 >= +1.0 pt/yr; pass needs both). Bars in JSON: `inputs.holm_alpha`,
  `inputs.A2_bar_pp`. Verdict fields: `hypotheses.<g>.A1.pass_holm`,
  `hypotheses.<g>.A2.pass`, `hypotheses.<g>.overall_pass`.

### `docs/research/_core_spy_blend_20260924.json`

- Inventory row: `stocks-core-spy-blend-keep`.
- Selection (step 1): the decision rule pre-registered in
  `docs/research/2026-09-24-core-spy-blend-keep.md` §1 and in the driver docstring of
  `scripts/core_spy_blend.py` judges KEEP on k = 1. Arm: `live_k1.0` only
  (`core_alone`, `live_k0.0`, `live_k0.5`, `live_k1.5` are not rows).
- Column 2: `meta.eval_start`, `meta.eval_end`.
- Column 3: `live_k1.0.cagr_pct`. Column 4: `spy.cagr_pct` (basis not stated in the
  JSON; establish per column 4).
- Column 5: derived `live_k1.0.cagr_pct - spy.cagr_pct` (same window).
- Columns 6, 7, 8, 9, 11: "not computable from artifact (missing: t-statistic)".
- Column 10: no field.
- Column 14: `live_k1.0.mean_spy_weight`, `core_idle_mean_pct`, `meta.reserve`.
- Column 12: report §1 decision rule (Sharpe above SPY and max drawdown no worse;
  no t or CAGR bar). Verdict field: `decision`.

### `docs/research/_pit_rebaseline_20260923.json`

- Inventory rows: `stocks-sp500-core-pit-rebaseline` → key `ensemble`;
  `stocks-ensemble-ic-pit` → `ensemble_ic`; `stocks-ensemble-kelly-pit` →
  `ensemble_kelly`; `stocks-live-ensemble-defaults-vs-pit-picks` → `ensemble` (the
  live strategy's own key).
- Selection (step 1): the inventory row names the strategy; one key per row. No
  sub-rows.
- Column 2: `<s>.eval_start`, `<s>.eval_end` (also `meta.*`). Fold windows
  (`<s>.fold_rows[i].test_start` / `.test_end`) are not row windows.
- Column 3: `<s>.cagr_pct`.
- Column 4: "not computable from artifact (missing: SPY CAGR; only `<s>.spy_sharpe`)".
- Columns 5 to 11: "not computable from artifact" citing the missing SPY CAGR or
  t-statistic.
- Column 14: no field.
- Column 12: the code-encoded checks under `<s>.verdict.checks.*` (for `ensemble_ic`
  and `ensemble_kelly`), with the report as the next source per column 12
  precedence. Verdict fields: `<s>.verdict.go`, `<s>.n_gate_pass`.

### `docs/research/_static_leverage_aftertax_results.json`

- Inventory rows: `stocks-a23-static-leverage` → arm
  `static 1.3x (70/30) band 8pp`; `stocks-a24-sleeve-trend-rider` → arm
  `A24 rider: 60 SPY + 40 sleeve (10m SMA +/-2%)`. SPY: `SPY buy-and-hold`.
- Selection (step 1): the report `docs/research/2026-09-30-static-fractional-leverage.md`
  §1 headline and §7 addendum judge on window `main_real_sso` at the primary 24%/15%
  bracket. Window: `main_real_sso` only (`holdout_synthetic` and
  `full_1993_2026_stitched` are not rows). Pre-tax columns 3 to 5 read bracket key
  `24/15`.
- Column 2: `windows.main_real_sso.<arm>.24/15.window` (two-element list).
- Column 3: `windows.main_real_sso.<arm>.24/15.pre_tax_cagr`. Column 4:
  `windows.main_real_sso.SPY buy-and-hold.24/15.pre_tax_cagr`. The driver
  `scripts/static_leverage_aftertax.py` builds SPY total return from unadjusted closes
  plus distributions; confirm per column 4.
- Column 5: derived column 3 minus column 4.
- Columns 6, 7, 8, 9, 11: "not computable from artifact (missing: t-statistic)".
- Column 10: `after_tax_cagr` for arm and SPY, listed for every bracket key in key
  order (`24/15`, `24/15 wash`, `40.8/23.8`, `40.8/23.8 wash`), with `24/15` labelled
  primary per the report.
- Column 14: leverage from the arm name; `cost_bps_per_side`; `beta_matched_pretax_cagr.*`
  as a note only.
- Not used: `rolling_10y_pretax_vs_spy.*`, `stress_panel_pretax`, `recovery_days.*`.
- Column 12: the A23 brief `docs/research/briefs/2026-09-30-static-fractional-leverage.md`
  and the report's pre-registered +1.0 pt/yr after-tax hurdle, per column 12
  precedence.
