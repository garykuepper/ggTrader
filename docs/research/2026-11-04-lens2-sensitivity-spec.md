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
  list their key names and value types (section 9). No result value was used to
  shape any rule here.

## 2. Unit of the table

- One row per inventory row (`id`). If the artifact holds several windows (for
  example a primary and a holdout window) and the original verdict used more than one,
  write one line per window, labelled `<id>/<window key>`. If the artifact holds several
  arms, use only the arm that the original verdict was judged on (named in the source
  report or the brief); other arms are not added as rows.
- Rows are never dropped, merged or reordered by their values. Order follows the
  inventory.
- Rows whose inventory `raw_artifact` is null are kept. Every computed cell in such a
  row reads **"no raw artifact"**. Columns 12 (original verdict and bar) and 14
  (exposure) are still filled from the source report where it states them.

## 3. Fixed columns, in order

| # | Column | Units |
|---|---|---|
| 1 | Result id | inventory `id` (plus `/<window key>` when split) |
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
artifact has no window field for that result, write "not computable from artifact
(missing: window)". Do not take a window from the report or infer one from fold dates.

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
  artifact or its driver defines it.
- If the artifact reports no t-statistic: "not computable from artifact (missing:
  t-statistic)". Do not compute a t-statistic from equity curves, and do not convert a
  p-value or a Sharpe ratio into one.
- **Never mix methods inside one row.** The t-statistic, the standard error in column
  6 and the power note in column 11 all use the one method named in column 15.
- The t-statistic in these artifacts tests the **mean excess return** (arithmetic,
  usually net of the artifact's costs), not the CAGR gap itself. Column 15 names the
  tested quantity. Column 7 is recorded as reported; it is not rescaled to the CAGR gap.

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
- SE is the column 6 standard error (same method). If that is not computable but the
  artifact reports an annualized tracking error and a sample length, use the
  independent-returns approximation `SE = TE / sqrt(years)` (years from `n_obs /
  periods_per_year`, or from the window as in column 3), and say so in column 15
  ("power SE: TE/sqrt(years), not Newey-West"). This is the only place a second
  method may appear in a row, and it must be labelled.
- If neither is available: "not computable from artifact (missing: <key>)".
- Lens 9 owns the full power analysis; this column is a note only.

### Column 12: original pre-registered verdict and bar

- Verdict: the inventory row's `verdict`.
- Bar: copied from the pre-registration (brief, report section on the pass bar, or the
  code constants that encode it, such as `T_BAR` and `CAGR_BAR_PP` in
  `src/ggTrader/lab/significance.py`), with the source cited.
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

## 9. Appendix: field map for six artifacts (key names only)

Key paths below were listed with a keys-and-types dump; no values were read. `<w>` is a
window key, `<arm>` an arm key, `<g>` a group or hypothesis key, `<b>` a bracket key.

### `docs/research/_production_replica_results.json`

- Window: `windows.<w>.eval_start`, `windows.<w>.eval_end` (`<w>` = primary, holdout).
- Strategy CAGR: `windows.<w>.arms.<arm>.stats.cagr_pct` (also `total_return_pct`).
  Arms: primary, lag0, cost3, cost10, cash0.
- SPY CAGR: `windows.<w>.arms.<arm>.spy_stats.cagr_pct`. Total-return basis is not
  stated in the JSON; establish it per column 4.
- Gap: `windows.<w>.arms.<arm>.excess_vs_spy.cagr_diff_pp`.
- t-statistic: `windows.<w>.arms.<arm>.excess_vs_spy.t_stat` (Newey-West, daily; lag
  count from `NW_LAGS` in `src/ggTrader/lab/significance.py`, not stored in the JSON).
- CI: no field. Derivable from `excess_vs_spy.ann_excess_pct` and `t_stat`.
- Power inputs: `excess_vs_spy.n_obs`, `excess_vs_spy.tracking_error_pct`;
  `years_for_t` is present but null.
- After-tax: no numeric field (`closure.after_tax` is a text string).
- Exposure: `windows.<w>.arms.<arm>.mean_exposure.{stocks,spy,cash}`,
  `config.risk.position_pct`, `config.reserve_pct`; also `excess_vs_exposure_matched.*`.
- Verdict fields: `verdict.verdict`, `closure.verdict`, `windows.<w>.arms.<arm>.pretax_pass`.

### `docs/research/_momentum_french_screen_results.json`

- Window: no full-window field. Only sub-period windows exist
  (`groups.<g>.diag[i].start`, `.end`) and `groups.<g>.worst_12m.start` / `.end`.
- Strategy CAGR and comparator CAGR: no fields.
- Gap: `groups.<g>.post_cagr_diff_pp` (post-period only; `<g>` = big, small),
  `groups.<g>.diag[i].cagr_diff_pp`, `groups.<g>.size_matched.cagr_diff_pp`.
- t-statistic: `groups.<g>.full_t` (Newey-West, monthly, 12 lags per
  `NW_LAGS_MONTHLY` in `src/ggTrader/lab/momentum_screen.py`), `groups.<g>.full_p`;
  stressed-cost variants under `groups.<g>.stress.*`.
- CI: no field; no annualized mean excess, so not derivable.
- After-tax: no field. Exposure: no field.
- Comparator: Fama-French market series, not SPY.
- Verdict fields: `pass.<g>`, `holm.<g>`.

### `docs/research/_factor_screen_results.json`

- Window: `inputs.common_window.start` / `.end` / `.n_months`, and
  `inputs.A2_window.start` / `.end` / `.n_months`.
- Strategy CAGR and comparator CAGR: no fields.
- Gap: `hypotheses.<g>.A2.cagr_diff_pp` (`<g>` = H1..H4); diagnostics under
  `hypotheses.<g>.diagnostics.*`.
- t-statistic: `hypotheses.<g>.A1.t`, `.p` (Newey-West, monthly; lags at
  `inputs.nw_lags`).
- CI: no field; no annualized mean excess, so not derivable.
- After-tax: no field. Exposure: no field.
- Comparator: Fama-French market series, not SPY.
- Bars: `inputs.holm_alpha`, `inputs.A2_bar_pp`; verdict fields
  `hypotheses.<g>.A1.pass_holm`, `hypotheses.<g>.A2.pass`, `hypotheses.<g>.overall_pass`.

### `docs/research/_core_spy_blend_20260924.json`

- Window: `meta.eval_start`, `meta.eval_end`.
- Strategy CAGR: `core_alone.cagr_pct`, `live_k<k>.cagr_pct` (k = 0.0, 0.5, 1.0, 1.5).
- SPY CAGR: `spy.cagr_pct`. Basis not stated in the JSON.
- Gap: no field; derive from the two CAGRs.
- t-statistic: no field. CI: no field.
- After-tax: no field.
- Exposure: `live_k<k>.mean_spy_weight`, `core_idle_mean_pct`, `meta.reserve`.
- Verdict field: `decision` (text).

### `docs/research/_pit_rebaseline_20260923.json`

- Window: `meta.eval_start`, `meta.eval_end`; per strategy `<s>.eval_start`,
  `<s>.eval_end` (`<s>` = ensemble, ensemble_ic, ensemble_kelly); fold windows
  `<s>.fold_rows[i].test_start` / `.test_end`.
- Strategy CAGR: `<s>.cagr_pct`.
- SPY CAGR: no field (only `<s>.spy_sharpe`).
- Gap, t-statistic, CI, after-tax: no fields.
- Exposure: no field.
- Verdict fields: `<s>.verdict.go`, `<s>.verdict.checks.*`, `<s>.n_gate_pass`.

### `docs/research/_static_leverage_aftertax_results.json`

- Window: `windows.<w>.<arm>.<b>.window` (two-element list) and `.years`; `<w>` =
  main_real_sso, holdout_synthetic, full_1993_2026_stitched.
- Strategy CAGR: `windows.<w>.<arm>.<b>.pre_tax_cagr`.
- SPY CAGR: `windows.<w>.SPY buy-and-hold.<b>.pre_tax_cagr`. The driver builds SPY
  total return from unadjusted closes plus distributions
  (`scripts/static_leverage_aftertax.py`); confirm per column 4.
- Gap: no direct field; derive from the two CAGRs in the same window and bracket.
- t-statistic: no field. CI: no field.
- After-tax: `windows.<w>.<arm>.<b>.after_tax_cagr` for strategy and SPY;
  `<b>` = 24/15, 24/15 wash, 40.8/23.8, 40.8/23.8 wash.
- Exposure: leverage is in the arm name (for example "static 1.3x (70/30) band 8pp");
  also `beta_matched_pretax_cagr.*`, `cost_bps_per_side`.
- Other: `rolling_10y_pretax_vs_spy.<arm>.*`, `stress_panel_pretax`.
