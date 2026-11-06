# Root-cause investigation, 2026-11-04: start here

## What this folder is

This folder holds the investigation into why the project's stock and crypto research has not beaten SPY by much. SPY is the fund that tracks the S&P 500, and it is the benchmark. The investigation is diagnosis only. It makes no recommendation about trading, strategies, deployments, the paper account or the account type.

It ran nine separate checks on the existing record, wrote one synthesis, ran an independent challenge against that synthesis and then collected two direction reviews, one from Claude Opus and one from Google's Gemini model Argon. This page is a route into the evidence, not another summary. Nine checks and two reviews is a lot of paper, so it picks three documents to read.

## Read in this order

1. [Findings in plain English](findings/2026-11-05-findings-plain-english.md): the reading edition of the synthesis. It answers the owner's seven questions and ranks the causes, with every term defined where it first appears.
2. [Consolidated direction review](reviews/2026-11-05-direction-reviews-consolidated.md): what the two frontier models recommend, which parts of the weaker review survived checking, the owner's open decisions and at most three next-step options.
3. [Reference sheet](reference-sheet.md): open it when a word stops you. It defines every abbreviation and label, and maps each check and round name to its file.

The other files are the sources of record. They keep their original wording and their lens, phase and gate numbers, because the numbers are part of the record. The reference sheet maps each name to its file.

## What was and was not done

- **The Optional Re-run List was not run.** It holds 29 items, ordered cheapest first, and sits in the [Main Evidence Round checkpoint](findings/2026-11-04-phase1-checkpoint.md). The owner decided on 2026-11-05 to run none of them. Every question they would settle is "unknown, re-run not run", not "passed".
- **The Argon review is unverified model output.** Of its 105 citations, 43 are broken or unsupported, and several specifics are invented. Read it only through the [verification](reviews/2026-11-05-argon-review-verification.md), which lists what survives. The consolidated review uses only the verified parts.
- **The price history of 19 symbols was rewritten by accident.** An accidental database refresh on 2026-11-04 rewrote the stored prices for 19 symbols. The owner chose to leave them as they are, so the investigation's price evidence on those symbols is not exactly reproducible. The Backtest Cleanliness check and the Verdict Trust check carry an erratum, and the checkpoint records the incident.

The investigation ran in three rounds, named here the way the reference sheet names them:

- **Early Trust Audit:** one unreviewed audit of the stock results.
- **Main Evidence Round:** an inventory of 144 results, then the nine checks.
- **Synthesis and Independent Challenge:** the write-up, then a second reader who argued the opposite case.

The Optional Re-run List was proposed after the Main Evidence Round and never ran. NO-GO means a test failed its pre-set standard. The tape is the project's stored price history.

## The folder map

Every file in the folder is listed once. The file names keep their numbers, so each name sits in a code span.

### Folder root

| File | What it is |
|---|---|
| `README.md` | This page. |
| [`reference-sheet.md`](reference-sheet.md) | Plain meanings for every abbreviation, check name, round name and label, with the file each one points to. |

### `findings/`

| File | What it is |
|---|---|
| [`2026-11-05-findings-plain-english.md`](findings/2026-11-05-findings-plain-english.md) | Reading edition of the synthesis, in plain words, with the caveats kept. |
| [`2026-11-04-root-cause-investigation.md`](findings/2026-11-04-root-cause-investigation.md) | The synthesis of the nine checks: the owner's seven questions answered, causes ranked, limits, owner decisions recorded and open, and the resolution of every challenge objection. |
| [`2026-11-04-root-cause-challenge.md`](findings/2026-11-04-root-cause-challenge.md) | The Independent Challenge: it argued the opposite case and checked 40 claims against the raw files. |
| [`2026-11-04-phase1-checkpoint.md`](findings/2026-11-04-phase1-checkpoint.md) | The Main Evidence Round checkpoint: its limits, the price-history incident and the Optional Re-run List that was not run. |
| [`2026-11-04-phase0-trust-audit.md`](findings/2026-11-04-phase0-trust-audit.md) | The Early Trust Audit: a single unreviewed audit of the stock results, stocks only, crypto not covered. |

### `lenses/`

The nine checks, one report each, except the Verdict Trust check, which has a method file and a report. Each report is followed by its JSON data file.

| File | What it is |
|---|---|
| [`2026-11-04-lens1-backtest-validity.md`](lenses/2026-11-04-lens1-backtest-validity.md) | Backtest Cleanliness check: what contamination remains in the stock prices and backtests after the five known fixes. |
| [`2026-11-04-lens2-sensitivity-spec.md`](lenses/2026-11-04-lens2-sensitivity-spec.md) | Verdict Trust check, method: the rules for classifying each result, fixed before any result was opened. |
| [`2026-11-04-lens2-verdict-audit.md`](lenses/2026-11-04-lens2-verdict-audit.md) | Verdict Trust check: which NO-GO and GO verdicts can be trusted, and for what size of effect. |
| [`2026-11-04-lens3-live-parity.md`](lenses/2026-11-04-lens3-live-parity.md) | Paper-versus-Backtest check: whether the live paper trader matches what the backtests measured. |
| [`2026-11-04-lens4-feasibility-and-ceiling.md`](lenses/2026-11-04-lens4-feasibility-and-ceiling.md) | Ceiling check: whether the target can be reached at all, and the measured maximum. |
| [`2026-11-04-lens5-coverage-gaps.md`](lenses/2026-11-04-lens5-coverage-gaps.md) | Coverage check: what the project's record contains and what it never tested. |
| [`2026-11-04-lens6-implementation-correctness.md`](lenses/2026-11-04-lens6-implementation-correctness.md) | Code Correctness check: whether the code does what the spec says, and whether live runs what was tested. |
| [`2026-11-04-lens7-research-accuracy.md`](lenses/2026-11-04-lens7-research-accuracy.md) | Report Accuracy check: whether the quoted numbers recompute from the raw files. |
| [`2026-11-04-lens8-crypto.md`](lenses/2026-11-04-lens8-crypto.md) | Crypto check: what was tried in crypto, what the evidence shows and what is unknown. |
| [`2026-11-04-lens9-power-and-regime.md`](lenses/2026-11-04-lens9-power-and-regime.md) | Test Power check: could the tests even see an effect, and how the answer changes by market regime. |
| [`_lens1_tape_state.json`](lenses/_lens1_tape_state.json) | Backtest Cleanliness check data: the price-history state behind each of the 144 results. |
| [`_lens2_sensitivity.json`](lenses/_lens2_sensitivity.json) | Verdict Trust check data: the classification rules and the class of every result. |
| [`_lens3_snapshot_gap.json`](lenses/_lens3_snapshot_gap.json) | Paper-versus-Backtest check data: the live-account snapshot and return comparison by account era. |
| [`_lens4_ceiling.json`](lenses/_lens4_ceiling.json) | Ceiling check data: the target, the soundness grades and the measured ceiling. |
| [`_lens5_coverage_matrix.json`](lenses/_lens5_coverage_matrix.json) | Coverage check data: the coverage class of all 144 results. |
| [`_lens6_replays.json`](lenses/_lens6_replays.json) | Code Correctness check data: hand replays of four strategies, one variant and one synthetic check of backtest fill rules. |
| [`_lens6_verdicts.json`](lenses/_lens6_verdicts.json) | Code Correctness check data: code verdicts for 90 results, 33 of them marked unverifiable. |
| [`_lens7_recompute.json`](lenses/_lens7_recompute.json) | Report Accuracy check data: the recomputed figures and the academic claims it checked. |
| [`_lens8_tape_checks.json`](lenses/_lens8_tape_checks.json) | Crypto check data: price-history checks, the buy-and-hold comparison and the annualization. |
| [`_lens9_power.json`](lenses/_lens9_power.json) | Test Power check data: the power result for each of the 144 results. Where a detectable effect could not be computed, the row says so. |

### `reviews/`

| File | What it is |
|---|---|
| [`2026-11-05-direction-reviews-consolidated.md`](reviews/2026-11-05-direction-reviews-consolidated.md) | Reading edition: an editorial comparison of the Opus and Argon direction reviews, with no new computation. |
| [`2026-11-05-senior-quant-direction-review.md`](reviews/2026-11-05-senior-quant-direction-review.md) | The Opus direction review: a ranked direction plus the owner's decision list. One reviewer, reliable as a single review. |
| [`2026-11-05-argon-synthesis-review.md`](reviews/2026-11-05-argon-synthesis-review.md) | The Argon direction review, kept unmodified and unverified as written. See the warning above. |
| [`2026-11-05-argon-review-verification.md`](reviews/2026-11-05-argon-review-verification.md) | Claude's line-by-line check of the Argon review: which citations and claims hold and which do not. |

### `records/`

| File | What it is |
|---|---|
| [`2026-11-04-investigation-inventory.md`](records/2026-11-04-investigation-inventory.md) | The 144 material results (every result that could change a verdict) that all nine checks cover, rendered from the inventory JSON. |
| [`2026-11-04-investigation-ledger.md`](records/2026-11-04-investigation-ledger.md) | A copy of the working ledger: task completions, review rounds, owner decisions and rulings. A working log, not a deliverable. |
| [`_investigation_inventory.json`](records/_investigation_inventory.json) | Source data for the inventory page, 144 rows. |
| [`_synthesis_inventory_map.json`](records/_synthesis_inventory_map.json) | Map from the synthesis tables to inventory rows. |
| [`_investigation_guard_baseline_2026-11-04.txt`](records/_investigation_guard_baseline_2026-11-04.txt) | A snapshot of the intraday-observer test copy, its timers, the live container and the main checkout, taken before the work started to show the live system was not touched. |

## Reading the raw files

The JSON and text files are historical evidence. Treat them as read-only. To reproduce the Verdict Trust check's classifications, run `scripts/lens2_build.py`. The other checks' scripts were not kept in the repository, and the synthesis (section 7) says what a fresh reader can and cannot reproduce.
